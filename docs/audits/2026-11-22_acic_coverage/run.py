"""Score the official low-dimensional ACIC contest against its population-effect key."""

import argparse
import hashlib
import json
import re
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist
from time import monotonic
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

EVALUATE = import_module("docs.audits.2026-11-21_evaluation_methods.additional.acic").evaluate
OBSERVED_SHA = "c728f9b9fd23fc5f2ee7bcdbb23b84ae49c59106d3466c3c0955e9ef3bcbaba4"
TRUTH_SHA = "f8628ff1b4109c46a7906881f9c9f2d37e22fe75b07a08d183ab63d4e0e16645"
PROTOCOL_SHA = "80ad02545b6232c0ff6977a854016de11b84415f71e2765bc2a575c18e50dca4"
CRITICAL = NormalDist().inv_cdf(0.975)
_archive = None


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def initialize(archive):
    global _archive
    _archive = ZipFile(archive)
    threadpool_limits(1)


def fit_member(member):
    dataset = Path(member).stem
    with _archive.open(member) as stream:
        data = pd.read_csv(stream)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            result, cate = EVALUATE(data, detailed=True)
            result["standardization"] = np.mean(cate)
            if not np.isfinite(list(result.values())).all():
                raise ValueError("Nonfinite score or diagnostic")
            result.update(
                status="ok",
                outcome_type="binary" if data.Y.isin([0, 1]).all() else "continuous",
                ci_lower=result["aipw"] - CRITICAL * result["aipw_se_iid"],
                ci_upper=result["aipw"] + CRITICAL * result["aipw_se_iid"],
            )
        except (ValueError, ArithmeticError, AssertionError) as exc:
            result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "dataset": dataset,
        **result,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in captured),
        "total_warnings": len(captured),
    }


def holm(p_values):
    ordered = p_values.sort_values()
    adjusted = np.maximum.accumulate(ordered * np.arange(len(ordered), 0, -1))
    return pd.Series(np.minimum(adjusted, 1), index=ordered.index).reindex(p_values.index)


def rank_correlation(x, y):
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.rank().corr(y.rank())


def summarize_group(design, group):
    planned = len(group)
    valid = group[group.status == "ok"]
    hits = int(group.covered.sum())
    ci = binomtest(hits, planned).proportion_ci(method="exact")
    result = {
        "design": design,
        "planned": planned,
        "valid": len(valid),
        "covered": hits,
        "coverage": hits / planned,
        "coverage_mc_ci_low": ci.low,
        "coverage_mc_ci_high": ci.high,
        "undercoverage_p": binom.cdf(hits, planned, 0.95),
        "population_ate": group.population_ate.iloc[0],
        "files_with_convergence_warning": int((group.convergence_warnings > 0).sum()),
    }
    if valid.empty:
        return result
    if valid.outcome_type.nunique() != 1:
        raise ValueError(f"Inconsistent outcome type in design {design}")
    result["outcome_type"] = valid.outcome_type.iloc[0]
    for method in ("aipw", "standardization", "arm_difference"):
        errors = valid[method] - valid.population_ate
        result[method + "_bias"] = errors.mean()
        result[method + "_bias_mcse"] = errors.sem()
        result[method + "_rmse"] = np.sqrt(np.mean(errors**2))
    error = valid.aipw - valid.population_ate
    delta = error**2 - (valid.standardization - valid.population_ate) ** 2
    result.update(
        aipw_empirical_sd=valid.aipw.std(),
        aipw_mean_reported_se=valid.aipw_se_iid.mean(),
        aipw_mean_width=(valid.ci_upper - valid.ci_lower).mean(),
        plr_mean=valid.plr.mean(),
        plr_minus_pate_mean=(valid.plr - valid.population_ate).mean(),
        paired_mse_delta=delta.mean(),
        paired_mse_delta_mcse=delta.sem(),
        mean_clipped_fraction=(valid.probabilities_clipped / valid.n).mean(),
        mean_ess0_fraction=(valid.ess0 / (valid.n - valid.treated)).mean(),
        mean_ess1_fraction=(valid.ess1 / valid.treated).mean(),
        rho_outcome_skill_abs_error=rank_correlation(valid.outcome_skill, error.abs()),
        rho_treatment_skill_abs_error=rank_correlation(valid.treatment_skill, error.abs()),
    )
    for name in [
        "outcome_mse",
        "propensity_brier",
        "outcome_skill",
        "treatment_skill",
        "mu0_factual_mse",
        "mu1_factual_mse",
    ]:
        result["mean_" + name] = valid[name].mean()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed_zip", type=Path)
    parser.add_argument("truth_zip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    for path, expected in [
        (args.observed_zip, OBSERVED_SHA),
        (args.truth_zip, TRUTH_SHA),
        (Path(__file__).with_name("protocol.md"), PROTOCOL_SHA),
    ]:
        if sha(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
    with ZipFile(args.observed_zip) as observed:
        files = [
            m
            for m in observed.namelist()
            if re.fullmatch(r"low\d+\.csv", Path(m).name) and not m.startswith("__MACOSX/")
        ]
    names = [Path(m).stem for m in files]
    with ZipFile(args.truth_zip) as archive, archive.open("lowDim_trueATE.csv") as stream:
        key = pd.read_csv(stream).rename(
            columns={"filename": "dataset", "DGPid": "design", "trueATE": "population_ate"}
        )
    sizes = key.groupby("design", dropna=False).size()
    if (
        len(sizes) != 32
        or not (sizes == 100).all()
        or key.isna().any().any()
        or key.dataset.duplicated().any()
        or len(files) != 3200
        or len(set(names)) != 3200
        or set(names) != set(key.dataset)
        or not key.groupby("design").population_ate.nunique().eq(1).all()
    ):
        raise ValueError("Official file roster and ground-truth key do not reconcile")
    start = monotonic()
    files.sort(key=lambda m: int(Path(m).stem[3:]))
    with ProcessPoolExecutor(
        args.jobs, initializer=initialize, initargs=(args.observed_zip,)
    ) as pool:
        results = list(pool.map(fit_member, files, chunksize=8))
    draws = pd.DataFrame(results).merge(key, on="dataset", how="left", validate="one_to_one")
    for name, loss, baseline in [
        ("outcome_skill", "outcome_mse", "outcome_baseline_mse"),
        ("treatment_skill", "propensity_brier", "propensity_baseline_brier"),
    ]:
        draws[name] = 1 - draws[loss] / draws[baseline].where(draws[baseline] > 0)
    draws["covered"] = (
        draws.status.eq("ok") & draws.population_ate.between(draws.ci_lower, draws.ci_upper)
    ).astype(int)
    summary = pd.DataFrame([summarize_group(g, part) for g, part in draws.groupby("design")])
    summary["undercoverage_p_holm"] = holm(summary.undercoverage_p)
    summary["reject_undercoverage"] = (summary.undercoverage_p_holm < 0.05).astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in [("draws", draws), ("by-design", summary)]:
        table.to_csv(args.output / (name + ".csv"), index=False, float_format="%.16g")
    receipt = {
        "inputs_sha256": {"observed": OBSERVED_SHA, "organizer_results": TRUTH_SHA},
        "protocol_sha256": PROTOCOL_SHA,
        "jobs": args.jobs,
        "threads_per_job": 1,
        "elapsed_seconds": round(monotonic() - start, 2),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "files": len(draws),
        "status": dict(Counter(draws.status)),
        "designs": len(summary),
        "covered": int(draws.covered.sum()),
        "rejected_designs": int(summary.reject_undercoverage.sum()),
        "files_with_convergence_warning": int((draws.convergence_warnings > 0).sum()),
        "warnings": int(draws.total_warnings.sum()),
    }
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
