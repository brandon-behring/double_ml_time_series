"""Exploratory propensity substitution on the 50 even-ranked Design 30 evaluation files.

The odd-ranked 50 sibling simulations estimate row propensities from treatment alone.
These are estimated reference probabilities conditional on identical covariate rows.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def fit_evaluation(data, reference_p):
    x = data.drop(columns=["Y", "A"]).to_numpy(dtype=float)
    y = data.Y.to_numpy(dtype=float)
    a = data.A.to_numpy(dtype=float)
    p = np.empty_like(y)
    mu = np.empty((2, len(y)))
    for train, held in KFold(5, shuffle=True, random_state=42).split(x):
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
        p[held] = logit.fit(x[train], a[train]).predict_proba(x[held])[:, 1]
        for arm in (0, 1):
            group = train[a[train] == arm]
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            mu[arm, held] = ridge.fit(x[group], y[group]).predict(x[held])
    result = {}
    for name, prob in [("ordinary", p), ("reference", reference_p)]:
        bounded = np.clip(prob, 0.01, 0.99)
        score = mu[1] - mu[0] + a * (y - mu[1]) / bounded - (1 - a) * (y - mu[0]) / (1 - bounded)
        result[name + "_aipw"] = score.mean()
        result[name + "_brier"] = np.mean((a - prob) ** 2)
        result[name + "_bounded_brier"] = np.mean((a - bounded) ** 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed_zip", type=Path)
    parser.add_argument("truth_zip", type=Path)
    parser.add_argument(
        "study", type=Path, help="Completed ACIC study directory containing draws.csv"
    )
    args = parser.parse_args()
    expected_hashes = json.loads((args.study / "run.json").read_text())["inputs_sha256"]
    for path, key in [(args.observed_zip, "observed"), (args.truth_zip, "organizer_results")]:
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected_hashes[key]:
                raise ValueError(f"Archive checksum mismatch for {path}")
    with ZipFile(args.truth_zip) as archive, archive.open("lowDim_trueATE.csv") as f:
        key = pd.read_csv(f).query("DGPid == 30")
    if len(key) != 100 or key.filename.nunique() != 100 or key.trueATE.nunique() != 1:
        raise ValueError("Unexpected Design 30 roster or truth key")
    roster = sorted(key.filename, key=lambda name: int(name.removeprefix("low")))
    reference, evaluation = roster[::2], roster[1::2]
    with ZipFile(args.observed_zip) as archive, threadpool_limits(1):
        with archive.open(reference[0] + ".csv") as f:
            x = pd.read_csv(f, usecols=lambda col: col not in {"Y", "A"})

        def read_member(name, include_outcome):
            with archive.open(name + ".csv") as stream:
                data = pd.read_csv(
                    stream, usecols=None if include_outcome else lambda col: col != "Y"
                )
            if not data.drop(columns=["Y", "A"], errors="ignore").equals(x):
                raise ValueError(f"Covariate values, columns or row order changed in {name}")
            if not data.A.isin([0, 1]).all():
                raise ValueError(f"Unexpected treatment code in {name}")
            return data

        treated = np.zeros(len(x))
        for name in reference:
            treated += read_member(name, False).A.to_numpy()
        reference_p = (treated + 0.5) / (len(reference) + 1)
        rows = []
        for name in evaluation:
            rows.append({"dataset": name, **fit_evaluation(read_member(name, True), reference_p)})
    results = pd.DataFrame(rows).set_index("dataset")
    original = pd.read_csv(args.study / "draws.csv").set_index("dataset").loc[evaluation]
    np.testing.assert_allclose(results.ordinary_aipw, original.aipw, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        results.ordinary_brier, original.propensity_brier, rtol=1e-9, atol=1e-9
    )
    truth = float(key.trueATE.iloc[0])
    summary = {
        "design": 30,
        "analysis": "Exploratory, selected after the primary coverage study",
        "population_ate": truth,
        "reference_files": len(reference),
        "evaluation_files": len(evaluation),
        "reference_rule": "Odd ranks (1-based) after sorting the 100 filenames by numeric suffix",
        "reference_estimator": (
            "The estimated assignment probability is (row treatment count + 0.5) / 51. "
            "It requires treatment assignments from 50 sibling simulations."
        ),
        "rows_per_file": len(x),
        "covariates": x.shape[1],
        "identical_X_files_verified": len(roster),
        "reference_rows_clipped": int(((reference_p < 0.01) | (reference_p > 0.99)).sum()),
        "mcse_scope": (
            "Paired SD across 50 evaluation files divided by sqrt(50), conditional on "
            "the observed set of reference assignments and fixed X."
        ),
    }
    results["population_ate"] = truth
    for name in ("ordinary", "reference"):
        err = results[name + "_aipw"] - truth
        results[name + "_error"] = err
        summary[name] = {
            "mean_aipw": results[name + "_aipw"].mean(),
            "bias": err.mean(),
            "rmse": np.sqrt(np.mean(err**2)),
            "mean_brier": results[name + "_brier"].mean(),
            "mean_bounded_brier": results[name + "_bounded_brier"].mean(),
        }
    results["reference_minus_ordinary"] = results.reference_aipw - results.ordinary_aipw
    results["squared_error_change"] = results.reference_error**2 - results.ordinary_error**2
    results["brier_change"] = results.reference_brier - results.ordinary_brier
    for metric in ("reference_minus_ordinary", "squared_error_change", "brier_change"):
        summary[metric] = {
            "mean": results[metric].mean(),
            "conditional_paired_mcse": results[metric].sem(),
        }
    summary["evaluation_files_smaller_absolute_error"] = int(
        (results.reference_error.abs() < results.ordinary_error.abs()).sum()
    )
    results.to_csv(args.study / "reference-propensity.csv", float_format="%.16g")
    out = json.dumps(summary, indent=2, allow_nan=False)
    (args.study / "reference-propensity.json").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
