"""Compare independent nuisance choices on historical paths and test frozen forward selectors."""

import argparse
import hashlib
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
BASE = import_module("docs.audits.2026-11-19_time_series_dml.paired.experiment")
PROTOCOL = HERE / "selection-protocol.json"
PROTOCOL_SHA = "7e8d832bbc6616abf0161aaf609decb526fa6703a93bdb0ad111c0eaa44b39ec"
FAMILIES = BASE.LEARNERS
PAIRS = tuple(itertools.product(FAMILIES, repeat=2))
RULES = (
    "fixed_trees",
    "outcome_mse_diagonal",
    "treatment_mse_diagonal",
    "separate_mse",
    "common_rloss_diagonal",
    "common_rloss_all",
)
DRAW_COLS = (
    "n rep regime split outcome treatment truth estimate iid hac outcome_mse "
    "treatment_mse score_linear score_product score_oracle"
).split()
SELECT_COLS = (
    "cohort rep regime block rule outcome treatment start end truth fitted frozen "
    "outcome_selection_loss treatment_selection_loss r_selection_loss"
).split()


def score_components(ey, ed, y, d, ell, m, truth, idx):
    """Decompose the unnormalised moment at date-specific truth exactly, before averaging."""
    v, u = d[idx] - m[idx], y[idx] - ell[idx] - truth[idx] * (d[idx] - m[idx])
    a = d[idx] - ed[idx] - m[idx]
    b = y[idx] - ey[idx] - ell[idx]
    linear = np.mean(v * (-b + truth[idx] * a) - a * u)
    product = np.mean(a * (b - truth[idx] * a))
    return linear, product, np.mean(v * u)


def run_selectors(rep, regime, score_dates, blocks, ys, ds, truth):
    """Score on a completed block, freeze choice, estimate on the subsequent block."""
    results = []
    cohort = "archive" if rep < 500 else "fresh"
    diagonal = tuple((name, name) for name in FAMILIES)

    def slope(outcome, treatment, idx):
        return BASE.theta_via_fwl(
            ys[outcome][idx], ds[treatment][idx], no_variation_msg="Degenerate effect"
        )[0]

    for k in (1, 2):
        validation = blocks[k][1]
        future = blocks[k + 1][1]
        earlier = score_dates[score_dates < validation[0]]
        history = score_dates[score_dates < future[0]]
        yloss = {name: np.mean(ys[name][validation] ** 2) for name in FAMILIES}
        dloss = {name: np.mean(ds[name][validation] ** 2) for name in FAMILIES}
        common_y, common_d = ys["trees"][validation], ds["trees"][validation]
        rloss = {}
        for outcome, treatment in PAIRS:
            est = slope(outcome, treatment, earlier)
            rloss[outcome, treatment] = np.mean((common_y - est * common_d) ** 2)
        y_winner = min(FAMILIES, key=yloss.get)
        d_winner = min(FAMILIES, key=dloss.get)
        choices = (
            ("trees", "trees"),
            (y_winner, y_winner),
            (d_winner, d_winner),
            (y_winner, d_winner),
            min(diagonal, key=rloss.get),
            min(PAIRS, key=rloss.get),
        )
        for rule, (outcome, treatment) in zip(RULES, choices, strict=True):
            results.append(
                (
                    cohort,
                    rep,
                    regime,
                    k + 1,
                    rule,
                    outcome,
                    treatment,
                    future[0],
                    future[-1],
                    truth[future].mean(),
                    slope(outcome, treatment, future),
                    slope(outcome, treatment, history),
                    yloss[outcome],
                    dloss[treatment],
                    rloss[outcome, treatment],
                )
            )
    return results


def run_path(job):
    n, rep, regime = job
    x, y, d, theta = BASE.generate(n, rep, regime)
    q = np.sin(2 * x[:, 0]) + 0.6 * x[:, 1] ** 2 + 2 * ((x[:, 0] > 0) & (x[:, 1] > 0))
    # These are the conditional means at that date even in the hidden-factor regime.
    ell = (1 + theta) * q
    schedules = BASE.folds(n)
    scored = np.concatenate([te for _, te in schedules["past"]])
    inputs = {name: BASE.learner_design(name, x) for name in FAMILIES}
    draws, choices = [], []
    splits = BASE.SPLITS if rep < 500 else ("past_buffer",)
    for split in splits:
        ys, ds = {}, {}
        for name, (features, model) in inputs.items():
            yh, dh = BASE.cross_fit_predictions(features, y, d, schedules[split], model, model)
            ys[name], ds[name] = y - yh, d - dh
        if rep < 500:
            for outcome, treatment in PAIRS:
                ey, ed = ys[outcome], ds[treatment]
                est, iid, hac = BASE.effect(ey[scored], ed[scored])
                terms = score_components(ey, ed, y, d, ell, q, theta, scored)
                actual_score = np.mean(ed[scored] * (ey[scored] - theta[scored] * ed[scored]))
                np.testing.assert_allclose(sum(terms), actual_score, atol=1e-9, rtol=1e-9)
                draws.append(
                    (
                        n,
                        rep,
                        regime,
                        split,
                        outcome,
                        treatment,
                        theta[scored].mean(),
                        est,
                        iid,
                        hac,
                        np.mean(ey[scored] ** 2),
                        np.mean(ed[scored] ** 2),
                        *terms,
                    )
                )
        if n == 1200 and split == "past_buffer":
            choices = run_selectors(rep, regime, scored, schedules[split], ys, ds, theta)
    return draws, choices


def diagonal_parity(draws):
    keys = ["rep", "n", "regime", "split", "learner"]
    numeric = ["truth", "estimate", "iid", "hac", "outcome_mse", "treatment_mse"]
    diagonal = draws.loc[draws.outcome.eq(draws.treatment)].rename(columns={"outcome": "learner"})
    reference = pd.read_csv(BASE.HERE / "draws.csv.gz")
    reference = reference[reference.rep.isin(draws.rep)]
    actual = diagonal.set_index(keys)[numeric].sort_index()
    expected = reference.set_index(keys)[numeric].sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=2e-10)
    return {
        "matched_estimates": len(actual),
        "max_abs_difference_by_column": (actual - expected).abs().max().to_dict(),
    }


def summaries(draws, selections):
    draws = draws.copy()
    draws["bias"] = draws.estimate - draws.truth
    draws["mse"] = draws.bias**2
    draws["cover_hac"] = (abs(draws.bias) <= BASE.Z * draws.hac).astype(float)
    draws["width_hac"] = 2 * BASE.Z * draws.hac
    draws["cover_iid"] = (abs(draws.bias) <= BASE.Z * draws.iid).astype(float)
    groups = ["n", "regime", "split", "outcome", "treatment"]
    group = draws.groupby(groups)
    cols = [
        "bias",
        "mse",
        "cover_hac",
        "width_hac",
        "cover_iid",
        "outcome_mse",
        "treatment_mse",
        "score_linear",
        "score_product",
        "score_oracle",
    ]
    tab = group[cols].mean().join(group[["bias", "mse", "cover_hac"]].sem().add_suffix("_mcse"))
    tab["rmse"] = np.sqrt(tab.mse)
    tab["paths"] = group.size()
    keys = ["n", "rep", "regime", "split"]
    ref = draws.query("outcome=='trees' and treatment=='trees'").set_index(keys)
    delta = draws.set_index(keys).join(ref[["mse", "bias", "cover_hac"]], rsuffix="_trees")
    for k in ("mse", "bias", "cover_hac"):
        delta[k + "_delta"] = delta[k] - delta[k + "_trees"]
    contrasts = ["mse_delta", "bias_delta", "cover_hac_delta"]
    by_pair = delta.reset_index().groupby(groups)[contrasts]
    paired = by_pair.mean().join(by_pair.sem().add_suffix("_mcse"))
    sel = selections.copy()
    for col in ("fitted", "frozen"):
        sel[col + "_mse"] = (sel[col] - sel.truth) ** 2
    keys = ["cohort", "rep", "regime", "block"]
    comparison = sel.set_index(keys).join(
        sel[sel.rule == "fixed_trees"].set_index(keys)[["fitted_mse", "frozen_mse"]],
        rsuffix="_trees",
    )
    for col in ("fitted_mse", "frozen_mse"):
        comparison[col + "_delta"] = comparison[col] - comparison[col + "_trees"]
    cols = ["fitted_mse", "frozen_mse", "fitted_mse_delta", "frozen_mse_delta"]
    per_path = comparison.reset_index().groupby(["cohort", "rep", "regime", "rule"])[cols].mean()
    by_rule = per_path.groupby(["cohort", "regime", "rule"])
    rules = by_rule.mean().join(by_rule.sem().add_suffix("_mcse"))
    rules["paths"] = by_rule.size()
    for col in ("fitted", "frozen"):
        rules[col + "_rmse"] = np.sqrt(rules[col + "_mse"])
    return tab.reset_index(), paired.reset_index(), rules.reset_index()


def one_role_differences(draws):
    """Paired changes in causal squared error when just one learner changes."""
    sq = (draws.estimate - draws.truth) ** 2
    wide = draws.assign(sq=sq).pivot(
        index=["n", "regime", "split", "rep"], columns=["outcome", "treatment"], values="sq"
    )
    rows = []
    for fixed in FAMILIES:
        for old, new in itertools.combinations(FAMILIES, 2):
            for role, delta in [
                ("outcome", wide[new, fixed] - wide[old, fixed]),
                ("treatment", wide[fixed, new] - wide[fixed, old]),
            ]:
                by_cell = delta.groupby(["n", "regime", "split"])
                table = by_cell.agg(mse_change="mean", mcse="sem", paths="size").reset_index()
                rows.append(
                    table.assign(
                        changed_role=role, fixed_family=fixed, previous=old, replacement=new
                    )
                )
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    if not 2 <= args.repetitions <= 500 or args.workers < 1:
        parser.error("Choose 2–500 paths per cell and positive workers.")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("Selection protocol differs from the pre-run frozen specification.")
    jobs = [(n, i, r) for i in range(args.repetitions) for n in BASE.SIZES for r in BASE.REGIMES]
    jobs.extend((1200, 500 + i, r) for i in range(args.repetitions) for r in BASE.REGIMES)
    workers = min(args.workers, os.cpu_count() or 1)
    start = perf_counter()
    paths, choices = [], []
    with (
        threadpool_limits(1),
        ProcessPoolExecutor(
            max_workers=workers, initializer=threadpool_limits, initargs=(1,)
        ) as pool,
    ):
        for i, (rows, sel) in enumerate(pool.map(run_path, jobs, chunksize=2), 1):
            paths.extend(rows)
            choices.extend(sel)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} paths", flush=True)
    draws = pd.DataFrame(paths, columns=DRAW_COLS)
    select = pd.DataFrame(choices, columns=SELECT_COLS)
    if not all(np.isfinite(x.select_dtypes("number")).all().all() for x in [draws, select]):
        raise ValueError("Nonfinite experiment output.")
    parity = diagonal_parity(draws)
    outputs = summaries(draws, select)
    args.output.mkdir(exist_ok=True, parents=True)
    for data, name in zip(
        outputs, ["factorial.csv", "paired-differences.csv", "selectors.csv"], strict=True
    ):
        BASE.save(data, args.output / name)
    BASE.save(draws, args.output / "factorial-draws.csv.gz")
    BASE.save(select, args.output / "selector-draws.csv.gz")
    BASE.save(one_role_differences(draws), args.output / "one-role-differences.csv")
    meta = {
        "frozen_protocol_sha256": PROTOCOL_SHA,
        "data_paths": len(jobs),
        "archived_factorial_estimates": len(draws),
        "selection_decisions": len(select),
        "workers": workers,
        "threads_per_worker": 1,
        "seconds": round(perf_counter() - start, 1),
        "diagonal_parity": parity,
    }
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
