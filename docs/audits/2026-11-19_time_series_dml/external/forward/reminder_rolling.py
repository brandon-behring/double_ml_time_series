"""Forecast the proximal myBP reminder effect for seven fresh quarterly cohorts."""

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from temporalcv import TimeSeriesCrossValidator

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402
from forward.reminders import AGE, ENROLL, P_ACTIVITY, P_NONE  # noqa: E402
from sequential.mybp import DAY, FILES, ID, OUTCOME, PRIOR_STEPS  # noqa: E402

from dml_ts.validation.effect_forecast import freeze, score  # noqa: E402

ORIGINS = pd.date_range("2022-01-01", "2023-07-01", freq="QS")
METHODS = ["zero", "historical", "aipw"]
TOLERANCES = (0.025, 0.05, 0.10)
TARGET = "activity-prompt assignment minus no-prompt assignment per eligible decision"
UNIT = "log(steps + 0.5)"
HORIZON = "steps in the next hour on study days 14 through 97"
SOURCE_CSV = "Data_MRT/MRT Data.csv"


def load_metadata(cache):
    paths = {
        key: fetch(f"{base}/{name}", cache, f"mybp-{name}", digest)
        for key, (base, name, digest) in FILES.items()
    }
    flags = ["IsDiceRollTypeActivity", "IsDiceRollTypeDiet", "IsDiceRollSuccess"]
    with ZipFile(paths["trials"]) as archive, archive.open(SOURCE_CSV) as source:
        decisions = pd.read_csv(
            source,
            usecols=[
                ID,
                DAY,
                PRIOR_STEPS,
                *flags,
                "DiceRollType",
                "IsPresent_HeartRate_30minBefore",
                "Phase",
            ],
            dtype={ID: str},
            low_memory=False,
        )
    # Keep source line numbers so held-out outcome fields can be skipped by the CSV parser.
    decisions["source_row"] = np.arange(len(decisions))
    demographics = pd.read_csv(
        paths["demographics"], dtype={ID: str}, usecols=[ID, AGE, ENROLL, "Gender", "Race_5cat"]
    )
    baseline = pd.read_csv(paths["baseline"], dtype={ID: str}, usecols=[ID, "base_step"])
    ids = pd.read_csv(paths["cohort"], dtype={ID: str})[ID]
    eligible = decisions.loc[
        decisions[ID].isin(ids)
        & decisions[DAY].between(14, 97)
        & decisions["IsPresent_HeartRate_30minBefore"].eq(1)
        & decisions.Phase.eq("active")
    ]
    eligible = eligible.merge(demographics, on=ID, validate="many_to_one")
    eligible = eligible.merge(baseline, on=ID, validate="many_to_one").dropna()
    eligible[ENROLL] = pd.to_datetime(eligible[ENROLL], format="%m/%d/%Y")
    eligible["activity"] = eligible.IsDiceRollTypeActivity * eligible.IsDiceRollSuccess
    eligible["none"] = 1 - eligible.IsDiceRollSuccess
    return eligible.set_index("source_row").sort_index(), paths["trials"]


def read_outcomes(archive_path, indices):
    if len(indices) == 0:
        return pd.Series(dtype=float, name="y")
    if not indices.is_unique or not indices.is_monotonic_increasing:
        raise ValueError("Outcome request requires unique source rows in archive order")
    keep = set(indices + 1)
    with ZipFile(archive_path) as archive, archive.open(SOURCE_CSV) as source:
        rows = pd.read_csv(source, usecols=[OUTCOME], skiprows=lambda i: i != 0 and i not in keep)
    if len(rows) != len(indices) or (rows[OUTCOME].dropna() < 0).any():
        raise ValueError("Invalid myBP outcome data or row mapping")
    return pd.Series(np.log(rows[OUTCOME].to_numpy() + 0.5), index=indices, name="y")


@dataclass
class Sample:
    people: pd.Index
    decisions: np.ndarray
    x_sum: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    arm_y: np.ndarray
    counts: dict


def summarize(frame, outcomes):
    """Retain cluster-wise sufficient statistics to refit all ridge models in every draw."""
    if not outcomes.index.equals(frame.index):
        raise ValueError("Outcomes and requested decision rows differ")
    missing = outcomes.isna()
    counts = {
        "eligible": len(frame),
        "missing": int(missing.sum()),
        "missing_activity": int((missing & frame.activity.eq(1)).sum()),
        "missing_none": int((missing & frame.none.eq(1)).sum()),
    }
    data = frame.loc[~missing]
    y = outcomes.loc[~missing].to_numpy()
    codes, people = pd.factorize(data[ID], sort=True)
    n = len(people)
    x = np.column_stack(
        (
            np.ones(len(data)),
            (data[AGE] - 60) / 10,
            (data[DAY] - 55) / 30,
            np.log1p(data.base_step),
            np.log1p(data[PRIOR_STEPS]),
        )
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite pre-decision covariates and observed outcomes")
    d = x.shape[1]
    xx, xy = np.zeros((n, 2, d, d)), np.zeros((n, 2, d))
    arm_y, x_sum = np.zeros((n, 2)), np.zeros((n, d))
    np.add.at(x_sum, codes, x)
    for arm, column in enumerate(["activity", "none"]):
        sel = data[column].eq(1).to_numpy()
        xa, ya = x[sel], y[sel]
        np.add.at(xx[:, arm], codes[sel], xa[:, :, None] * xa[:, None, :])
        np.add.at(xy[:, arm], codes[sel], xa * ya[:, None])
        np.add.at(arm_y[:, arm], codes[sel], ya)
        counts[column] = int(sel.sum())
        counts[f"{column}_people"] = int(np.unique(codes[sel]).size)
    counts.update(people=n, observed=len(data))
    return Sample(pd.Index(people), np.bincount(codes, minlength=n), x_sum, xx, xy, arm_y, counts)


def arm_difference(sample, weights):
    counts = weights @ sample.xx[:, :, 0, 0]
    if np.any(counts <= 0):
        return None
    means = (weights @ sample.arm_y) / counts
    return means[:, 0] - means[:, 1]


def fold_membership(people):
    # Assignment uses participant ID only and never looks at outcomes or future enrollment.
    order = sorted(
        range(len(people)), key=lambda i: hashlib.sha256(str(people[i]).encode()).digest()
    )
    folds = np.empty(len(people), dtype=int)
    folds[order] = np.arange(len(people)) % min(5, len(people))
    return folds


def crossfit(sample, weights):
    """Known-probability AIPW, averaged over every observed historical decision."""
    if len(sample.people) < 2:
        return None
    folds = fold_membership(sample.people)
    n_features = sample.x_sum.shape[1]
    penalty = np.eye(n_features)
    penalty[0, 0] = 0
    score_sum = np.zeros(weights.shape[0])
    for fold in np.unique(folds):
        train, valid = folds != fold, folds == fold
        for arm, (p, sign) in enumerate(((P_ACTIVITY, 1), (P_NONE, -1))):
            cross = sample.xx[train, arm]
            if cross[:, 0, 0].sum() == 0:
                return None
            gram = (weights[:, train] @ cross.reshape(train.sum(), -1)).reshape(
                -1, n_features, n_features
            )
            rhs = weights[:, train] @ sample.xy[train, arm]
            beta = np.linalg.solve(gram + penalty, rhs[:, :, None])[:, :, 0]
            # The intercept row of X'X contains the arm-specific feature sums.
            adjusted_x = sample.x_sum[valid] - sample.xx[valid, arm, 0, :] / p
            score_sum += sign * (
                ((weights[:, valid] @ adjusted_x) * beta).sum(axis=1)
                + (weights[:, valid] @ sample.arm_y[valid, arm]) / p
            )
    return score_sum / (weights @ sample.decisions)


def fit(sample, weights):
    return {
        "zero": np.zeros(len(weights)),
        "historical": arm_difference(sample, weights),
        "aipw": crossfit(sample, weights),
    }


def protocol(origin):
    return {
        "target": TARGET,
        "unit": UNIT,
        "horizon": HORIZON,
        "origins": [origin],
        "methods": METHODS,
        "maturity": "enrollment + 98 days strictly before origin",
        "evaluation": "new enrollments in the calendar quarter beginning on the origin",
        "assignment_probabilities": {"activity": P_ACTIVITY, "none": P_NONE},
        "learner": "Ridge alpha=1, unpenalized intercept, 5 whole-person folds",
        "features": [
            "(age-60)/10",
            "(study_day-55)/30",
            "log1p(base_step)",
            "log1p(prior_30min_steps)",
        ],
        "tolerances": list(TOLERANCES),
        "input_sha256": {k: v[2] for k, v in FILES.items()},
    }


def forecast_rows(origin, estimates):
    rows = []
    for method in METHODS:
        values = estimates[method]
        row = {
            "origin": origin,
            "method": method,
            "value": float(np.round(values[0], 12)) if values is not None else None,
        }
        if values is None:
            row["reason"] = "insufficient mature training participants or arm support"
        rows.append(row)
    return rows


def joint_person_weights(enrollment, draws, seed):
    """Resample whole people with one shared weight across origins, fixed within-quarter totals."""
    rng = np.random.default_rng(seed)
    weights = rng.exponential(size=(draws, len(enrollment)))
    quarters = pd.DatetimeIndex(enrollment).to_period("Q")
    for quarter in quarters.unique():
        mask = quarters == quarter
        weights[:, mask] /= weights[:, mask].mean(axis=1)[:, None]
    return weights


def replay(metadata, outcome_reader, output, origins=ORIGINS, draws=999, seed=20261120):
    """Historical operator replay; read only mature labels until each origin has been frozen."""
    all_ids = metadata[[ID, ENROLL]].drop_duplicates().sort_values(ID).set_index(ID)[ENROLL]
    if not all_ids.index.is_unique:
        raise ValueError("A participant cannot have two enrollment dates")
    weights = joint_person_weights(all_ids, draws, seed)

    rows, errors, hashes = [], {}, {}
    for start in origins:
        origin = start.date().isoformat()
        # Splitting a calendar grid makes the gap 98 whole days even across empty dates.
        stop = start + pd.DateOffset(months=3)
        first_day = min(start - pd.Timedelta(days=99), metadata[ENROLL].min())
        days = pd.date_range(first_day, stop, inclusive="left")
        splitter = TimeSeriesCrossValidator(
            n_splits=1, min_train_size=1, gap=98, test_size=(stop - start).days
        )
        earlier_days, later_days = next(splitter.split(days))
        training = metadata.loc[metadata[ENROLL].isin(days[earlier_days])]
        train = summarize(training, outcome_reader(training.index))
        frozen = forecast_rows(origin, fit(train, np.ones((1, len(train.people)))))
        path = output / f"{origin}.json"
        digest = freeze(path, protocol(origin), frozen)
        hashes[origin] = digest

        later = metadata.loc[metadata[ENROLL].isin(days[later_days])]
        target = summarize(later, outcome_reader(later.index))
        if len(train.people.intersection(target.people)):
            raise ValueError("A participant appears in both training and evaluation")
        estimate = arm_difference(target, np.ones((1, len(target.people))))
        reference = float(estimate[0]) if estimate is not None else None
        observed = {
            "origin": origin,
            "target": TARGET,
            "unit": UNIT,
            "horizon": HORIZON,
            "estimate": reference,
        }
        if reference is None:
            observed["reason"] = "one or both evaluation arms absent"
        scored = score(path, digest, [observed])

        w_train = weights[:, all_ids.index.get_indexer(train.people)]
        w_later = weights[:, all_ids.index.get_indexer(target.people)]
        boot = fit(train, w_train)
        ref_draws = arm_difference(target, w_later)
        dates = {}
        for row in scored:
            method, error = row["method"], row["signed_error"]
            estimable = min(
                target.counts["activity_people"], target.counts["none_people"]
            ) >= 2 and (
                method == "zero"
                or min(train.counts["activity_people"], train.counts["none_people"]) >= 2
            )
            if error is not None and estimable:
                dates[method] = np.r_[error, boot[method] - ref_draws]
            row["later_effect"] = row.pop("estimate")
            row["freeze_sha256"] = digest
            row["reason"] = row["reason"] or ""
            row.update(
                {
                    prefix + key: val
                    for prefix, counts in (("train_", train.counts), ("later_", target.counts))
                    for key, val in counts.items()
                }
            )
            if error is not None and target.counts["people"] < 2:
                row["reason"] = "point only: one later participant"
            elif error is not None and not estimable:
                row["reason"] = "point only: fewer than two participants in an arm"
            interval = None
            if method in dates:
                v = dates[method][1:]
                row["ci95_low"], row["ci95_high"] = np.quantile(v, [0.025, 0.975]).tolist()
                interval = np.quantile(v, [0.05, 0.95])
                row["ci90_low"], row["ci90_high"] = interval.tolist()
            for delta in TOLERANCES:
                row[f"within_{delta:g}"] = (
                    None
                    if interval is None
                    else bool(interval[0] >= -delta and interval[1] <= delta)
                )
            rows.append(row)
        errors[origin] = dates

    full = [o for o, date in errors.items() if set(date) == set(METHODS)]
    gains = {}
    if full:
        for name in ("zero", "historical"):
            paired = np.mean([errors[o][name] ** 2 - errors[o]["aipw"] ** 2 for o in full], axis=0)
            gains[name] = {
                "mean": float(paired[0]),
                "ci95": np.quantile(paired[1:], [0.025, 0.975]).tolist(),
            }
    return rows, {
        "origins": list(hashes),
        "full_comparison_origins": full,
        "multiplier_draws": draws,
        "seed": seed,
        "freeze_sha256": hashes,
        "aipw_gain": gains,
    }


def render(rows, summary):
    def fmt(v):
        return "—" if v is None else f"{v:+.4f}"

    lines = [
        "# Forecasting reminder effects for new participants",
        "",
        "At each quarter start we forecast the effect of assigning an activity prompt versus no prompt "
        "on log(next-hour steps + 0.5) during study days 14–97 among new enrollees. Every eligible "
        "observed decision has equal weight. All seven enrollment quarters appear below. Four support "
        "both historical methods and between-person error intervals. The final quarter has one new "
        "participant. These are retrospective operator forecasts using a public release published "
        "after the study. [Original trial](https://doi.org/10.2196/78218) · "
        "[Pinned decisions and codebook](https://github.com/WIRED-L-um/MRT_myBPmyLife/tree/fa86c0d44ced3876f4a7351bc2d475fc4249342e)",
        "",
        "| Forecast issued | Earlier / new people | Earlier arm difference | Earlier AIPW | "
        "Later randomized effect | AIPW error and 95% interval |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for origin in summary["origins"]:
        methods = {r["method"]: r for r in rows if r["origin"] == origin}
        row = methods["aipw"]
        bounds = ""
        if row.get("ci95_low") is not None:
            bounds = f" [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
        lines.append(
            f"| {origin} | {row['train_people']} / {row['later_people']} | "
            f"{fmt(methods['historical']['forecast'])} | {fmt(row['forecast'])} | "
            f"{fmt(row['later_effect'])} | {fmt(row['signed_error'])}{bounds} |"
        )
    lines.extend(
        [
            "",
            "The zero forecast can be issued at every origin. Both other forecasts require mature "
            "earlier outcomes. At each of the four fully evaluable dates, the AIPW forecast and "
            "later point estimate fell on opposite sides of zero. Each corresponding signed-error "
            "interval includes zero. The July 2023 point describes the one available participant. "
            "The signed error is the frozen forecast minus the later randomized contrast.",
            "",
            "## Incremental forecast accuracy",
            "",
            "We gave each of the four fully evaluable quarters equal weight. A positive difference "
            "in squared errors favors AIPW. The intervals propagate uncertainty in the people sampled "
            "in these particular quarters. Four dates provide limited evidence about new calendar periods.",
            "",
            "| Comparator | Mean squared-error gain of AIPW (log units squared) | 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, values in summary["aipw_gain"].items():
        label = {"zero": "Zero", "historical": "Earlier arm difference"}[name]
        lo, hi = values["ci95"]
        lines.append(f"| {label} | {values['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(
        [
            "",
            "For tolerance sensitivity we ask whether a pointwise 90% signed-error interval fits "
            "entirely within each symmetric margin. Each cell counts qualifying quarters among the four "
            "shared evaluation dates. These are illustrative accuracy margins in log units. The trial "
            "supplies no monetary break-even threshold for acting on the forecast.",
            "",
            "| Method | ±0.025 | ±0.05 | ±0.10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        valid = [
            r
            for r in rows
            if r["method"] == method and r["origin"] in summary["full_comparison_origins"]
        ]
        counts = [sum(r[f"within_{t:g}"] for r in valid) for t in TOLERANCES]
        lines.append(f"| {method} | " + " | ".join(f"{v}/{len(valid)}" for v in counts) + " |")
    lines.extend(
        [
            "",
            "## How the estimates were produced",
            "",
            "The temporalcv splitter operates on a complete calendar-day grid with a 98-day gap. "
            "Enrollment date plus 98 days must precede the forecast date for every training participant. "
            "Evaluation uses new people enrolled in the next three months. The archive gives dates of "
            "enrollment and relative study days. The one-day allowance after day 97 covers a final "
            "next-hour outcome. Historical ingestion dates are unavailable. Eligibility requires the "
            "trial’s active phase, pre-decision heart rate, valid assignment and baseline covariates. "
            "The CSV reports outcomes missing after these filters by period and arm. Complete-case "
            "causal interpretation assumes that assignment did not selectively affect observation.",
            "",
            "The earlier and later randomized contrasts each subtract the observed no-prompt arm mean "
            "from the activity-prompt arm mean. Their assignment probabilities are 0.75 and 0.125, "
            "respectively. The later contrast is a noisy, self-normalized estimate with random denominators. "
            "Paired loss gains describe the comparison against that estimate. A third arm assigns diet prompts.",
            "",
            "DML uses the known randomization probabilities and separate ridge regressions for the two "
            "arms. Age, study day, baseline steps and steps in the prior 30 minutes enter with fixed "
            "scaling. Each earlier participant stays in one of five folds. Each residual correction "
            "uses predictions trained on the other folds among the mature history. The AIPW score "
            "averages over all earlier eligible complete decisions, including diet-arm decisions. "
            "This forecasts the earlier population's marginal effect for the new cohort without "
            "using its covariates. The original single-cutoff analysis pools multiple later quarters. "
            "[Original split and eligibility](../forward/reminders.md)",
            "",
            f"The {summary['multiplier_draws']} shared bootstrap draws use one exponential weight per "
            "person, normalized within enrollment quarter and reused across dates. Each draw refits both "
            "arm regressions, recomputes the earlier and later effects, and then recomputes the paired "
            "losses. Calendar periods and fold membership stay fixed. Follow-up overlaps calendar "
            "weeks across quarterly enrollment cohorts. These intervals condition on the observed "
            "calendar and do not propagate shared calendar shocks to a new year. The reported "
            "percentile intervals are pointwise.",
            "",
            "## Reproduce",
            "",
            "`uv run --locked python docs/audits/2026-11-19_time_series_dml/external/forward/reminder_rolling.py "
            "--cache-dir /tmp/dml-external-bench-cache --output-dir docs/audits/2026-11-19_time_series_dml/rolling`",
            "",
            "[All 21 method-date rows, arm counts, missing outcomes and intervals](mybp-origins.csv) · "
            "[Paired summary and freeze digests](mybp-summary.json) · [Frozen forecasts](mybp-freezes/) · "
            "[Source and scoring code](../external/forward/reminder_rolling.py)",
            "",
        ]
    )
    return "\n".join(lines)


def run(cache, output=None):
    metadata, archive = load_metadata(cache)
    with tempfile.TemporaryDirectory() as temp:
        freezes = Path(temp) if output is None else output / "mybp-freezes"
        rows, summary = replay(metadata, lambda ix: read_outcomes(archive, ix), freezes)
    if output is not None:
        pd.DataFrame(rows).to_csv(output / "mybp-origins.csv", index=False, float_format="%.12g")
        (output / "mybp-summary.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2) + "\n"
        )
        (output / "reminders.md").write_text(render(rows, summary))
    return {"summary": summary, "origins": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir, args.output_dir), indent=2, allow_nan=False))
