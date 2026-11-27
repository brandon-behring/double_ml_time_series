"""Development-only comparison against half the archived historical causal forecast."""

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from forward import reminder_rolling as reminders  # noqa: E402

from dml_ts.validation.effect_forecast import (  # noqa: E402
    forecast_comparison,
    ranking_precision,
    score,
)

ROLLING = HERE.parents[2] / "rolling"
METHODS = ("zero", "historical", "half_history", "dml")


def frozen_values(path, digest):
    """Read the old three-method record from disk and derive the fourth value separately."""
    records = score(path, digest, [])
    if len({r["origin"] for r in records}) != 1:
        raise ValueError("Expected one recorded origin")
    values = {r["method"]: r["forecast"] for r in records}
    adjusted = values["aipw"] if "aipw" in values else values["dml"]
    historical = values["historical"]
    return {
        "zero": values["zero"],
        "historical": historical,
        "half_history": None if historical is None else historical / 2,
        "dml": adjusted,
    }


def comparison(values, estimate, se, ci):
    """The interval conditions on the supplied forecast numbers for this issuance."""
    forecasts = []
    for method, forecast in values.items():
        error = None if forecast is None or estimate is None else forecast - estimate
        forecasts.append(
            {
                "method": method,
                "forecast": forecast,
                "error": error,
                "squared_error": None if error is None else error**2,
            }
        )
    pair = None
    direction = None
    dml, historical = values["dml"], values["historical"]
    if dml is not None and historical is not None:
        pair = forecast_comparison(dml, values["half_history"], estimate, ci)
        pair.update(ranking_precision(dml, values["half_history"]))
        if historical * dml < 0:
            direction = "reversed"
        elif dml == historical:
            direction = "unchanged"
        elif abs(dml) < abs(historical):
            direction = "toward_zero"
        else:
            direction = "away_from_zero"
    return {
        "estimate": estimate,
        "se": se,
        "ci95": ci,
        "forecasts": forecasts,
        "dml_vs_half_fixed": pair,
        "dml_move_from_history": direction,
    }


def reminder_replay(cache, archive=ROLLING):
    summary = json.loads((archive / "mybp-summary.json").read_text())
    draws = summary["multiplier_draws"]
    metadata, data_file = reminders.load_metadata(cache)
    enroll = metadata[[reminders.ID, reminders.ENROLL]].drop_duplicates().sort_values(reminders.ID)
    enroll = enroll.set_index(reminders.ID)[reminders.ENROLL]
    if not enroll.index.is_unique:
        raise ValueError("Multiple enrollment dates for one participant")
    weights = reminders.joint_person_weights(enroll, draws, summary["seed"])
    full = set(summary["full_comparison_origins"])
    issued = []
    losses = {method: [] for method in METHODS}
    for name in summary["origins"]:
        origin = pd.Timestamp(name)
        digest = summary["freeze_sha256"][name]
        values = frozen_values(archive / "mybp-freezes" / f"{name}.json", digest)
        early_rows = metadata.loc[metadata[reminders.ENROLL] + pd.Timedelta(days=98) < origin]
        target_rows = metadata.loc[
            metadata[reminders.ENROLL].ge(origin)
            & metadata[reminders.ENROLL].lt(origin + pd.DateOffset(months=3))
        ]
        early = reminders.summarize(
            early_rows, reminders.read_outcomes(data_file, early_rows.index)
        )
        later = reminders.summarize(
            target_rows, reminders.read_outcomes(data_file, target_rows.index)
        )
        if not early.people.intersection(later.people).empty:
            raise ValueError("Shared participant between earlier history and later cohort")
        later_weights = weights[:, enroll.index.get_indexer(later.people)]
        later_point = reminders.arm_difference(later, np.ones((1, len(later.people))))
        later_draws = reminders.arm_difference(later, later_weights)
        z = float(later_point[0]) if later_point is not None else None
        ci, se = None, None
        if (
            later_draws is not None
            and min(later.counts["activity_people"], later.counts["none_people"]) >= 2
        ):
            ci = np.quantile(later_draws, [0.025, 0.975]).tolist()
            se = float(later_draws.std(ddof=1))
        row = comparison(values, z, se, ci)
        row.update(
            origin=name,
            source_freeze_sha256=digest,
            train_people=len(early.people),
            later_people=len(later.people),
        )
        issued.append(row)
        if name not in full:
            continue
        # Refit every earlier learner with the same person's multiplicity at every date.
        early_weights = weights[:, enroll.index.get_indexer(early.people)]
        fits = reminders.fit(early, np.vstack([np.ones(len(early.people)), early_weights]))
        refits = {
            "zero": fits["zero"],
            "historical": fits["historical"],
            "half_history": 0.5 * fits["historical"],
            "dml": fits["aipw"],
        }
        for method in METHODS:
            np.testing.assert_allclose(refits[method][0], values[method], atol=1e-10, rtol=0)
            losses[method].append(
                np.r_[(values[method] - z) ** 2, (refits[method][1:] - later_draws) ** 2]
            )
    pooled = {method: np.mean(losses[method], axis=0) for method in METHODS}
    gains = {}
    for method in ("zero", "historical", "half_history"):
        advantage = pooled[method] - pooled["dml"]
        gains[method] = {
            "mean": float(advantage[0]),
            "ci95_refit": np.quantile(advantage[1:], [0.025, 0.975]).tolist(),
        }
    return {
        "schedule": issued,
        "full_comparison_origins": sorted(full),
        "mse": {k: float(v[0]) for k, v in pooled.items()},
        "dml_gains": gains,
        "bootstrap_reps": draws,
        "seed": summary["seed"],
    }


def jtpa_replay(archive=ROLLING):
    results = []
    for variant, prefix in (("original", "jtpa"), ("post_unblinding", "jtpa-corrected")):
        original = json.loads((archive / f"{prefix}-results.json").read_text())
        estimate, se = original["estimate"], original["se"]
        ci = [estimate + sign * NormalDist().inv_cdf(0.975) * se for sign in (-1, 1)]
        for lag, issuance in original["issuances"].items():
            digest = issuance["sha256"]
            values = frozen_values(archive / f"{prefix}-lag{lag}-frozen.json", digest)
            row = comparison(values, estimate, se, ci)
            row.update(
                origin="1989-01-01",
                reporting_lag_quarters=int(lag),
                source_freeze_sha256=digest,
                source_variant=variant,
            )
            results.append(row)
    return results


def run(cache):
    return {
        "use": "retrospective development comparison",
        "historical_fraction": 0.5,
        "reminders": reminder_replay(cache),
        "jtpa": jtpa_replay(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir), indent=2, allow_nan=False))
