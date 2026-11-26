"""Forecast the effect of an activity prompt in a later randomized enrollment cohort."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from temporalcv import TimeSeriesCrossValidator

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402
from sequential.mybp import DAY, FILES, ID, OUTCOME, PRIOR_STEPS  # noqa: E402

from dml_ts.dml.fwl import fwl_estimate  # noqa: E402

CUTOFF = pd.Timestamp("2023-01-01")
ENROLL = "EnrollmentDate_ET"
AGE = "AgeAtConsent_years_calc_int"
P_ACTIVITY, P_NONE = 0.125, 0.75
PROBABILITIES = np.array([P_ACTIVITY, P_NONE])


def load(cache):
    paths = {
        key: fetch(f"{base}/{name}", cache, f"mybp-{name}", digest)
        for key, (base, name, digest) in FILES.items()
    }
    flags = ["IsDiceRollTypeActivity", "IsDiceRollTypeDiet", "IsDiceRollSuccess"]
    cols = [
        ID,
        DAY,
        OUTCOME,
        PRIOR_STEPS,
        "DiceRollType",
        *flags,
        "IsPresent_HeartRate_30minBefore",
        "Phase",
        "Overlaps_60min_json",
    ]
    with ZipFile(paths["trials"]) as archive, archive.open("Data_MRT/MRT Data.csv") as f:
        data = pd.read_csv(f, usecols=cols, dtype={ID: str}, low_memory=False)
    demographics = pd.read_csv(
        paths["demographics"], usecols=[ID, AGE, ENROLL, "Gender", "Race_5cat"], dtype={ID: str}
    )
    baseline = pd.read_csv(paths["baseline"], dtype={ID: str})
    ids = pd.read_csv(paths["cohort"], dtype={ID: str})[ID]
    data = (
        data[data[ID].isin(ids)]
        .merge(demographics, on=ID, validate="many_to_one")
        .merge(baseline[[ID, "base_step"]], on=ID, validate="many_to_one")
    )
    data = data.dropna(
        subset=[
            DAY,
            OUTCOME,
            PRIOR_STEPS,
            *flags,
            "DiceRollType",
            AGE,
            ENROLL,
            "Gender",
            "Race_5cat",
            "base_step",
        ]
    )
    data = data.loc[data["IsPresent_HeartRate_30minBefore"].eq(1) & data.Phase.eq("active")].copy()
    data[ENROLL] = pd.to_datetime(data[ENROLL], format="%m/%d/%Y")
    data["activity"] = data["IsDiceRollTypeActivity"] * data["IsDiceRollSuccess"]
    data["none"] = 1 - data["IsDiceRollSuccess"]
    data["log_steps"] = np.log(data[OUTCOME] + 0.5)
    return data


def cohorts(data, last_day):
    """Split enrollment dates as units and require a fully observable earlier window."""
    dates = np.sort(data[ENROLL].unique())
    test_size = len(dates) - np.searchsorted(dates, CUTOFF.to_datetime64())
    if test_size in (0, len(dates)):
        raise ValueError("Expected enrollment on both sides of the calendar cutoff")
    splitter = TimeSeriesCrossValidator(n_splits=1, min_train_size=1, test_size=test_size)
    early_ix, late_ix = next(splitter.split(dates))
    # Next-hour steps for the final evening decision could fall on the next day.
    eligible = data[data[DAY].between(14, last_day)]
    earlier = eligible[
        eligible[ENROLL].isin(dates[early_ix])
        & (eligible[ENROLL] + pd.to_timedelta(last_day + 1, unit="D") < CUTOFF)
    ]
    later = eligible[eligible[ENROLL].isin(dates[late_ix])]
    if set(earlier[ID]).intersection(later[ID]):
        raise ValueError("Participant crosses the calendar-cohort boundary")
    return earlier, later


def person_totals(frame):
    """Compute sufficient sums per person for design-based means and cluster resampling."""
    cols = frame[[ID, AGE, "activity", "none", "log_steps"]].copy()
    for arm in ("activity", "none"):
        cols[f"{arm}_outcome"] = cols[arm] * cols.log_steps
    cols["decisions"] = 1
    columns = ["activity", "none", "activity_outcome", "none_outcome", "decisions"]
    grouped = cols.groupby(ID, sort=True)
    if (grouped[AGE].nunique() != 1).any():
        raise ValueError("Age at enrollment must be constant within person")
    return grouped[columns].sum().to_numpy(), grouped[AGE].first().to_numpy() > 65


def effect(totals):
    """Hajek means for randomized activity versus no prompt, on the log outcome scale."""
    counts, sums = totals[..., :2] / PROBABILITIES, totals[..., 2:4] / PROBABILITIES
    if np.any(counts <= 0):
        raise ValueError("An estimation stratum has no support in one arm")
    means = sums / counts
    return means[..., 0] - means[..., 1]


def forecast(early, early_age, early_weights, late_age, late_weights):
    """The frozen prediction uses only the early sample. The second uses later enrollment ages."""
    primary = effect(early_weights @ early)
    early_n = [early_weights[:, early_age == flag].sum(axis=1) for flag in (False, True)]
    late_n = [late_weights[:, late_age == flag].sum(axis=1) for flag in (False, True)]
    standardized_sum = np.zeros_like(primary)
    exposure_sum = np.zeros_like(primary)
    for i, flag in enumerate((False, True)):
        totals = early_weights[:, early_age == flag] @ early[early_age == flag]
        # Project late decision counts from the early decision frequency in that age group.
        projected = late_n[i] * totals[:, 4] / early_n[i]
        standardized_sum += effect(totals) * projected
        exposure_sum += projected
    return primary, standardized_sum / exposure_sum


def interval(point, draws):
    return {"estimate": float(point), "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def later_augmented(early, late, early_age, late_age, early_weights, late_weights):
    """One-step score with age-specific means trained only on the earlier cohort."""
    arm_sums = np.zeros((len(early_weights), 2))
    n_decisions = np.zeros(len(early_weights))
    for flag in (False, True):
        train = early_weights[:, early_age == flag] @ early[early_age == flag]
        test = late_weights[:, late_age == flag] @ late[late_age == flag]
        predicted = train[:, 2:4] / train[:, :2]
        residuals = test[:, 2:4] - test[:, :2] * predicted
        arm_sums += test[:, [4]] * predicted + residuals / PROBABILITIES
        n_decisions += test[:, 4]
    return (arm_sums[:, 0] - arm_sums[:, 1]) / n_decisions


def compare(earlier, later, reps=2000, seed=20261120):
    early, early_age = person_totals(earlier)
    late, late_age = person_totals(later)
    rng = np.random.default_rng(seed)

    # First row gives the point estimate. Remaining multiplicities resample whole people.
    def weights(n):
        return np.vstack((np.ones(n, dtype=int), rng.multinomial(n, np.full(n, 1 / n), size=reps)))

    eweights, lweights = weights(len(early)), weights(len(late))
    primary, at_enrollment = forecast(early, early_age, eweights, late_age, lweights)
    target = effect(lweights @ late)
    augmented = later_augmented(early, late, early_age, late_age, eweights, lweights)
    result = {}
    for key, values in {
        "frozen": primary,
        "age_standardized": at_enrollment,
        "later": target,
        "later_augmented": augmented,
        "frozen_minus_later_augmented": primary - augmented,
        "frozen_minus_later": primary - target,
        "age_standardized_minus_later": at_enrollment - target,
        "zero_minus_later": -target,
        "absolute_error_gain_vs_zero": abs(target) - abs(primary - target),
    }.items():
        result[key] = interval(values[0], values[1:])
    for name, frame in [("earlier", earlier), ("later", later)]:
        result[f"{name}_counts"] = {
            "people": int(frame[ID].nunique()),
            "decisions": len(frame),
            "activity": int(frame.activity.sum()),
            "no_prompt": int(frame.none.sum()),
        }
        # Native FWL independently checks the arm contrast without pooling the diet arm.
        used = frame.loc[frame.activity.eq(1) | frame.none.eq(1)]
        native = fwl_estimate(
            used.log_steps.to_numpy(),
            used.activity.to_numpy(),
            np.ones((len(used), 1)),
            add_intercept=False,
            method="ols",
        )
        expected = primary[0] if name == "earlier" else target[0]
        np.testing.assert_allclose(native.theta, expected, atol=1e-10, rtol=0)
    return result


def run(cache):
    data = load(cache)
    early, late = cohorts(data, 97)
    early_short, late_short = cohorts(data, 41)
    if (len(early), len(late), early[ID].nunique(), late[ID].nunique()) != (34643, 32548, 131, 115):
        raise ValueError("Published sources no longer reproduce the pre-outcome cohort counts")
    return {
        "cutoff_exclusive": CUTOFF.date().isoformat(),
        "bootstrap_reps": 2000,
        "days14_97": compare(early, late),
        "days14_41": compare(early_short, late_short),
        "days14_97_no_overlap": compare(
            early[early.Overlaps_60min_json.eq("[]")], late[late.Overlaps_60min_json.eq("[]")]
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
