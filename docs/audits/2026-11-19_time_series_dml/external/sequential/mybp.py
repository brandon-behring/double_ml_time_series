"""Reproduce the myBPmyLife activity contrast using the two public study archives."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch

SOURCE = "https://raw.githubusercontent.com/WIRED-L-um"
MRT = f"{SOURCE}/MRT_myBPmyLife/fa86c0d44ced3876f4a7351bc2d475fc4249342e"
PRIMARY = f"{SOURCE}/myBPmyLife-Primary-Analysis/f47ce32b896f55b2c7279b35aab790a75f4398b1"
FILES = {
    "trials": (
        MRT,
        "Data_MRT.zip",
        "172296f6ec08e75b9e3b1a86452ed4b0c09628aaa464eccab7802131810d2251",
    ),
    "baseline": (
        MRT,
        "Baseline_Step_Counts.csv",
        "508ec238476d357be5f3e16255feb14da68a69bd1f543f5db53c2249000fa0cf",
    ),
    "cohort": (
        MRT,
        "coreCohortID.csv",
        "f3bbf0b22aa211aa15f16017f702123b5f52b101c75324338b2699489fe13ebc",
    ),
    "demographics": (
        PRIMARY,
        "myBPmyLife_Analytic_5_10_2025.csv",
        "0aa809ee3022485cdfdb8fc611a66f7a4baf3ea356a74aede2e71d9480b0931e",
    ),
}
ID = "ParticipantIdentifier"
OUTCOME = "SumValues_Steps_60minAfter"
PRIOR_STEPS = "SumValues_Steps_30minBefore"
DAY = "TimeEnrollment_to_PrefDate_days"


def activity_effect(data, baseline_mean):
    age = data["AgeAtConsent_years_calc_int"] > 65
    white = data["Race_5cat"] == "White"
    gender = pd.get_dummies(data["Gender"], drop_first=True)
    activity = data["IsDiceRollTypeActivity"] * data["IsDiceRollSuccess"]
    diet = data["IsDiceRollTypeDiet"] * data["IsDiceRollSuccess"]
    x = np.column_stack(
        [
            np.ones(len(data)),
            age,
            gender,
            white,
            data["base_step"] > baseline_mean,
            data[PRIOR_STEPS],
            data[DAY],
            activity - 0.125,
            diet - 0.125,
        ]
    ).astype(float)
    log_steps = np.log(data[OUTCOME].to_numpy() + 0.5)
    beta = np.linalg.lstsq(x, log_steps, rcond=None)[0]
    residuals = log_steps - x @ beta
    # Influence of each row on the activity coefficient, allowing correlation within person.
    contrast = np.linalg.inv(x.T @ x)[-2]
    influence = (x @ contrast) * residuals
    grouped = np.bincount(pd.factorize(data[ID])[0], weights=influence)
    se = np.linalg.norm(grouped)
    ratio = np.exp(beta[-2])
    df = data[ID].nunique() - x.shape[1]
    half_width = t.ppf(0.975, df) * ratio * se  # author's delta-method CI
    return {
        "ratio": ratio,
        "lower": ratio - half_width,
        "upper": ratio + half_width,
        "p": 2 * t.sf(abs(beta[-2] / se), df),
        "log_se": se,
        "independent_row_log_se": np.linalg.norm(influence),
    }


def run(cache_dir):
    paths = {
        key: fetch(f"{base}/{name}", cache_dir, f"mybp-{name}", digest)
        for key, (base, name, digest) in FILES.items()
    }
    trial_cols = [
        ID,
        OUTCOME,
        PRIOR_STEPS,
        DAY,
        "DiceRollType",
        "IsDiceRollTypeActivity",
        "IsDiceRollTypeDiet",
        "IsDiceRollSuccess",
        "IsPresent_HeartRate_30minBefore",
    ]
    with ZipFile(paths["trials"]) as archive, archive.open("Data_MRT/MRT Data.csv") as source:
        data = pd.read_csv(source, usecols=trial_cols, dtype={ID: str}, low_memory=False)
    demo_cols = [ID, "Race_5cat", "AgeAtConsent_years_calc_int", "Gender"]
    demo = pd.read_csv(paths["demographics"], usecols=demo_cols, dtype={ID: str})
    baseline = pd.read_csv(paths["baseline"], dtype={ID: str})
    ids = pd.read_csv(paths["cohort"], dtype={ID: str})[ID]
    baseline = baseline[baseline[ID].isin(ids)]
    baseline_mean = baseline["base_step"].mean()
    data = data[data[ID].isin(ids) & data["DiceRollType"].notna()]
    randomizations = len(data)
    data = data.merge(demo, on=ID, how="left", validate="many_to_one").merge(
        baseline, on=ID, how="left", validate="many_to_one"
    )
    complete = data.dropna(
        subset=[
            *demo_cols,
            "base_step",
            OUTCOME,
            PRIOR_STEPS,
            DAY,
            "IsDiceRollTypeActivity",
            "IsDiceRollTypeDiet",
            "IsDiceRollSuccess",
        ]
    )
    available = complete[complete["IsPresent_HeartRate_30minBefore"] == 1]
    counts = (available[ID].nunique(), randomizations, len(complete), len(available))
    if counts != (287, 187517, 181710, 153983):
        raise ValueError(f"Unexpected participant, randomized, complete, available counts {counts}")
    estimate = activity_effect(available, baseline_mean)
    # The public CSV names differ from the analysis notebook. Use the published precision.
    np.testing.assert_allclose(
        [estimate[k] for k in ("ratio", "lower", "upper", "p")],
        [1.01, 0.98, 1.04, 0.40],
        atol=0.005,
        rtol=0,
    )
    return {
        "participants": int(counts[0]),
        "randomizations": randomizations,
        "complete": len(complete),
        "available": len(available),
        "activity": estimate,
        "all_complete_activity": activity_effect(complete, baseline_mean),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2))
