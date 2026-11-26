"""Reproduce four external checks of effect persistence, transfer, and assignment support."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402
from forward.retail_assignments import retail_replay  # noqa: E402
from forward.sierra_indices import sierra_replay  # noqa: E402

EFFORT_URL = "https://devingpope.com/assets/files/Website_Stability.pdf"
EFFORT_SHA = "860de8894dd197a2f80051f151ca8349d5d0cd8b6f456eb31a3219ba5c383526"
SIERRA_URL = "https://zenodo.org/api/records/7411565/files/cdd-replication.zip/content"
SIERRA_SHA = "2e7c5a3de0be3e0dabeb8aad91294b1f922cd71d6cb463b7bc2b69d3b5b430f1"
NUDGE_URL = (
    "https://www.econometricsociety.org/publications/econometrica/2022/01/01/"
    "rcts-scale-comprehensive-evidence-two-nudge-units/supp/18709_Data_and_Programs.zip"
)
NUDGE_SHA = "8cf6e754b6f34710a0a27286c471dec17d45b2b40d4219052b0b53ae890f059e"


def effort_replay(path):
    table = subprocess.check_output(
        ["pdftotext", "-f", "8", "-l", "9", "-layout", str(path), "-"], text=True
    )
    # Columns 3 and 4 give paired integer button-press means. Column 5 is decimal.
    pairs = re.findall(r"\b(\d,\d{3})\s+(\d,\d{3})\s+\d{2}\.\d{2}\b", table)
    if len(pairs) != 15:
        raise ValueError(f"Expected 15 experimental arms in Table 1, found {len(pairs)}")
    points = [[int(number.replace(",", "")) for number in pair] for pair in pairs]
    observed = float(spearmanr(*np.array(points).T).statistic)
    if round(observed, 2) != 0.91:
        raise ValueError("Observed rank correlation disagrees with Table 2")
    return {
        "years": [2015, 2018],
        "rank_correlation": observed,
        "arm_means_in_table_order": points,
        "table_2_observed_reference": 0.91,
        "table_2_authors_bootstrap_stability_reference": 0.93,
    }


def predict_trials(arms):
    """Forward historical-mean and zero forecasts with one vote per held-out randomized trial."""
    fields = ["trialnumber", "year", "treatmenteffect"]
    if arms.empty or arms[fields].isna().any().any() or not np.isfinite(arms.treatmenteffect).all():
        raise ValueError("Each randomized arm needs a trial, year and finite effect in pp")
    if arms.groupby("trialnumber").year.nunique().gt(1).any():
        raise ValueError("Trial crosses training and test years")
    trials = arms.groupby(["year", "trialnumber"], sort=True).treatmenteffect.mean().reset_index()
    years = sorted(trials.year.unique())
    if len(years) < 2:
        raise ValueError("Forward forecasts require at least two distinct trial years")
    schedule = []
    pooled = []
    for year in years[1:]:
        old = trials.loc[trials.year < year, "treatmenteffect"]
        later = trials.loc[trials.year == year, "treatmenteffect"]
        historical = float(old.mean())
        loss_history = (later - historical) ** 2
        loss_zero = later**2
        advantages = loss_zero - loss_history
        pooled.extend(advantages)
        for method, forecast, squared_error in [
            ("zero", 0.0, loss_zero),
            ("historical", historical, loss_history),
        ]:
            schedule.append(
                {
                    "origin_year": int(year),
                    "method": method,
                    "forecast_pp": forecast,
                    "early_trials": len(old),
                    "held_out_trials": len(later),
                    "mean_benchmark_pp": float(later.mean()),
                    "mean_signed_error_pp": float(forecast - later.mean()),
                    "mean_abs_error_pp": float((forecast - later).abs().mean()),
                    "mse_pp2": float(squared_error.mean()),
                    "advantage_vs_zero_pp2": float((loss_zero - squared_error).mean()),
                }
            )
    return {
        "arms": len(arms),
        "trials": len(trials),
        "arm_mean_pp": float(arms.treatmenteffect.mean()),
        "schedule": schedule,
        "held_out_trials": len(pooled),
        "pooled_historical_advantage_pp2": float(np.mean(pooled)),
    }


def nudge_replay(path):
    with ZipFile(path) as archive, archive.open("cleaned_data/NudgeUnits.dta") as file:
        arms = pd.read_stata(
            file, columns=["trialnumber", "year", "treatmenteffect"], convert_categoricals=False
        )
    result = predict_trials(arms)
    if (result["arms"], result["trials"], round(result["arm_mean_pp"], 1)) != (241, 126, 1.4):
        raise ValueError(
            "Data do not reproduce the trial counts and published arm-average take-up effect"
        )
    return result


def run(cache):
    return {
        "effort": effort_replay(fetch(EFFORT_URL, cache, "effort-stability.pdf", EFFORT_SHA)),
        "sierra_leone": sierra_replay(fetch(SIERRA_URL, cache, "sierra-cdd.zip", SIERRA_SHA)),
        "nudge_units": nudge_replay(
            fetch(NUDGE_URL, cache, "nudge-units.zip", NUDGE_SHA, user_agent="Mozilla/5.0")
        ),
        "retail": retail_replay(cache),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir), indent=2, allow_nan=False))
