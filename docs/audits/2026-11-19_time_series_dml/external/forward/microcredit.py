"""Hold one location out of seven randomized microcredit studies of business profit."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from benchmark_io import fetch  # noqa: E402

SOURCE = (
    "https://api.bitbucket.org/2.0/repositories/rmeager/aggregate-average-impacts-microcredit/"
    "src/be1fe9098d5a27bd5a3a9cf84686087f501b0e3b/microcredit_project_data.RData"
)
DIGEST = "024ca06b0e15c23898f661a0c64aa6a21a3e56a3d9b9187bd5668fa64f50da31"


def holdout(effects, errors, target):
    """Predict a fixed held-out study effect from equally weighted independent studies."""
    effects, errors = np.asarray(effects, dtype=float), np.asarray(errors, dtype=float)
    train = np.arange(len(effects)) != target
    forecast = float(effects[train].mean())
    gap = forecast - float(effects[target])
    gap_se = float(np.sqrt(np.sum(errors[train] ** 2) / train.sum() ** 2 + errors[target] ** 2))
    return {
        "forecast": forecast,
        "forecast_minus_local": gap,
        "fixed_sites_error_ci95": [gap - 1.96 * gap_se, gap + 1.96 * gap_se],
    }


def run(cache_dir):
    path = fetch(SOURCE, cache_dir, "meager-microcredit-workspace.RData", DIGEST)
    output = subprocess.check_output(
        ["Rscript", "--vanilla", str(HERE.with_suffix(".R")), str(path)], text=True
    )
    frame = pd.read_csv(io.StringIO(output))
    if len(frame) != 7 or frame.country.nunique() != 7 or frame.source_rows.sum() != 40267:
        raise ValueError("Expected seven distinct study populations and 40,267 archived rows")
    rows = frame.to_dict("records")
    for i, row in enumerate(rows):
        row.update(holdout(frame.reported, frame.se, i))
    return {
        "units": "2009 PPP USD per household per two weeks",
        "sites": rows,
        "mean_absolute_error": np.mean([abs(row["forecast_minus_local"]) for row in rows]),
        "zero_mean_absolute_error": np.abs(frame.reported).mean(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
