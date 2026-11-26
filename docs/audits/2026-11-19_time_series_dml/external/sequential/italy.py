"""Rebuild the Italian bank capital study's Figure 2 and stress-test its GDP result."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from dml_ts.dml import TemporalPLRDML  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch

URL = "https://zenodo.org/api/records/21030557/files/3-replication-package.zip/content"
SHA = "0f5a2145afcc9c3955f9544e8a937bf7447e90d13b9f8ea3922b44f81257cdd0"
DATA_SHA = "b96cec8af4b0f3e666765ea387b5d843f2f5ee1ef5ae6ec3d2d54372c47d4738"
OUTCOMES = (
    "Tier 1 capital",
    "Risk-weighted assets",
    "PNFC_Spread",
    "PNFC_Lending_K2020",
    "GDP_K2020",
)
BASE = Path("results/estimation/calibration_goldi")

# Save the exact GDP design at the end of the authors' baseline script. Its own
# preprocessing includes three lags and all of its retained series are first-differenced.
BASELINE = """
import runpy
import numpy as np
ctx = runpy.run_path("replications/01_empirical_example.py", run_name="__main__")
assert ctx["outcome_var"] == "GDP_K2020"
assert set(map(int, ctx["df_fixed"].loc["Transform:"])) == {2, 5}
p = ctx["processor"]
np.savez("gdp-design.npz", X=ctx["X"], policy=ctx["policy"], leads=ctx["leads"],
         dates=p.original_index.to_numpy(dtype="datetime64[D]"),
         raw_dates=p.original_data.index.to_numpy(dtype="datetime64[D]"))
"""


def calendar_support(dates, raw_dates, horizon):
    """Require every raw quarter from t-4 to t+h for three lags of first differences."""
    quarters = pd.DatetimeIndex(raw_dates).to_period("Q")
    if not quarters.is_unique or not quarters.is_monotonic_increasing:
        raise ValueError("Raw dates must be unique and in ascending order")
    positions = quarters.get_indexer(pd.DatetimeIndex(dates).to_period("Q"))
    if (positions < 4).any() or (positions + horizon >= len(quarters)).any():
        raise ValueError("Index lacks the required four-quarter history or outcome window")
    return quarters.asi8[positions + horizon] - quarters.asi8[positions - 4] == horizon + 4


def forward_estimates(path, scale):
    h = 2
    with np.load(path) as data:
        dates, design = data["dates"], data["X"]
        if design.shape != (54, 105):
            raise ValueError(f"Unexpected GDP design dimensions {design.shape}")
        keep = calendar_support(dates, data["raw_dates"], h)
        cumulative = data["leads"][:, : h + 1].sum(axis=1)
        estimates = {}
        for name, mask in [
            ("published_rows", np.ones(len(dates), dtype=bool)),
            ("continuous_calendar", keep),
        ]:
            model = TemporalPLRDML(
                n_lags=0,
                model_y="ridge",
                model_t="ridge",
                cv_strategy="time_series_split",
                n_splits=3,
                gap=h,
                hac_bandwidth=h + 1,
                random_state=42,
            )
            fit = model.fit(
                cumulative[mask], data["policy"][mask], design[mask], time_index=dates[mask]
            )
            estimates[name] = {
                "origins": int(mask.sum()),
                "scored": fit.n_periods,
                "dropped": int(fit.dropped_initial_rows),
                "theta": float(fit.theta * scale),
                "se": float(fit.se * scale),
                "lower": float(fit.ci_lower * scale),
                "upper": float(fit.ci_upper * scale),
                "outcome_r2": fit.outcome_r2_cv,
                "treatment_r2": fit.treatment_r2_cv,
            }
        return estimates, dates[~keep].astype(str).tolist()


def run(cache_dir, author_python):
    archive_path = fetch(URL, cache_dir, "italian-bank-capital.zip", SHA)
    python = shutil.which(str(author_python))
    if not python:
        raise ValueError(f"Could not find author Python {author_python}")
    python = os.path.abspath(python)
    with tempfile.TemporaryDirectory(prefix="dml-italy-") as tmp:
        package = Path(tmp) / "replication-package"
        with ZipFile(archive_path) as z:
            z.extractall(
                tmp, members=[n for n in z.namelist() if n.startswith("replication-package/")]
            )
        expected = {}
        for name in OUTCOMES:
            path = package / BASE / name / "results" / "estimation_results_scaled.csv"
            expected[name] = pd.read_csv(path, index_col="Horizon")
        # Ensure the scripts regenerate data, estimates, and figure from the raw inputs.
        data_path = package / "data/data_macroprudential.csv"
        data_path.unlink()
        shutil.rmtree(package / BASE)
        figure = package / "results/figures/figure2.pdf"
        figure.unlink()
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(Path(tmp) / "mpl"),
        )
        env.pop("X13PATH", None)

        def author_step(*args):
            result = subprocess.run(
                [python, *args],
                cwd=package,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"Author step failed ({args[0]})\n{result.stdout[-3500:]}")

        author_step("replications/00_run_generate_data_macroprudential.py")
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != DATA_SHA:
            raise ValueError("Regenerated 71-quarter source CSV differs from the frozen input")
        author_step("-c", BASELINE)
        author_step("replications/02_scale_irfs.py")
        author_step("replications/03_plot_empirical_example.py")
        if not figure.read_bytes().startswith(b"%PDF"):
            raise ValueError("Figure 2 was not regenerated")

        actual = {}
        max_error = 0.0
        for name, target in expected.items():
            table = pd.read_csv(
                package / BASE / name / "results/estimation_results_scaled.csv", index_col="Horizon"
            )
            if list(table.index) != list(range(9)) or not table.columns.equals(target.columns):
                raise ValueError(f"Unexpected horizons or confidence bands for {name}")
            np.testing.assert_allclose(table, target, rtol=0, atol=1e-8)
            max_error = max(max_error, float(np.max(np.abs(table.to_numpy() - target.to_numpy()))))
            actual[name] = table

        # Freeze economic units across specifications at the paper's 50bp normalization.
        def unscaled_impact(name):
            return pd.read_csv(package / BASE / name / "results/estimation_results.csv").loc[
                0, "Coefficient"
            ]

        scale = 0.5 / (unscaled_impact("Tier 1 capital") - unscaled_impact("Risk-weighted assets"))
        np.testing.assert_allclose(
            actual["Tier 1 capital"].loc[0, "Coefficient"]
            - actual["Risk-weighted assets"].loc[0, "Coefficient"],
            0.5,
        )
        forward, exclusions = forward_estimates(package / "gdp-design.npz", scale)
        return {
            "raw_quarters": 71,
            "origins": 54,
            "regressors": 105,
            "cells_compared": 225,
            "max_abs_error": max_error,
            "scale": scale,
            "GDP_h2": actual["GDP_K2020"].loc[2].to_dict(),
            "lending_h2": actual["PNFC_Lending_K2020"].loc[2].to_dict(),
            "forward_h2": forward,
            "excluded_origins": exclusions,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--author-python", default=os.environ.get("DML_TS_ITALY_PYTHON", sys.executable)
    )
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir, args.author_python), indent=2))
