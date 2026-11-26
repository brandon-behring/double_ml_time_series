"""Compare the real default forest case-bootstrap diagnostic on paired simulated samples."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    live = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(live))
    from dml_ts.validation.bootstrap_diagnostics import BootstrapDiagnostics
    from dml_ts.validation.dgp_generator import DGPGenerator

    original = args.snapshot / "dml_ts/validation/bootstrap_diagnostics.py"
    spec = importlib.util.spec_from_file_location(
        "dml_ts.validation._historical_bootstrap", original
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = []
    for seed in range(6):
        data = DGPGenerator(
            n=200, p=5, true_effect=2, confounding_strength=0.5, random_state=100 + seed
        ).generate()
        old = module.BootstrapDiagnostics(data, "LinearDML", random_state=70 + seed)
        new = BootstrapDiagnostics(data, "LinearDML", random_state=70 + seed)
        before, after = old._bootstrap_estimates(4), new._bootstrap_estimates(4)
        for index, (a, b) in enumerate(zip(before, after, strict=True)):
            rows.append(
                {
                    "data_seed": 100 + seed,
                    "draw": index,
                    "position_folds": a,
                    "original_unit_folds": b,
                }
            )
    a = np.array([row["position_folds"] for row in rows])
    b = np.array([row["original_unit_folds"] for row in rows])
    receipt = {
        "snapshot_source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "data_sets": 6,
        "paired_resamples_per_data_set": 4,
        "sample_size": 200,
        "plant": 2,
        "estimator": "BootstrapDiagnostics with its default 100-tree nuisance models",
        "position_folds_mean": a.mean(),
        "original_unit_folds_mean": b.mean(),
        "position_folds_rmse_vs_plant": np.sqrt(np.mean((a - 2) ** 2)),
        "original_unit_folds_rmse_vs_plant": np.sqrt(np.mean((b - 2) ** 2)),
        "maximum_absolute_paired_shift": np.max(np.abs(a - b)),
        "scope": "The four resamples per dataset are dependent; these checks do not measure coverage.",
        "draws": rows,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
