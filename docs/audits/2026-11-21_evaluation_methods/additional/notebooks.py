"""Replay both notebooks in isolated Jupyter kernels; leave source notebooks untouched."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[4]
NOTEBOOKS = ("example_validation", "method1_bias_validation_results")
ORIGINAL_INPUTS = (
    "BIAS_VALIDATION_results_20251114_103701.csv",
    "BIAS_VALIDATION_metadata_20251114_103701.json",
)


def replay(folder, name, git_ref):
    notebook_file = f"notebooks/{name}.ipynb"
    if git_ref:
        source = subprocess.check_output(["git", "show", f"{git_ref}:{notebook_file}"], cwd=ROOT)
    else:
        source = (ROOT / notebook_file).read_bytes()
    notebook = nbformat.reads(source.decode(), as_version=4)
    working = folder / name / "notebooks"
    working.mkdir(parents=True)
    results = working.parent / "results"
    inputs = results / "simulations"
    inputs.mkdir(parents=True)
    for file in ORIGINAL_INPUTS:
        shutil.copy(ROOT / "results" / "simulations" / file, inputs)
    client = NotebookClient(notebook, kernel_name="audit", timeout=240)
    try:
        client.execute(cwd=str(working))
    except CellExecutionError:
        if not git_ref:
            raise
        for i, cell in enumerate(notebook.cells):
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    return {
                        "stopped_at_cell": i,
                        "exception": output.ename,
                        "message": output.evalue,
                    }
        raise
    report = {"completed_code_cells": sum(c.cell_type == "code" for c in notebook.cells)}
    if name == "example_validation":
        (path,) = (results / "validation/results").glob("*.json")
        record = json.loads(path.read_text())
        report.update(
            {
                k: record[k]
                for k in [
                    "n_simulations",
                    "bias",
                    "mse",
                    "coverage",
                    "ci_lower",
                    "ci_upper",
                    "status",
                    "coverage_p_value",
                ]
            }
        )
        report["identical_serial_parallel_draws"] = record["n_simulations"]
        report["controlled_and_naive_example"] = (
            "".join(o.get("text", "") for o in notebook.cells[8].outputs).strip().splitlines()
        )
    else:
        df = pd.read_csv(inputs / ORIGINAL_INPUTS[0])
        report.update(
            {
                "archived_configs": len(df),
                "archived_status": df.status.value_counts().to_dict(),
                "largest_absolute_bias": df.bias.abs().max(),
                "figures_written": len(list((results / "figures").glob("*.png"))),
                "tables_written": len(list((results / "latex").glob("*.tex"))),
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--original-ref", help="also execute both uncorrected notebooks from this git revision"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        spec = folder / "jupyter/kernels/audit"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "isolated audit",
                    "language": "python",
                }
            )
        )
        os.environ.update(
            {
                "JUPYTER_PATH": str(folder / "jupyter"),
                "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.getenv("PYTHONPATH")])),
                "MPLBACKEND": "Agg",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        report = {
            "versions": {p: version(p) for p in ["nbclient", "ipykernel", "pandas", "statsmodels"]}
        }
        if args.original_ref:
            report["original_ref"] = args.original_ref
            report["original"] = {
                n: replay(folder / "original", n, args.original_ref) for n in NOTEBOOKS
            }
        report["corrected"] = {n: replay(folder / "corrected", n, None) for n in NOTEBOOKS}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
