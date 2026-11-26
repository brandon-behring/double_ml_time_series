"""Recalculate the Fulton teaching-file benchmarks and check instrument support by date."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/audits/2026-11-19_time_series_dml/external"))
from benchmark_io import fetch  # noqa: E402

MIT = "https://economics.mit.edu/sites/default/files/inline-files/fish.dta"
TEACHING = "https://raw.githubusercontent.com/ccolonescu/PoEdata/c41bee8fb53ccdec53961fc9fd85b7b84bd17d1c/data/fultonfish.rda"
MIT_SHA = "3e9835e401ed5b8ef0ed5bd00d8c95d6fb13a5a58b8abb5ff03fd91446b3f308"
TEACHING_SHA = "9509671e5ce61f790a91b32dba906b1f936a698eda3b20defd160f3479d4d9e5"


def run(cache):
    archive = pd.read_stata(fetch(MIT, cache, "fulton-fish.dta", MIT_SHA))
    teaching = fetch(TEACHING, cache, "fulton-teaching.rda", TEACHING_SHA)
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "fulton.csv"
        subprocess.run(
            [
                "Rscript",
                "-e",
                "a=commandArgs(TRUE); load(a[1]); write.csv(fultonfish,a[2],row.names=FALSE)",
                str(teaching),
                str(csv),
            ],
            check=True,
        )
        df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df.date.astype(str), format="%y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    if df.date.duplicated().any() or (df.stormy + df.mixed > 1).any():
        raise ValueError("Dates must be unique and weather categories disjoint")
    indices = np.array_split(np.arange(len(df)), 5)
    controls = np.column_stack([np.ones(len(df)), df[["mon", "tue", "wed", "thu"]]])
    variables = df[["lprice", "lquan", "stormy"]].to_numpy()
    price, quantity, storm = (
        variables - controls @ np.linalg.lstsq(controls, variables, rcond=None)[0]
    ).T
    ols = price @ quantity / (price @ price)
    first_stage = storm @ price / (storm @ storm)
    reduced_form = storm @ quantity / (storm @ storm)
    return {
        "author_stata": {"sha256": MIT_SHA, "rows": len(archive), "fields": list(archive.columns)},
        "dated_teaching_file": {
            "sha256": TEACHING_SHA,
            "rows": len(df),
            "first_date": df.date.min().date().isoformat(),
            "last_date": df.date.max().date().isoformat(),
            "stormy": int(df.stormy.sum()),
            "mixed": int(df.mixed.sum()),
            "fair": int((1 - df.stormy - df.mixed).sum()),
        },
        "weekday_adjusted": {
            "ols": float(ols),
            "first_stage": float(first_stage),
            "reduced_form": float(reduced_form),
            "storm_iv_ratio": float(reduced_form / first_stage),
        },
        "five_date_blocks": [
            {
                "dates": len(ix),
                "stormy": int(df.stormy.iloc[ix].sum()),
                "fair": int((1 - df.stormy.iloc[ix] - df.mixed.iloc[ix]).sum()),
            }
            for ix in indices
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.cache_dir), indent=2, allow_nan=False) + "\n")
