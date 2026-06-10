"""One-time download of 401(k) dataset for offline testing.

Usage:
    python test/fixtures/download_401k.py

Downloads the Chernozhukov et al. 401(k) dataset from the doubleml package
and saves it as a CSV file (~1MB) for offline test use.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Download and cache the 401(k) dataset."""
    try:
        from doubleml.datasets import fetch_401K
    except ImportError as e:
        raise ImportError("doubleml package required: pip install doubleml") from e

    data = fetch_401K(return_type="DataFrame")

    # fetch_401K returns a dict with 'x', 'y', 'd' keys when return_type="DataFrame"
    # or a DoubleMLData object. Handle both cases.
    if isinstance(data, pd.DataFrame):
        df = data
    elif hasattr(data, "data"):
        # DoubleMLData object
        df = data.data
    else:
        # Dict-like with separate components — reconstruct
        raise TypeError(
            f"Unexpected return type from fetch_401K: {type(data)}. Please check doubleml version."
        )

    out = Path(__file__).parent / "401k_data.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {out}")
    print(f"File size: {out.stat().st_size / 1024:.0f} KB")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
