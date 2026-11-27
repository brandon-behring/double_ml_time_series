"""Diagnose support in a deliberately naive California-only PLR."""

import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from temporalcv import TimeSeriesCrossValidator

from dml_ts import TemporalPLRDML


def main(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "550d24c2f743507d097692d181bdbbf36c6168a39a0b6fc831cc89d98082f7de":
        raise ValueError("Source CSV checksum differs from the audited release")
    df = pd.read_csv(path, sep=";")
    assert not df.duplicated(["State", "Year"]).any()
    sales = df.pivot(index="Year", columns="State", values="PacksPerCapita").sort_index()
    assert sales.shape == (31, 39) and sales.notna().all().all()
    years = sales.index.to_numpy()
    assert np.array_equal(years, np.arange(1970, 2001))
    d = (years >= 1989).astype(float)
    y = sales["California"].to_numpy()
    x = sales.drop(columns="California").mean(axis=1).to_numpy().reshape(-1, 1)
    cv = TimeSeriesCrossValidator(n_splits=3, gap=1, expanding=True)
    for train, test in cv.split(x):
        print(
            "Train, test years / treated counts:",
            years[train[[0, -1]]],
            years[test[[0, -1]]],
            int(sum(d[train])),
            int(sum(d[test])),
        )
    # A finite coefficient is possible even where earlier folds lack treated support.
    for bandwidth in (0, 2, 4):
        est = TemporalPLRDML(
            n_lags=0,
            model_y="ridge",
            model_t="ridge",
            gap=1,
            n_splits=3,
            hac_bandwidth=bandwidth,
            random_state=42,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = est.fit(y, d, x, time_index=years)
        print(
            f"BW {bandwidth}: estimate {fit.theta:.4f}, SE {fit.se:.4f}, "
            f"CI [{fit.ci_lower:.4f}, {fit.ci_upper:.4f}], scored {fit.n_periods}, "
            f"OOF prefix dropped {fit.dropped_initial_rows}, outcome/treatment R2 "
            f"{fit.outcome_r2_cv:.3f}/{fit.treatment_r2_cv:.3f}, "
            f"mean residual D squared {np.mean(fit.T_residual**2):.4f}"
        )
        for message in sorted({str(w.message) for w in caught}):
            print("Warning:", message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    main(parser.parse_args().csv)
