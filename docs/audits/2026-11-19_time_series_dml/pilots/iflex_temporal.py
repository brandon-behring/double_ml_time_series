"""Fit the repository's temporal PLR to the fixed-cohort Trondheim arm difference."""

import argparse

import numpy as np
import pandas as pd

from dml_ts.dml import TemporalPLRDML


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("daily_csv", help="File written by the companion iflex.R")
    args = parser.parse_args()
    data = pd.read_csv(args.daily_csv, parse_dates=["day"])
    if not data["day"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Expected one sorted row per calendar day")
    # At 18:00 on the day before an event, the preceding full day's mean
    # temperature is already available. Use a two-day lag throughout.
    calendar = data["day"].dt.dayofweek.to_numpy()
    time = np.arange(len(data)) / len(data)
    features = np.column_stack(
        [
            time,
            np.sin(2 * np.pi * calendar / 7),
            np.cos(2 * np.pi * calendar / 7),
            data["temperature"].shift(2),
        ]
    )
    data = data.iloc[2:]
    dates = data["day"].to_numpy()
    event = data["event"].to_numpy()
    difference = data["difference"].to_numpy()
    print("gap bandwidth n theta se lower upper drop warmup lag")
    for gap, bandwidth in [(2, 7), (2, 14), (7, 7)]:
        est = TemporalPLRDML(
            n_lags=1, model_y="ridge", model_t="ridge",
            cv_strategy="time_series_split", n_splits=4, gap=gap,
            hac_bandwidth=bandwidth, random_state=20261119,
        )
        fit = est.fit(difference, event, features[2:], time_index=dates)
        print(gap, bandwidth, fit.n_periods, *(f"{v:.6f}" for v in
              (fit.theta, fit.se, fit.ci_lower, fit.ci_upper)),
              len(event) - fit.n_periods, fit.dropped_initial_rows, fit.lagged_rows_dropped)
        if gap == 2 and bandwidth == 7:
            # Forward test blocks include every row after the burn-in.
            start = len(event) - fit.n_periods
            print("score start/end, event count", dates[start], dates[-1], int(sum(event[start:])))
            print("nuisance out-of-fold outcome/treatment R2", fit.outcome_r2_cv, fit.treatment_r2_cv)


if __name__ == "__main__":
    main()
