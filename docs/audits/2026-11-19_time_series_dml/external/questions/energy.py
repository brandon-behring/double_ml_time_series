"""Peak, off-peak and two-day consumption contrasts for the Norwegian iFlex experiment."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path[:0] = [str(Path(__file__).resolve().parents[i]) for i in (1, 5)]
from benchmark_io import fetch  # noqa: E402

from dml_ts.dml.double_ml import double_ml  # noqa: E402

URL = "https://zenodo.org/records/8248802/files/iFlex%20dataset.RData"
SHA = "0ed8a76365f6379cf613c12069180d10db0ef6f56de0e73268e2de28f62908a2"
# As in pilots/iflex.R, R preserves the workspace's factor labels and Date columns.
READ_HOURS = r"""
load(commandArgs(trailingOnly=TRUE)[1])
r <- subset(participants, Group_Phase2 %in% c('Trond_1','Trond_Control'))
stopifnot(!anyDuplicated(r$ID), all(r$Participation_status_Phase2 == 'Ok'),
          all(r$Participation_Experiment == 'Yes'))
x <- data_hourly[data_hourly$Participation_Phase == 'Phase_2' & data_hourly$ID %in% r$ID,
                 c('ID','Date','Hour','Demand_kWh','Price_signal')]
events <- subset(experiment_days, Group == 'Trond_1' & Participation_Phase == 'Phase_2')
stopifnot(nrow(events) == 21L, !anyDuplicated(events$Date),
          all(events$Price_signal %in% c('B_2','B_5','B_10','B_15')))
tr <- r$Group_Phase2[match(x$ID,r$ID)] == 'Trond_1'
hr <- as.integer(as.character(x$Hour))
stopifnot(nrow(x) == 1127520, !anyNA(x$Demand_kWh), !anyDuplicated(x[c('ID','Date','Hour')]),
          all(table(x$ID,x$Date)==24), all(hr %in% 1:24), all(x$Price_signal[!tr]==''))
signal <- as.character(events$Price_signal[match(x$Date[tr],events$Date)])
signal[is.na(signal)] <- ''
stopifnot(all(as.character(x$Price_signal[tr]) == signal))
pk <- hr>=8 & hr<=20
key <- interaction(x$ID,x$Date,drop=TRUE)
sums <- rowsum(cbind(peak=x$Demand_kWh*pk,offpeak=x$Demand_kWh*!pk),key)
i <- match(rownames(sums),as.character(key)); stopifnot(!anyNA(i))
out <- data.frame(household=x$ID[i],day=as.character(as.Date(x$Date[i])),arm=as.integer(tr[i]),
                  event=as.integer(x$Date[i] %in% events$Date),sums)
write.csv(out,stdout(),row.names=FALSE)
"""


def isolated_starts(event_days, observed_days):
    """Allow two complete days with no adjacent event or subsequent announcement."""
    events = set(pd.DatetimeIndex(event_days))
    observed = set(pd.DatetimeIndex(observed_days))
    day = pd.Timedelta(days=1)
    return sorted(
        d
        for d in events
        if {d - day, d, d + day, d + 2 * day}.issubset(observed)
        and not {d - day, d + day, d + 2 * day}.intersection(events)
    )


def household_windows(daily, starts, hours):
    """Average complete event windows per household before comparing randomized arms."""
    starts = pd.DatetimeIndex(starts)
    data = daily.set_index(["household", "day"]).sort_index()
    if not data.index.is_unique or not len(starts) or not starts.is_unique:
        raise ValueError("Duplicate household-days or invalid event dates")
    index = pd.MultiIndex.from_product(
        [data.index.get_level_values(0).unique(), starts], names=data.index.names
    )
    same = data.reindex(index)[["peak", "offpeak"]]
    if hours == 48:
        second_index = pd.MultiIndex.from_arrays(
            [index.get_level_values(0), index.get_level_values(1) + pd.Timedelta(days=1)],
            names=index.names,
        )
        second = data.reindex(second_index)[["peak", "offpeak"]]
        second.index = index
        same = same + second
    elif hours != 24:
        raise ValueError("Expected a 24-hour or 48-hour window")
    if same.isna().any().any():
        raise ValueError("Missing readings inside a requested household window")
    same["total"] = same.peak + same.offpeak
    return same.groupby(level="household").mean()


def welch(outcome, arm):
    treated = outcome[arm == 1]
    control = outcome[arm == 0]
    t = stats.ttest_ind(treated, control, equal_var=False)
    ci = t.confidence_interval()
    return {
        "effect": float(treated.mean() - control.mean()),
        "se": float(
            np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        ),
        "ci95": [float(ci.low), float(ci.high)],
    }


def compare(after, baseline, arm, baseline_days=1):
    if not after.index.equals(baseline.index) or not after.index.equals(arm.index):
        raise ValueError("Household sets differ across comparisons")
    x = baseline[["peak", "offpeak"]].to_numpy(dtype=float)
    treatment = arm.to_numpy(dtype=float)
    out = {}
    for target in ["peak", "offpeak", "total"]:
        y = after[target].to_numpy(dtype=float)
        before = baseline_days * baseline[target].to_numpy(dtype=float)
        model = double_ml(y, treatment, x, model="ridge", n_folds=5, random_state=42)
        out[target] = {
            "assigned_arms": welch(y, treatment),
            "change_from_baseline": welch(y - before, treatment),
            "crossfit_ridge": {
                "effect": model.theta,
                "se": model.se,
                "ci95": [model.ci_lower, model.ci_upper],
            },
        }
    for method in ["assigned_arms", "change_from_baseline", "crossfit_ridge"]:
        np.testing.assert_allclose(
            out["total"][method]["effect"],
            out["peak"][method]["effect"] + out["offpeak"][method]["effect"],
            atol=1e-8,
        )
    return out


def analyze(daily):
    daily = daily.sort_values(["household", "day"])
    events = daily.loc[daily.event.eq(1), "day"].drop_duplicates().sort_values()
    dates = pd.DatetimeIndex(daily.day.unique()).sort_values()
    days = pd.Timedelta(days=1)
    if (
        daily.groupby("household").arm.nunique().max() != 1
        or daily.groupby("day").event.nunique().max() != 1
        or not np.all(np.diff(dates.values) == np.timedelta64(1, "D"))
    ):
        raise ValueError("Inconsistent arm assignments or shared calendar")
    arm = daily.groupby("household", sort=True).arm.first()
    if arm.value_counts().to_dict() != {0: 204, 1: 201} or len(dates) != 116 or len(events) != 21:
        raise ValueError("Unexpected Trondheim cohort or event schedule")
    pre = dates[dates < events.iloc[0] - days]
    if len(pre) != 14:
        raise ValueError("Expected 14 complete pre-announcement days")
    baseline = household_windows(daily, pre, 24)
    isolated = isolated_starts(events, dates)
    all_followed = [d for d in events if d + days in dates]

    def window(starts, hours):
        outcomes = household_windows(daily, starts, hours)
        return {
            "event_count": len(starts),
            "hours": hours,
            "contrasts": compare(outcomes, baseline, arm, baseline_days=hours // 24),
        }

    result = {
        "households": {"assigned": 201, "control": 204},
        "calendar_days": len(dates),
        "events": len(events),
        "baseline_dates": [str(pre[0].date()), str(pre[-1].date())],
        "isolated_event_dates": [str(d.date()) for d in isolated],
        "first_24h": window([events.iloc[0]], 24),
        "all_24h": window(events, 24),
        "overlap_allowed_48h": window(all_followed, 48),
        "isolated_24h": window(isolated, 24),
        "isolated_next_24h": window([d + days for d in isolated], 24),
        "isolated_48h": window(isolated, 48),
        "pre_announcement_placebo": compare(
            household_windows(daily, pre[7:], 24), household_windows(daily, pre[:7], 24), arm
        ),
    }
    # The same households, pre-event covariates, folds and linear estimators
    # make the 48-hour point estimate additive across the two days.
    for target in ("peak", "offpeak", "total"):
        for method in ("assigned_arms", "change_from_baseline", "crossfit_ridge"):
            first, next_day, both = [
                result[key]["contrasts"][target][method]["effect"]
                for key in ("isolated_24h", "isolated_next_24h", "isolated_48h")
            ]
            np.testing.assert_allclose(both, first + next_day, rtol=0, atol=1e-8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    source = fetch(URL, args.cache_dir, "iflex-v2.RData", SHA)
    proc = subprocess.run(
        ["Rscript", "--vanilla", "-e", READ_HOURS, str(source)], check=True, stdout=subprocess.PIPE
    )
    daily = pd.read_csv(io.BytesIO(proc.stdout), parse_dates=["day"])
    print(json.dumps(analyze(daily), indent=2))


if __name__ == "__main__":
    main()
