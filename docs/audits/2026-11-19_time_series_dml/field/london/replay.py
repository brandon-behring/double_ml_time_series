"""Replay the London supply-following price experiment using the public meter archive."""

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "external"))
from benchmark_io import fetch  # noqa: E402

BASE = "https://data.london.gov.uk/download/vqm0d/"
TARIFF = BASE + "14855047-44c2-4856-8a48-e5649200e6ce/Tariffs.xlsx"
READINGS = BASE + "04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip"
TARIFF_SHA = "8a2eff6dcb737aee96cfa6c357d2947604d7f812896405c2ab5f98433d2cb150"
READINGS_SHA = "149a6a9c43c622fd0a14b7d11be055665317d3018d9fb7e1043bd51420bfaea5"
# Schofield, Table 4.5. Trial days start at 05:00 local time.
CONSTRAINT_DAYS = {
    f"2013-{md}"
    for md in "01-19 01-20 01-29 02-09 02-20 02-21 03-16 03-17 03-21 03-27 03-28 "
    "03-29 04-23 05-01 05-02 05-03 10-13 10-20 11-26 11-27 12-15".split()
}
XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CLOCKS = ("gmt_end", "same_label", "gmt_start", "local_end")
GROUP = {"Std": 0, "ToU": 1}
SEED = 20261121


def tariff_events(workbook):
    with ZipFile(workbook) as source:
        strings = [
            node.text for node in ET.fromstring(source.read("xl/sharedStrings.xml")).iter(XML + "t")
        ]
        sheet = ET.fromstring(source.read("xl/worksheets/sheet1.xml"))
    rows = list(sheet.iter(XML + "row"))
    values = [[node.text for node in row.iter(XML + "v")] for row in rows[1:]]
    calendar = pd.DataFrame(values, columns=["excel_time", "tariff"])
    calendar["tariff"] = calendar.tariff.astype(int).map(dict(enumerate(strings)))
    calendar["clock"] = pd.Timestamp("1899-12-30") + pd.to_timedelta(
        (calendar.excel_time.astype(float) * 48).round().astype(int) * 30, unit="min"
    )
    if len(calendar) != 17520 or calendar.clock.duplicated().any():
        raise ValueError("Unexpected tariff calendar.")
    calendar["date"] = (calendar.clock - pd.Timedelta(hours=5)).dt.strftime("%Y-%m-%d")
    events = []
    for date, subset in calendar.groupby("date", sort=True):
        active = subset.loc[subset.tariff != "Normal"]
        if date in CONSTRAINT_DAYS or len(active) != 6 or active.tariff.nunique() != 1:
            continue
        if not active.clock.diff().iloc[1:].eq(pd.Timedelta(minutes=30)).all():
            continue
        start = active.clock.iloc[0]
        events.append((date, start, start.hour, pd.Timestamp(date).quarter, active.tariff.iloc[0]))
    events = pd.DataFrame(events, columns=["date", "local_start", "hour", "quarter", "tariff"])
    counts = pd.crosstab(events.hour, events.tariff)
    if counts.shape != (8, 2) or (counts != 3).any().any():
        raise ValueError(f"Unexpected three-hour schedules: {counts}")
    return events


def meter_times(events, clock):
    start = events.local_start
    if clock in ("gmt_start", "gmt_end"):
        start = start.dt.tz_localize("Europe/London").dt.tz_convert("UTC").dt.tz_localize(None)
    first = 1 if clock in ("gmt_end", "local_end") else 0
    return [
        [
            (s + pd.Timedelta(minutes=30 * h)).strftime("%Y-%m-%d %H:%M:%S.0000000")
            for h in range(first, first + 6)
        ]
        for s in start
    ]


def read_partition(member, archive, key):
    with ZipFile(archive) as zf, zf.open(member) as source:
        frame = pd.read_csv(source, dtype=str)
    frame.columns = frame.columns.str.strip()
    groups = frame[["LCLid", "stdorToU"]].drop_duplicates()
    selected = frame.DateTime.isin(key)
    subset = frame.loc[selected]
    observations = pd.DataFrame(
        {
            "meter": subset.LCLid.str.removeprefix("MAC").astype("int32"),
            "group": subset.stdorToU.map(GROUP).astype("int8"),
            "interval": subset.DateTime.map(key).astype("int16"),
            "kwh": pd.to_numeric(
                subset["KWH/hh (per half hour)"].str.strip().replace("Null", np.nan), errors="raise"
            ),
        }
    )
    return observations, groups, len(frame)


def extract(archive, events, workers):
    windows = {clock: meter_times(events, clock) for clock in CLOCKS}
    key = {
        ts: idx
        for idx, ts in enumerate(
            sorted({ts for window in windows.values() for event in window for ts in event})
        )
    }
    with ZipFile(archive) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
    if len(members) != 168:
        raise ValueError(f"Unexpected partition count: {len(members)}")
    reader = partial(read_partition, archive=archive, key=key)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(reader, members))
    readings = pd.concat([p[0] for p in parts], ignore_index=True)
    roster = pd.concat([p[1] for p in parts]).drop_duplicates().copy()
    if roster.LCLid.duplicated().any() or not roster.stdorToU.isin(GROUP).all():
        raise ValueError("Meter changes tariff group or has an unknown group.")
    roster["meter"] = roster.LCLid.str.removeprefix("MAC").astype(int)
    roster = roster.set_index("meter").sort_index()
    bad = readings.duplicated(["meter", "interval"], keep=False)
    good = readings.loc[~bad]
    panel = good.pivot(index="meter", columns="interval", values="kwh").reindex(
        index=roster.index, columns=range(len(key))
    )
    values = panel.to_numpy()
    # Keep a missing total when any of the six half-hours is absent or duplicated.
    event_totals = {
        clock: values[:, [[key[ts] for ts in event] for event in window]].sum(axis=2)
        for clock, window in windows.items()
    }
    info = {
        "partitions": len(members),
        "raw_rows": sum(p[2] for p in parts),
        "selected_rows_all_clock_conventions": len(readings),
        "duplicated_meter_interval_rows_all_clocks": int(bad.sum()),
        "missing_readings_all_clocks": int(readings.kwh.isna().sum()),
        "households": roster.stdorToU.value_counts().to_dict(),
    }
    return event_totals, roster.stdorToU.map(GROUP).to_numpy(dtype=np.int8), info


def stratified_contrast(event_means, high, slots):
    """Equal-start-hour contrast and Welch-Satterthwaite SE across event days."""
    diffs, terms, dfs = [], [], []
    for slot in np.unique(slots):
        for is_high in (True, False):
            cell = event_means[(slots == slot) & (high == is_high)]
            if len(cell) < 2:
                raise ValueError("At least two assignment dates per arm and slot required for SE.")
            diffs.append((1 if is_high else -1) * cell.mean())
            terms.append(cell.var(ddof=1) / len(cell))
            dfs.append(len(cell) - 1)
    terms = np.asarray(terms)
    nslots = len(np.unique(slots))
    variance = terms.sum()
    se = np.sqrt(variance) / nslots
    df = variance**2 / np.sum(terms**2 / dfs) if variance > 0 else np.inf
    return np.sum(diffs) / nslots, se, df


def observed(totals, groups, events):
    results, event_info = [], events.copy()
    high, slots = events.tariff.eq("High").to_numpy(), events.hour.to_numpy()
    for clock, matrix in totals.items():
        for cohort, code in GROUP.items():
            available = matrix[groups == code]
            for policy in ("available", "all_48"):
                panel = (
                    available
                    if policy == "available"
                    else available[np.isfinite(available).all(axis=1)]
                )
                counts = np.isfinite(panel).sum(axis=0)
                if min(counts) < 1:
                    raise ValueError("Empty event in a selected cohort.")
                means = np.nanmean(panel, axis=0)
                beta, se, df = stratified_contrast(means, high, slots)
                half_width = float(t.ppf(0.975, df) * se)
                results.append(
                    {
                        "clock": clock,
                        "cohort": cohort,
                        "cohort_rule": policy,
                        "high_minus_low": float(beta),
                        "date_se": float(se),
                        "df": float(df),
                        "lower": float(beta - half_width),
                        "upper": float(beta + half_width),
                        "min_households": int(min(counts)),
                        "max_households": int(max(counts)),
                    }
                )
                if clock == "gmt_end" and policy == "available":
                    event_info[cohort + "_households"] = counts
                    event_info[cohort + "_kwh"] = means
    return results, event_info


def support(events):
    rows = []
    for quarter, frame in [(0, events), *events.groupby("quarter")]:
        counts = pd.crosstab(frame.hour, frame.tariff).reindex(
            index=sorted(events.hour.unique()), columns=["Low", "High"], fill_value=0
        )
        both = counts.gt(0).all(axis=1)
        rows.append(
            {
                "quarter": quarter,
                "high": int(counts.High.sum()),
                "low": int(counts.Low.sum()),
                "matched_hours": [int(h) for h in counts.index[both]],
                "high_hours": [int(h) for h in counts.index[counts.High > 0]],
                "low_hours": [int(h) for h in counts.index[counts.Low > 0]],
                "all_eight_start_times_estimable": bool(both.all()),
            }
        )
    return rows


def planted(events, matrix, repetitions):
    """Rerandomize price by date within each start-hour stratum on flat-tariff histories."""
    complete = matrix[np.isfinite(matrix).all(axis=1)]
    household_counts = [32, 256, 1024]
    if len(complete) < max(household_counts):
        raise ValueError("Insufficient households with 48 complete observed outcomes.")
    slot_rows = np.stack(
        [np.flatnonzero(events.hour == hour) for hour in sorted(events.hour.unique())]
    )
    draws = []
    for rep in range(repetitions):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, rep]))
        homes = rng.choice(len(complete), max(household_counts), replace=False)
        # Alternate high/low positions after independently shuffling each slot's six dates.
        assignment = np.stack([rng.permutation(indices) for indices in slot_rows])
        for n_households in household_counts:
            sample = complete[homes[:n_households]]
            averages, second = sample.mean(axis=0), np.square(sample).mean(axis=0)
            for per_slot in (4, 6):
                indices = assignment[:, :per_slot].reshape(8, per_slot // 2, 2)
                by_day = averages[indices]
                arm_means = by_day.mean(axis=1)
                # Independent experimental units are dates. The second calculation
                # deliberately treats every household/date observation as independent.
                m = per_slot // 2
                date_terms = by_day.var(axis=1, ddof=1) / m
                n = m * n_households
                row_terms = (second[indices].mean(axis=1) - arm_means**2) / (n - 1)
                for method, terms, denominator in [
                    ("assignment_day", date_terms, m - 1),
                    ("household_date_iid", row_terms, n - 1),
                ]:
                    varsum = terms.sum()
                    se = np.sqrt(varsum) / 8
                    df = varsum**2 / (np.square(terms).sum() / denominator)
                    halfwidth = t.ppf(0.975, df) * se
                    # Adding 0.03 kWh to each of the six low-price half-hours gives
                    # a constant high-minus-low response of -0.18 kWh.
                    for shift in (0.0, 0.18):
                        low_means = arm_means[:, 1] + shift
                        b = np.mean(arm_means[:, 0] - low_means)
                        draws.append(
                            (
                                rep,
                                n_households,
                                per_slot * 8,
                                -shift,
                                method,
                                b,
                                se,
                                halfwidth,
                                abs(b + shift) <= halfwidth,
                            )
                        )
    cols = [
        "rep",
        "households",
        "days",
        "truth",
        "inference",
        "estimate",
        "se",
        "halfwidth",
        "covered",
    ]
    draws = pd.DataFrame(draws, columns=cols)
    group = ["households", "days", "truth", "inference"]
    draws["error"] = draws.estimate - draws.truth
    summary = draws.groupby(group, as_index=False).agg(
        repetitions=("rep", "size"),
        bias=("error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(np.square(x)))),
        sampling_sd=("estimate", "std"),
        mean_reported_se=("se", "mean"),
        mean_width=("halfwidth", lambda x: 2 * x.mean()),
        coverage=("covered", "mean"),
    )
    summary["coverage_mcse"] = np.sqrt(
        summary.coverage * (1 - summary.coverage) / summary.repetitions
    )
    return draws.drop(columns="error"), summary


def publish(events, matrices, cohort, info, output, repetitions):
    results, event_details = observed(matrices, cohort, events)
    draws, summary = planted(events, matrices["gmt_end"][cohort == GROUP["Std"]], repetitions)
    meta = {
        "source_sha256": {"tariffs": TARIFF_SHA, "meters": READINGS_SHA},
        **info,
        "allocation_support": support(events),
        "planting_seed": SEED,
        "planting_repetitions_per_cell": repetitions,
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        work = Path(tmp)
        event_details.to_csv(work / "events.csv", index=False, float_format="%.9g")
        pd.DataFrame(results).to_csv(work / "effects.csv", index=False, float_format="%.9g")
        summary.to_csv(work / "calibration.csv", index=False, float_format="%.9g")
        draws.to_csv(
            work / "draws.csv.gz",
            index=False,
            float_format="%.10g",
            compression={"method": "gzip", "mtime": 0},
        )
        (work / "run.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        for path in work.iterdir():
            path.replace(output / path.name)
    primary = next(
        row
        for row in results
        if (row["clock"], row["cohort"], row["cohort_rule"]) == ("gmt_end", "ToU", "available")
    )
    print(
        json.dumps(
            {
                "study": "London",
                "events": len(events),
                "primary": primary,
                "calibration_draws": len(draws),
            },
            allow_nan=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/dml-ts/external")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    if args.workers < 1 or args.repetitions < 1:
        parser.error("workers and repetitions must be positive")
    workbook = fetch(TARIFF, args.cache_dir, "london-tariffs.xlsx", TARIFF_SHA)
    archive = fetch(READINGS, args.cache_dir, "london-meter-168.zip", READINGS_SHA)
    events = tariff_events(workbook)
    matrices, cohort, info = extract(archive, events, args.workers)
    publish(events, matrices, cohort, info, args.output, args.repetitions)


if __name__ == "__main__":
    main()
