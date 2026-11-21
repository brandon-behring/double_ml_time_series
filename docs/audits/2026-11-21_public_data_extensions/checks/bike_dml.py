"""Recover known offer effects in the observed UCI bike-rental series using forward cross-fitting."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

ZIP_HASH = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
HORIZONS = (0, 1, 2, 6)
EFFECT = (10.0, 5.0, 2.0)
NY = ZoneInfo("America/New_York")


def data(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != ZIP_HASH:
        raise ValueError("Expected the original UCI bike-sharing ZIP")
    with zipfile.ZipFile(path) as archive:
        with io.TextIOWrapper(archive.open("hour.csv")) as f:
            records = list(csv.DictReader(f))
    rows = []
    for r in records:
        local = datetime.fromisoformat(r["dteday"]) + timedelta(hours=int(r["hr"]))
        first, second = [local.replace(tzinfo=NY, fold=i) for i in (0, 1)]
        if first.utcoffset() != second.utcoffset():
            continue  # Ambiguous or nonexistent local hour has no unique UTC timestamp.
        stamp = int(first.astimezone(timezone.utc).timestamp() // 3600)
        rows.append((stamp, local, int(r["workingday"]), float(r["cnt"])))
    stamps = np.asarray([r[0] for r in rows])
    stamps -= stamps.min()
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("Unexpected duplicate or unsorted physical time")
    times = [r[1] for r in rows]
    hour = np.array([t.hour for t in times])
    month = np.array([t.month for t in times])
    work = np.array([r[2] for r in rows])
    count = np.array([r[3] for r in rows])
    # Hour by workday, month, linear trend and growth by commuting hour.
    category = np.eye(48)[hour + 24 * work]
    trend = stamps / (24 * 365)
    x = np.column_stack(
        (category, np.eye(12)[month - 1, 1:], trend, trend[:, None] * category[:, 1:])
    )
    propensity = 0.35 + 0.20 * work + 0.10 * np.sin(2 * np.pi * hour / 24)
    return stamps, np.array(times), count, x, propensity


def shifted_indices(stamps, offset):
    lookup = np.full(stamps[-1] + 1, -1)
    lookup[stamps] = np.arange(len(stamps))
    target = stamps + offset
    valid = (0 <= target) & (target < len(lookup))
    index = np.full(len(stamps), -1)
    index[valid] = lookup[target[valid]]
    return index


def fit_blocks(x, times, stamps, targets, gap):
    boundaries = [
        datetime.fromisoformat(t)
        for t in (
            "2011-07-01",
            "2011-11-01",
            "2012-03-01",
            "2012-07-01",
            "2012-09-01",
            "2013-01-01",
        )
    ]
    out = {}
    for h, future in targets.items():
        blocks = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
            eval_rows = np.flatnonzero((times >= lo) & (times < hi) & (future >= 0))
            train = np.flatnonzero((future >= 0) & (stamps + h < stamps[eval_rows[0]] - gap))
            # All fitting uses earlier blocks. The training outcome at t+h must also respect the gap.
            solve = np.linalg.solve(x[train].T @ x[train] + np.eye(x.shape[1]), x[train].T)
            blocks.append((train, eval_rows, solve))
        out[h] = blocks
    return out


def estimate(y, d, p, x, stamps, future, blocks, method, lag):
    residual_y, residual_d, scored_times = [], [], []
    for train, test, solve in blocks:
        prediction = x[test] @ (solve @ y[future[train]])
        if method == "known propensity":
            fit = p[test]
        elif method == "hour and workday":
            # Pool past assignments in each of the 48 groups used by the actual assignment rule.
            cell = x[:, :48].argmax(axis=1)
            totals = np.bincount(cell[train], minlength=48)
            treated = np.bincount(cell[train], weights=d[train], minlength=48)
            fit = ((treated + 0.5) / (totals + 1))[cell[test]]
        else:
            fit = np.clip(x[test] @ (solve @ d[train]), 0.05, 0.95)
        residual_y.append(y[future[test]] - prediction)
        residual_d.append(d[test] - fit)
        scored_times.append(stamps[test])
    v, u = np.concatenate(residual_d), np.concatenate(residual_y)
    scale = v @ v
    beta = (v @ u) / scale
    score = np.zeros(stamps[-1] + 1)
    score[np.concatenate(scored_times)] = v * (u - beta * v)
    # Newey-West lags refer to elapsed hours, preserving gaps in the observed record.
    long_run = score @ score
    for k in range(1, lag + 1):
        long_run += 2 * (1 - k / (lag + 1)) * (score[k:] @ score[:-k])
    return beta, np.sqrt(max(0.0, long_run)) / scale, len(v)


def run(args):
    stamps, times, base, x, prop = data(args.zip)
    targets = {h: shifted_indices(stamps, h) for h in HORIZONS}
    blocks = fit_blocks(x, times, stamps, targets, args.gap)
    if args.last_block:
        blocks = {h: folds[-1:] for h, folds in blocks.items()}
    methods = ("known propensity", "hour and workday", "expanded calendar fit")
    observations = {(m, h): [] for m in methods for h in HORIZONS}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.repetitions):
        d = (rng.random(len(stamps)) < prop).astype(float)
        y = base.copy()
        for h, effect in enumerate(EFFECT):
            future = targets[h]
            has_endpoint = future >= 0
            y[future[has_endpoint]] += effect * d[has_endpoint]
        for (m, h), values in observations.items():
            values.append(estimate(y, d, prop, x, stamps, targets[h], blocks[h], m, args.lag))
    results = []
    for (method, h), values in observations.items():
        vals, se, n = np.array(values).T
        target = EFFECT[h] if h < len(EFFECT) else 0
        results.append(
            dict(
                method=method,
                horizon=h,
                target=target,
                n=int(n[0]),
                mean=float(vals.mean()),
                bias=float(vals.mean() - target),
                mc_se=float(vals.std(ddof=1) / np.sqrt(len(vals))),
                rmse=float(np.mean((vals - target) ** 2) ** 0.5),
                coverage=float((np.abs(vals - target) <= 1.96 * se).mean()),
            )
        )
    return dict(
        seed=args.seed,
        assignments=args.repetitions,
        utc_hours=int(len(stamps)),
        gap_hours=args.gap,
        hac_lag_hours=args.lag,
        last_block_only=args.last_block,
        effects=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument(
        "--last-block", action="store_true", help="score just September–December 2012"
    )
    parser.add_argument("--gap", type=int, default=168)
    parser.add_argument("--lag", type=int, default=48)
    args = parser.parse_args()
    if args.repetitions < 2 or not 0 <= args.gap <= 24 * 30 or not 0 <= args.lag <= 24 * 30:
        parser.error("use at least two assignments and a 0–30 day gap and HAC bandwidth")
    print(json.dumps(run(args), indent=2))
