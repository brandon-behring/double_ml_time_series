"""Measure carryover bias against a known block-assignment contrast on public ride counts."""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


INPUT_HASH = "cc3d92c60f4dd5a5e4c19b04aa0c5cff398022c1fa372d51b9640f2a646b3bf1"
INTERVALS_PER_DAY = 96
PERSISTENCE = 0.5 ** 0.25
SUSTAINED_EFFECT = 100.0


def read_demand(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != INPUT_HASH:
        raise ValueError("Unexpected ride-count release")
    with path.open() as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["15min_interval", "num_rides"]:
            raise ValueError("Unexpected columns in the ride series")
        rows = list(reader)
    times = [datetime.fromisoformat(r["15min_interval"]) for r in rows]
    if times != [datetime(2022, 1, 1) + i * timedelta(minutes=15) for i in range(31 * INTERVALS_PER_DAY)]:
        raise ValueError("Expected a complete January grid")
    demand = np.array([int(row["num_rides"]) for row in rows], dtype=float)
    slot = np.arange(len(rows)) % INTERVALS_PER_DAY
    weekend = np.array([t.weekday() >= 5 for t in times])
    group = slot + INTERVALS_PER_DAY * weekend
    train = 7 * INTERVALS_PER_DAY
    means = np.bincount(group[:train], weights=demand[:train]) / np.bincount(group[:train])
    return demand[train:], means[group[train:]]


def benchmark(demand, baseline, width, repetitions, seed):
    rng = np.random.default_rng(seed)
    nblocks, remainder = divmod(len(demand), width)
    if remainder or repetitions < 2:
        raise ValueError("Need full assignment blocks and two or more replications")
    estimates = []
    for _ in range(repetitions):
        assignments = np.repeat(rng.integers(0, 2, size=nblocks), width)
        state = lfilter([1 - PERSISTENCE], [1, -PERSISTENCE], assignments)
        outcomes = demand + SUSTAINED_EFFECT * state
        # With P(Z=1)=1/2, subtracting a fixed preassignment baseline preserves the contrast.
        weight = 2 * (2 * assignments - 1)
        estimates.append([np.mean(weight * outcomes), np.mean(weight * (outcomes - baseline))])
    estimates = np.array(estimates)
    average = estimates.mean(axis=0)
    spread = estimates.std(axis=0, ddof=1)
    return dict(
        block_quarters=width,
        assignments_per_repetition=nblocks,
        analytic_current_block_effect=float(
            SUSTAINED_EFFECT * np.mean(1 - PERSISTENCE ** np.arange(1, width + 1))
        ),
        raw_mean=float(average[0]),
        adjusted_mean=float(average[1]),
        raw_randomization_sd=float(spread[0]),
        adjusted_randomization_sd=float(spread[1]),
        adjusted_monte_carlo_se=float(spread[1] / np.sqrt(repetitions)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--repetitions", type=int, default=2000)
    a = p.parse_args()
    demand, baseline = read_demand(a.csv)
    result = dict(
        csv_sha256=INPUT_HASH,
        tested_quarters=len(demand),
        repetitions=a.repetitions,
        sustained_contrast=SUSTAINED_EFFECT,
        carryover_half_life_quarters=4,
        seed_base_plus_block_quarters=20261120,
        designs=[benchmark(demand, baseline, b, a.repetitions, 20261120 + b) for b in (4, 24)],
    )
    print(json.dumps(result, indent=2))
