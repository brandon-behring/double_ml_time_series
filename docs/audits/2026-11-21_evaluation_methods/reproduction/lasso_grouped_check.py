"""Pair the repaired household-fold bootstrap with three verified full-fixture historical draws."""

import argparse
import collections
import contextlib
import functools
import hashlib
import json
import os
import signal
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lasso_resources import process_identity
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from standalone_lasso import CHECK, FIXTURE, SEED, diagnostic, save, stamp


@contextlib.contextmanager
def share_live_pool(cache):
    if cache is None:
        yield
        return
    heartbeat = cache / "admission.jsonl"
    if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 15:
        raise ValueError("The optional live pool must be actively governed")
    marker = cache / "serial-peak-active.json"
    if marker.exists():
        raise ValueError("A different verification is using the shared pool")
    pid = os.getpid()
    save(marker, {"pid": pid, "proc_start_ticks": process_identity(pid)})
    try:
        os.kill(pid, signal.SIGSTOP)
        yield
    finally:
        marker.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_root", type=Path)
    parser.add_argument("verified_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--governed-cache", type=Path, help="Use a worker slot in a concurrent replay"
    )
    args = parser.parse_args()
    with share_live_pool(args.governed_cache):
        root = args.live_root.resolve()
        historical = json.loads((args.verified_cache / "verification.json").read_text())
        manifest = json.loads((args.verified_cache / "manifest.json").read_text())
        digest = hashlib.sha256((root / FIXTURE).read_bytes()).hexdigest()
        if digest != manifest["source_sha256"][str(FIXTURE)] or not historical["exact"]:
            raise ValueError("Fixture or historical verification differs")
        obj = diagnostic(root)
        y, t, x = obj.preprocess_data()
        rng = np.random.RandomState(SEED)
        indices = [rng.choice(len(y), size=len(y), replace=True) for _ in range(CHECK)]
        observed = collections.defaultdict(list)

        def record_fit(original, name):
            @functools.wraps(original)
            def fit(model, X, target, *fit_args, **fit_kwargs):
                draw_ids = indices[len(observed[name])]
                np.testing.assert_array_equal(X, x[draw_ids])
                expected = y if name == "LassoCV" else t
                np.testing.assert_array_equal(np.ravel(target), expected[draw_ids])
                train_test_copies = sum(
                    np.isin(draw_ids[test], draw_ids[train]).sum() for train, test in model.cv
                )
                if train_test_copies:
                    raise ValueError("Duplicate original household in tuning training and test")
                observed[name].append(
                    {"records": len(X), "folds": len(model.cv), "overlap": int(train_test_copies)}
                )
                return original(model, X, target, *fit_args, **fit_kwargs)

            return fit

        with (
            patch.object(LassoCV, "fit", record_fit(LassoCV.fit, "LassoCV")),
            patch.object(
                LogisticRegressionCV,
                "fit",
                record_fit(LogisticRegressionCV.fit, "LogisticRegressionCV"),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            corrected = obj.analyze_bootstrap_distribution(n_bootstrap=CHECK)
        if sorted(len(rows) for rows in observed.values()) != [CHECK, CHECK]:
            raise ValueError("Both real tuning methods must be traced on all three samples")
        old, new = np.array(historical["serial"]), corrected.ate_estimates
        save(
            args.output,
            {
                "rows_per_fit": len(y),
                "fixture_sha256": digest,
                "seed": SEED,
                "zero_based_draws": list(range(CHECK)),
                "historical_position_folds": old,
                "corrected_household_folds": new,
                "paired_differences": new - old,
                "corrected_tuning_trace": dict(observed),
                "corrected_warnings": stamp(caught),
                "scope": "Three exact common resamples; no interval coverage target in the observational data",
            },
        )


if __name__ == "__main__":
    main()
