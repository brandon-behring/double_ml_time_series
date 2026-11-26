"""Check original-row overlap in the exact five-fold bootstrap used by the Lasso diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold
from standalone_lasso import COUNT, SEED, diagnostic


class FoldsRecorded(Exception):
    """End the original fit immediately after EconML constructs the outer folds."""


def capture_folds(obj, rng_state):
    captured = []

    def record(models, folds, *args, **kwargs):
        captured.extend(list(folds))
        raise FoldsRecorded

    obj._rng.set_state(rng_state)
    with patch("econml._ortho_learner._crossfit", record):
        try:
            obj.analyze_bootstrap_distribution(n_bootstrap=1)
        except FoldsRecorded:
            pass
    if len(captured) != 5:
        raise ValueError("Expected to capture all five outer folds before fitting")
    return captured


def overlap(original, folds, labels):
    copied = loss = 0
    numerator = denominator = 0.0
    for train, test in folds:
        seen = np.bincount(original[train], minlength=len(original)) > 0
        in_training = seen[original[test]]
        # Replace observed treatment and assets with randomized T and Y=2T.
        # Signs are 2T-1. The population outcome nuisance is one for new IDs.
        lookup = np.ones(len(original))
        lookup[original[train]] = 1 + labels[original[train]]
        outcome = 1 + labels[original[test]]
        residual_y = outcome - lookup[original[test]]
        residual_t = labels[original[test]] / 2  # T minus the known assignment probability.
        loss += np.square(residual_y).sum()
        copied += in_training.sum()
        numerator += residual_t @ residual_y
        denominator += residual_t @ residual_t
    return int(copied), int(loss), numerator / denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    obj = diagnostic(args.snapshot.resolve())
    _, t, _ = obj.preprocess_data()
    n = len(t)
    rng = np.random.RandomState(SEED)
    labels = np.random.default_rng(413).choice([-1, 1], size=n)
    zero = np.zeros(n)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    original_fold = np.empty(n, dtype=int)
    for k, (_, test) in enumerate(splitter.split(zero, t)):
        original_fold[test] = k
    rows = []
    digest = hashlib.sha256()
    checked = [0, COUNT // 2 - 1, COUNT - 1]
    for draw in range(COUNT):
        state = rng.get_state()
        original = rng.choice(n, size=n, replace=True)
        digest.update(original.astype("<i8", copy=False).tobytes())
        folds = list(splitter.split(zero, t[original]))
        if draw in checked:
            actual = capture_folds(obj, state)
            for a, b in zip(actual, folds, strict=True):
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
        copied, loss, effect = overlap(original, folds, labels)
        inherited = original_fold[original]
        by_original = [
            (np.flatnonzero(inherited != k), np.flatnonzero(inherited == k)) for k in range(5)
        ]
        grouped_copied, grouped_loss, grouped_effect = overlap(original, by_original, labels)
        if grouped_copied or grouped_loss != n or copied + loss != n or grouped_effect != 2:
            raise ValueError("Control error or duplicated-ID scoring mismatch")
        rows.append([len(np.unique(original)), copied, loss, effect, grouped_effect])
    summary = np.asarray(rows)
    total = n * COUNT
    data = {
        "sampling_unit": "Original fixture row, identified by its zero-based source index",
        "rows_per_draw": n,
        "draws": COUNT,
        "folds_per_draw": 5,
        "original_econml_fold_arrays_checked_on_zero_based_draws": checked,
        "index_sha256": digest.hexdigest(),
        "original_ids_per_draw_mean": summary[:, 0].mean(),
        "scored_bootstrap_rows": total,
        "copies_in_training_total": int(summary[:, 1].sum()),
        "copies_in_training_fraction": summary[:, 1].sum() / total,
        "fraction_by_draw_range": (summary[:, 1].min() / n, summary[:, 1].max() / n),
        "inherited_original_unit_folds_copies_in_training_total": 0,
        "memorized_random_label_mse_bootstrap_position_folds": summary[:, 2].sum() / total,
        "memorized_random_label_mse_inherited_original_unit_folds": 1.0,
        "randomized_planted_eligibility_effect": 2.0,
        "plr_with_known_assignment_and_lookup_outcome_nuisance": {
            "bootstrap_position_folds_mean": summary[:, 3].mean(),
            "bootstrap_position_folds_range": [summary[:, 3].min(), summary[:, 3].max()],
            "inherited_original_unit_folds_mean": summary[:, 4].mean(),
        },
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
