"""Keep all copies of an original sampling unit together in pairs-bootstrap cross-fitting."""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def original_fold_ids(treatment: np.ndarray, seed: int | None, *, stratified: bool) -> np.ndarray:
    """Make one shuffled fivefold partition before resampling the original observations."""
    if stratified:
        splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(treatment, treatment)
    else:
        splits = KFold(5, shuffle=True, random_state=seed).split(treatment)
    ids = np.empty(len(treatment), dtype=int)
    for k, (_, test) in enumerate(splits):
        ids[test] = k
    return ids


def resampled_folds(
    fold_ids: np.ndarray, source_indices: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map original fold membership onto positions of the resampled copies."""
    draw_folds = fold_ids[source_indices]
    represented = np.unique(draw_folds)
    if represented.size < 2:
        raise ValueError("Bootstrap sample needs at least two represented original-unit folds")
    return [(np.flatnonzero(draw_folds != k), np.flatnonzero(draw_folds == k)) for k in represented]
