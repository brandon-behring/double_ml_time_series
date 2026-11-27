"""Standard errors of a scalar panel score: unit clusters, two-way clusters, date HAC.

Private module. ``PanelDML`` does not use it yet; its redesign will (see
``docs/plans/active/2026-11-27_paneldml_design_gate.md``, §4.7).
"""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from temporalcv import finite_se, newey_west_se

PanelInference = Literal["unit_cluster", "two_way_cluster", "date_hac"]

_INFERENCE = ("unit_cluster", "two_way_cluster", "date_hac")


def group_codes(labels: ArrayLike, n: int, name: str) -> NDArray[np.int64]:
    """Encode identifiers as integer codes, comparing them as given.

    The identifiers are not converted first: the unit ``1`` and the unit ``"1"`` are two
    units, and the units 1.00, 1.01, ..., 1.29 are thirty. Identifiers that Python treats as
    equal (``1``, ``1.0`` and ``True``) are one unit. Codes follow the identifiers' sorted
    order, numbers before strings when both occur, so dates are coded in time order; a
    pandas categorical is sorted in the order of its categories.

    Args:
        labels: One identifier per observation (list, array, Series or Index).
        n: Number of observations.
        name: Argument name used in error messages.

    Returns:
        Integer codes in ``0 .. k - 1`` for the ``k`` distinct identifiers.

    Raises:
        ValueError: If there is not one identifier per observation, if one is missing, or if
            the identifiers cannot be compared and sorted.

    Examples:
        >>> group_codes([1, "1", 2], 3, "unit").tolist()
        [0, 2, 1]
        >>> len(set(group_codes([1 + k / 100 for k in range(30)], 30, "unit")))
        30
    """
    if isinstance(labels, (pd.Series, pd.Index)):
        # A categorical keeps its own order of categories, as pandas sorts it.
        is_categorical = isinstance(labels.dtype, pd.CategoricalDtype)
        values = labels.array if is_categorical else labels.to_numpy()
    elif isinstance(labels, np.ndarray):
        values = labels
    else:
        try:
            # np.asarray would turn [1, "1"] into two equal strings.
            values = pd.Series(list(labels), dtype=object).to_numpy()  # type: ignore[arg-type]
        except TypeError:
            raise ValueError(f"{name} must contain one identifier per observation") from None
    if values.shape != (n,):
        raise ValueError(f"{name} must contain one identifier per observation")
    try:
        codes, _ = pd.factorize(values, sort=True)
    except TypeError as exc:
        raise ValueError(f"{name} identifiers cannot be compared and sorted: {exc}") from None
    if (codes < 0).any():
        raise ValueError(f"Missing {name} in panel data")
    return np.asarray(codes, dtype=np.int64)


def _cluster_variance(scores: NDArray[np.float64], groups: NDArray[np.int64]) -> float:
    """CR1 with a group-count correction on observation-weighted score sums.

    ``groups`` holds integer codes, one column per clustering dimension; rows with the same
    codes form one cluster.
    """
    _, codes = np.unique(groups, return_inverse=True, axis=0)
    sums = np.bincount(codes.reshape(-1), weights=scores)
    g = len(sums)
    if g < 2:
        raise ValueError("Cluster inference requires at least 2 retained clusters per dimension")
    return float((g / (g - 1)) * np.dot(sums, sums) / len(scores) ** 2)


def panel_standard_error(
    scores: ArrayLike, unit: ArrayLike, date: ArrayLike, inference: PanelInference
) -> tuple[float, int]:
    """Standard error of the mean of a scalar score, and the HAC bandwidth.

    ``scores`` are the influence-function values ``psi_i`` of an estimate ``theta``, so that
    ``theta - theta_0`` is approximately ``mean(psi)``. The variance of that mean is estimated
    as follows.

    - ``unit_cluster``: ``g / (g - 1) * sum_c (sum_{i in c} psi_i)**2 / n**2`` over the ``g``
      units (CR1 with the group-count factor only).
    - ``two_way_cluster``: the unit and the date variances minus the variance clustered by
      (unit, date) cell (Cameron, Gelbach and Miller), each with its own group-count factor.
      A variance that is not positive is refused.
    - ``date_hac``: the scores are summed by date, scaled by ``d / n`` for ``d`` dates, and the
      Newey-West standard error of their mean is taken in the dates' sorted order (Bartlett
      kernel, automatic bandwidth). The bandwidth is returned; it is 0 for the cluster methods.

    Args:
        scores: The influence-function values, one per observation.
        unit: Unit identifier per observation, compared as given (see ``group_codes``).
        date: Date identifier per observation, compared as given and sorted for ``date_hac``.
        inference: ``"unit_cluster"``, ``"two_way_cluster"`` or ``"date_hac"``.

    Returns:
        ``(se, bandwidth)``.

    Raises:
        ValueError: For an unknown method, scores that are not a finite vector, identifiers
            that ``group_codes`` refuses, fewer than two clusters or dates, or a variance that
            is not finite and positive.

    Examples:
        >>> se, bandwidth = panel_standard_error(
        ...     [1.0, 0.0, 0.5, 0.5], ["a", "a", "b", "b"], [1, 2, 1, 2], "unit_cluster"
        ... )
        >>> round(se, 6), bandwidth
        (0.5, 0)
    """
    if inference not in _INFERENCE:
        raise ValueError(f"Unknown panel inference method {inference!r}")
    psi = np.asarray(scores, dtype=np.float64)
    if psi.ndim != 1 or not np.all(np.isfinite(psi)):
        raise ValueError("scores must be a finite vector, one value per observation")
    n = len(psi)
    if inference == "date_hac":
        date_codes = group_codes(date, n, "date")
        sums = np.bincount(date_codes, weights=psi)
        if len(sums) < 2:
            raise ValueError("Date HAC requires at least 2 retained dates")
        date_scores = sums * (len(sums) / n)
        result = newey_west_se(date_scores, bandwidth="auto")
        finite_se(result.se, name="date_hac standard error")
        return float(result.se), int(result.bandwidth)
    u = group_codes(unit, n, "unit")
    variance = _cluster_variance(psi, u[:, None])
    if inference == "two_way_cluster":
        d = group_codes(date, n, "date")
        variance += _cluster_variance(psi, d[:, None]) - _cluster_variance(
            psi, np.column_stack((u, d))
        )
    finite_se(variance, name=f"{inference} variance")
    return float(np.sqrt(variance)), 0
