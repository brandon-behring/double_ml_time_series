"""Shared machinery for estimator result objects (Track B1).

All estimator results are ``@dataclass(frozen=True, slots=True, eq=False)``
subclasses of :class:`ResultBase`, mirroring the temporalcv result-object
idiom: shallow-frozen (field REBINDING is a hard error; ndarray *contents*
remain mutable — numpy arrays cannot be frozen by the dataclass machinery),
slotted (no silent attribute growth), and identity-equality (array-bearing
dataclasses make generated ``__eq__`` a trap — comparing them raises or
returns ambiguous arrays).
"""

from dataclasses import fields
from typing import Any


class ResultBase:
    """Base class for frozen estimator result dataclasses.

    ``__post_init__`` dispatches to :meth:`_validate`; subclasses fill it
    with numeric hard-fails (non-positive/non-finite SEs, inverted CIs,
    p-values outside [0, 1], non-PSD covariance — the B3 validators pass).
    CAVEAT: validation fires at construction only — ``pickle.loads`` bypasses
    ``__init__``/``__post_init__``, so unpickled instances are NOT
    re-validated.
    """

    __slots__ = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Numeric validity hook — overridden per result class (B3)."""

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict of result fields (arrays returned as-is, not copied)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]
