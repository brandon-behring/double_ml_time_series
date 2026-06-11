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

    ``__post_init__`` dispatches to :meth:`_validate`, which is a deliberate
    no-op today — the B3 numeric-validators pass fills it with hard-fails at
    the arithmetic boundary (non-finite SEs, inverted CIs, ...).
    """

    __slots__ = ()

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Numeric validity hook — filled by the B3 validators PR."""

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict of result fields (arrays returned as-is, not copied)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]
