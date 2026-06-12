"""Double ML Time Series -- ``dml_ts`` package root.

The four headline estimators are re-exported here for convenience, so readers can
write ``from dml_ts import TemporalPLRDML``. Loaders, DGPs, sensitivity, and
production utilities deliberately stay namespaced under their subpackages (e.g.
``from dml_ts.data import FREDLoader``) to keep ``import dml_ts`` lightweight --
importing the package eagerly pulls in only ``dml`` and ``validation``.
"""

from . import dml, validation
from .dml import (
    DynamicGEstimationDML,
    PanelDML,
    RollingWindowDML,
    TemporalPLRDML,
    double_ml,
)

__version__ = "1.1.1"

__all__ = [
    "dml",
    "validation",
    "double_ml",
    "TemporalPLRDML",
    "RollingWindowDML",
    "PanelDML",
    "DynamicGEstimationDML",
    "__version__",
]
