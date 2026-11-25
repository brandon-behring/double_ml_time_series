"""Compare single-partition EconML fits on the 1991 SIPP 401(k) sample.

Chernozhukov et al. (2018), Table 2, Panel B, report fivefold partially
linear eligibility coefficients of $9,247 (forest) and $8,187 (rigorous
lasso), aggregated over 100 partitions. The companion uses a single
fivefold partition, allows the effect to vary linearly with its input
covariates, and reports the mean fitted effect on net financial assets.
On the distributed sample its eleven inputs include two wealth measures,
``nifa`` and ``tw``, in addition to the nine published controls.

Table 3 instruments participation (``p401``) with eligibility (``e401``)
and estimates a complier effect. Fitting participation directly in this
module measures an adjusted participation contrast with a different target.

Reference: https://doi.org/10.1111/ectj.12097
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass
class ReplicationResult:
    """Mean fitted effect and its descriptive distance from a published reference.

    Attributes:
        method: Companion learner identifier ("PLR_RF" or "PLR_Lasso").
        ate_estimate: Mean fitted effect on net financial assets, in dollars.
        std_error: EconML standard error of the mean fitted effect.
        ci_lower: Lower bound of the 95% interval for the mean effect.
        ci_upper: Upper bound of the 95% interval for the mean effect.
        published_ate: Fivefold Table 2B coefficient for eligibility, if applicable.
        difference: Signed estimate minus published reference, in dollars.
        rel_difference: Signed difference divided by the published reference.
        p_value: Legacy field. A test of the difference between estimates on
            the same sample would need their joint uncertainty. New fits return None.
        status: Descriptive distance relative to the supplied tolerance, or
            NO_BENCHMARK for directly fitted participation.
        tolerance: Relative difference cutoff (0.15 means 15%).
        timestamp: When the fit was summarized.
        metadata: Treatment, fitted specification, benchmark and run settings.
    """

    method: str
    ate_estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    published_ate: float | None
    difference: float | None
    rel_difference: float | None
    p_value: float | None
    status: Literal["WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE", "NO_BENCHMARK"]
    tolerance: float
    timestamp: datetime
    metadata: dict[str, Any]


class FourZeroOneKReplication:
    """Fit cross-sectional 401(k) companion specifications on one fivefold partition.

    Args:
        data_path: CSV with the SIPP columns, or None to fetch via doubleml.
        random_state: Seed for the partition and the nuisance learners.
        tolerance: Descriptive relative gap cutoff for the eligibility comparisons.

    Examples:
        >>> FourZeroOneKReplication.PUBLISHED_ATES["PLR_RF"]
        9247.0
    """

    # Chernozhukov et al. (2018), Table 2, Panel B, fivefold row.
    PUBLISHED_ATES = {"PLR_RF": 9247.0, "PLR_Lasso": 8187.0}

    def __init__(
        self,
        data_path: str | None = None,
        random_state: int | None = None,
        tolerance: float = 0.15,
    ):
        self.data_path = data_path
        self.random_state = random_state
        self.tolerance = tolerance
        self._data: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """Load the supplied CSV or the DoubleML 401(k) dataset, caching the frame."""
        if self._data is None:
            if self.data_path is not None:
                self._data = pd.read_csv(self.data_path)
            else:
                try:
                    from doubleml.datasets import fetch_401K
                except ImportError as e:
                    raise ImportError(
                        "doubleml is required for fetching the data. Install doubleml "
                        "or supply a local CSV with data_path."
                    ) from e
                self._data = fetch_401K(return_type="DataFrame")
        assert self._data is not None
        return self._data

    def _control_columns(self) -> list[str]:
        return [c for c in self.load_data().columns if c not in {"net_tfa", "e401", "p401"}]

    def preprocess_data(self, treatment: str = "e401") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return net financial assets, the treatment, and the companion feature matrix.

        The default CSV column order is preserved. Every column except the
        outcome and the two treatment indicators becomes an effect modifier.
        """
        if treatment not in {"e401", "p401"}:
            raise ValueError(f"treatment must be 'e401' or 'p401', got {treatment}")
        df = self.load_data()
        features = df[self._control_columns()]
        return df["net_tfa"].to_numpy(), df[treatment].to_numpy(), features.to_numpy()

    def replicate_plr_rf(
        self, treatment: str = "e401", n_estimators: int = 500, max_depth: int | None = None
    ) -> ReplicationResult:
        """Fit a conditional linear effect using random-forest nuisance models.

        Uses a regression forest for wealth and a classification forest for
        binary treatment. Both use the requested tree count and depth, and
        a minimum leaf size of ten.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        forest_options = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": 10,
            "random_state": self.random_state,
        }
        return self._fit(
            "PLR_RF",
            treatment,
            RandomForestRegressor(**forest_options),
            RandomForestClassifier(**forest_options),
            {"n_estimators": n_estimators, "max_depth": max_depth},
        )

    def replicate_plr_lasso(
        self, treatment: str = "e401", alpha: float = 0.01
    ) -> ReplicationResult:
        """Fit a conditional linear effect using cross-validated lasso nuisances.

        The legacy alpha argument is recorded for compatibility. LassoCV and
        LogisticRegressionCV select their own penalties on their training data.
        """
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        return self._fit(
            "PLR_Lasso",
            treatment,
            LassoCV(cv=5, random_state=self.random_state),
            LogisticRegressionCV(
                cv=5, penalty="l1", solver="saga", random_state=self.random_state, max_iter=1000
            ),
            {"alpha": alpha, "penalty_selection": "fivefold cross-validation"},
        )

    def _fit(
        self, method: str, treatment: str, model_y: Any, model_t: Any, settings: dict[str, Any]
    ) -> ReplicationResult:
        from econml.dml import LinearDML

        y, t, x = self.preprocess_data(treatment)
        model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=5,
            random_state=self.random_state,
        )
        model.fit(Y=y, T=t, X=x)
        inference = model.ate_inference(X=x)
        estimate = float(np.squeeze(inference.mean_point))
        lower, upper = inference.conf_int_mean(alpha=0.05)

        published = self.PUBLISHED_ATES[method] if treatment == "e401" else None
        difference = None
        rel_difference = None
        status: Literal["WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE", "NO_BENCHMARK"] = "NO_BENCHMARK"
        if published is not None:
            difference = estimate - published
            rel_difference = difference / published
            status = (
                "WITHIN_TOLERANCE" if abs(rel_difference) <= self.tolerance else "OUTSIDE_TOLERANCE"
            )

        return ReplicationResult(
            method=method,
            ate_estimate=estimate,
            std_error=float(np.squeeze(inference.stderr_mean)),
            ci_lower=float(np.squeeze(lower)),
            ci_upper=float(np.squeeze(upper)),
            published_ate=published,
            difference=difference,
            rel_difference=rel_difference,
            p_value=None,
            status=status,
            tolerance=self.tolerance,
            timestamp=datetime.now(),
            metadata={
                "treatment": treatment,
                "estimand": (
                    "Mean fitted eligibility effect"
                    if treatment == "e401"
                    else "Adjusted participation contrast under direct treatment modeling"
                ),
                "benchmark": (
                    "Table 2, Panel B, fivefold partially linear eligibility coefficient"
                    if published is not None
                    else None
                ),
                "n_obs": len(y),
                "n_controls": x.shape[1],
                "control_columns": self._control_columns(),
                "control_role": "EconML X (linear effect modifiers and nuisance inputs)",
                "n_folds": 5,
                "n_partitions": 1,
                "random_state": self.random_state,
                **settings,
            },
        )
