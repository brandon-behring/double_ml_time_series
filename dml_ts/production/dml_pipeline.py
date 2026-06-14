"""
Research/demo DML pipeline utilities.

This module provides InsuranceDMLPipeline as book-companion code for organizing
insurance/annuity competitor pricing examples. It is not deployment-ready
production infrastructure.

Pipeline stages:
1. Data ingestion (FRED macro + insurance pricing data)
2. Feature engineering (transformers, interactions)
3. DML-style estimation helpers
4. Monitoring integration (causal-specific checks)
5. Model versioning (registry integration)
6. Inference-style demo methods

Key design principle: Separate training and inference paths clearly,
as DML requires different handling than standard ML prediction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

# Import our DML components
try:
    from temporalcv import (
        TimeSeriesCrossValidator,  # noqa: F401
        newey_west_se,
    )

    from dml_ts.dml.double_ml import double_ml  # noqa: F401
    from dml_ts.dml.temporal_plr_dml import TemporalPLRDML  # noqa: F401

    DML_AVAILABLE = True
except ImportError as _import_error:
    DML_AVAILABLE = False
    warnings.warn(
        f"DML/HAC components unavailable ({_import_error!r}); the demo "
        "pipeline will fall back to NAIVE iid standard errors, which "
        "understate uncertainty under autocorrelation.",
        RuntimeWarning,
        stacklevel=2,
    )

# Import registry and monitoring components
from dml_ts.production.causal_monitor import CausalMonitor, MonitoringResult
from dml_ts.production.model_registry import DMLModelRegistry, DMLModelVersion
from dml_ts.production.retrain_pipeline import RetrainScheduler, RetrainTrigger


@dataclass
class PipelineConfig:
    """
    Configuration for InsuranceDMLPipeline.

    Attributes:
        n_folds: Number of cross-fitting folds
        n_jobs: Parallel jobs for estimation
        random_state: Random seed for reproducibility
        model_registry_path: Path for model versioning
        propensity_model: Model for treatment propensity
        outcome_model: Model for outcome regression
        use_hac: Use HAC standard errors for time series
        hac_bandwidth: Bandwidth for HAC (None = Andrews data-driven selection)
        monitoring_enabled: Enable companion monitoring utilities
        feature_columns: List of feature/covariate column names
        treatment_column: Treatment variable name
        outcome_column: Outcome variable name
        time_column: Time index column name (for time series)
        entity_column: Entity/panel identifier (for panel data)
    """

    n_folds: int = 5
    n_jobs: int = -1
    random_state: int = 42
    model_registry_path: str = "./models/dml_registry"
    propensity_model: Any | None = None
    outcome_model: Any | None = None
    use_hac: bool = True
    hac_bandwidth: int | None = None
    monitoring_enabled: bool = True
    feature_columns: list[str] = field(default_factory=list)
    treatment_column: str = "treatment"
    outcome_column: str = "outcome"
    time_column: str | None = None
    entity_column: str | None = None


@dataclass
class PipelineResult:
    """
    Result from a pipeline fit or predict operation.

    Attributes:
        ate: Average Treatment Effect estimate
        ate_se: Standard error of ATE
        ate_ci_lower: Lower bound of 95% CI
        ate_ci_upper: Upper bound of 95% CI
        cate: Conditional ATE (if computed)
        nuisance_metrics: R² for propensity and outcome models
        monitoring_results: Results from causal monitoring
        model_version: Version ID of fitted model
        fit_timestamp: When model was fitted
        metadata: Additional metadata
    """

    ate: float
    ate_se: float
    ate_ci_lower: float
    ate_ci_upper: float
    cate: np.ndarray | None = None
    nuisance_metrics: dict[str, float] = field(default_factory=dict)
    monitoring_results: list[MonitoringResult] = field(default_factory=list)
    model_version: str | None = None
    fit_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ate": self.ate,
            "ate_se": self.ate_se,
            "ate_ci_lower": self.ate_ci_lower,
            "ate_ci_upper": self.ate_ci_upper,
            "nuisance_metrics": self.nuisance_metrics,
            "monitoring_results": [r.to_dict() for r in self.monitoring_results],
            "model_version": self.model_version,
            "fit_timestamp": self.fit_timestamp,
            "metadata": self.metadata,
        }


class InsuranceDMLPipeline:
    """
    Production DML pipeline for insurance/annuity pricing analysis.

    Integrates:
    - Feature transformation (scaling, encoding)
    - DML estimation with time series cross-validation
    - HAC standard errors for autocorrelated data
    - Model versioning and registry
    - Causal-specific monitoring
    - Retraining triggers

    Example:
        >>> import tempfile
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(60, 2))
        >>> T = rng.binomial(1, 0.5, size=60)
        >>> Y = rng.normal(size=60)
        >>> config = PipelineConfig(
        ...     feature_columns=["age", "income"],
        ...     treatment_column="competitor_price",
        ...     outcome_column="retention",
        ...     n_folds=2,
        ...     use_hac=False,
        ...     model_registry_path=tempfile.mkdtemp(),
        ...     propensity_model=LogisticRegression(),
        ...     outcome_model=LinearRegression(),
        ... )
        >>> pipeline = InsuranceDMLPipeline(config)
        >>> result = pipeline.fit(X, T, Y)
        >>> bool(np.isfinite(result.ate))
        True
    """

    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize insurance DML pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self._scaler: StandardScaler | None = None
        self._fitted = False
        self._propensity_scores: np.ndarray | None = None
        self._nuisance_models: dict[int, tuple[Any, Any]] | None = None
        self._ate: float | None = None
        self._ate_se: float | None = None
        self._current_version: DMLModelVersion | None = None

        # Initialize model registry
        self._registry = DMLModelRegistry(self.config.model_registry_path)

        # Initialize monitoring
        if self.config.monitoring_enabled:
            self._monitor = CausalMonitor()
            self._scheduler = RetrainScheduler(monitor=self._monitor)
        else:
            self._monitor = None
            self._scheduler = None

        # Set default models if not provided
        if self.config.propensity_model is None:
            from sklearn.linear_model import LogisticRegressionCV

            self.config.propensity_model = LogisticRegressionCV(cv=3, max_iter=1000)

        if self.config.outcome_model is None:
            from sklearn.linear_model import RidgeCV

            self.config.outcome_model = RidgeCV(cv=3)

    def _prepare_data(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for DML estimation.

        Args:
            X: Covariate matrix (n_samples, n_features)
            T: Treatment vector (n_samples,)
            Y: Outcome vector (n_samples,)

        Returns:
            Tuple of (X_scaled, T, Y) arrays
        """
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        Y = np.asarray(Y).flatten()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Fit or transform with scaler
        if self._scaler is None:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = self._scaler.transform(X)

        return X_scaled, T, Y

    def _create_cross_validator(self, n_samples: int) -> TimeSeriesCrossValidator | Any:
        """
        Create appropriate cross-validator.

        Uses TimeSeriesCrossValidator if time_column is specified,
        otherwise uses standard KFold.

        Args:
            n_samples: Number of samples

        Returns:
            Cross-validator instance
        """
        if self.config.time_column is not None and not DML_AVAILABLE:
            # Refuse the silent downgrade: shuffled KFold on time-series
            # data is leakage, not a fallback (same family as issue #11).
            raise RuntimeError(
                "time_column is set but temporal CV components are "
                "unavailable (imports failed at module load). Refusing to "
                "silently fall back to shuffled KFold on time-series data."
            )
        if self.config.time_column is not None and DML_AVAILABLE:
            return TimeSeriesCrossValidator(
                n_splits=self.config.n_folds,
                gap=0,
            )
        else:
            from sklearn.model_selection import KFold

            return KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        baseline_ate: float | None = None,
        baseline_ate_se: float | None = None,
    ) -> PipelineResult:
        """
        Fit DML pipeline to data.

        Performs:
        1. Data preparation (scaling)
        2. Cross-fitted nuisance model estimation
        3. ATE estimation with orthogonalization
        4. HAC standard errors (if enabled)
        5. Monitoring checks
        6. Model versioning

        Args:
            X: Covariate matrix (n_samples, n_features)
            T: Treatment vector (n_samples,)
            Y: Outcome vector (n_samples,)
            baseline_ate: Optional baseline ATE for effect stability check
            baseline_ate_se: Optional baseline SE for effect stability check

        Returns:
            PipelineResult with estimates and diagnostics
        """
        # Prepare data
        X_scaled, T, Y = self._prepare_data(X, T, Y)
        n_samples = len(T)

        # Create cross-validator
        cv = self._create_cross_validator(n_samples)

        # Fit nuisance models and compute orthogonalized residuals
        propensity_scores = np.zeros(n_samples)
        outcome_predictions = np.zeros(n_samples)
        self._nuisance_models = {}

        propensity_r2_list = []
        outcome_r2_list = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Clone and fit propensity model
            from sklearn.base import clone

            prop_model = clone(self.config.propensity_model)
            out_model = clone(self.config.outcome_model)

            # Fit propensity (treatment) model
            if hasattr(prop_model, "predict_proba"):
                prop_model.fit(X_train, T_train)
                propensity_scores[test_idx] = prop_model.predict_proba(X_test)[:, 1]
                # Pseudo-R² for classification
                prop_r2 = prop_model.score(X_test, T_test)
            else:
                # Continuous treatment
                prop_model.fit(X_train, T_train)
                propensity_scores[test_idx] = prop_model.predict(X_test)
                prop_r2 = prop_model.score(X_test, T_test)

            # Fit outcome model (on control or full sample depending on setup)
            out_model.fit(X_train, Y_train)
            outcome_predictions[test_idx] = out_model.predict(X_test)
            out_r2 = out_model.score(X_test, Y_test)

            propensity_r2_list.append(prop_r2)
            outcome_r2_list.append(out_r2)

            # Store fitted models
            self._nuisance_models[fold_idx] = (prop_model, out_model)

        # Compute DML estimate
        # Using simplified doubly-robust estimator
        # theta = E[(Y - m(X)) / e(X) * T] + E[m(X)]
        # For binary treatment:
        # theta = mean((Y - Y_hat) * T / e_hat - (Y - Y_hat) * (1-T) / (1-e_hat))

        # Clip propensity scores for stability
        e_clipped = np.clip(propensity_scores, 0.01, 0.99)

        # Residualized outcome
        Y_residual = Y - outcome_predictions

        # AIPW estimator for binary treatment
        if len(np.unique(T)) <= 2:
            # Binary treatment
            treated_term = Y_residual * T / e_clipped
            control_term = Y_residual * (1 - T) / (1 - e_clipped)
            psi = treated_term - control_term
        else:
            # Continuous treatment - use partially linear model approach
            T_residual = T - propensity_scores
            # Estimate coefficient via regression of Y_residual on T_residual
            ate_coefficient = np.sum(Y_residual * T_residual) / np.sum(T_residual**2)
            psi = Y_residual - ate_coefficient * T_residual
            self._ate = ate_coefficient

        # Compute ATE
        if len(np.unique(T)) <= 2:
            self._ate = np.mean(psi)
        ate = self._ate

        # Compute standard error
        if self.config.use_hac and DML_AVAILABLE:
            # Historical quirk preserved: the retired dml_ts hac treated
            # bandwidth=None as Andrews selection (an else-branch accident).
            # temporalcv rejects None, so map it explicitly; note temporalcv's
            # Andrews uses the literature alpha(1) constant for Bartlett
            # (documented deviation from the old implementation).
            bw = self.config.hac_bandwidth if self.config.hac_bandwidth is not None else "andrews"
            se = float(newey_west_se(psi, bandwidth=bw).se)
        else:
            if self.config.use_hac and not DML_AVAILABLE:
                raise RuntimeError(
                    "use_hac=True but HAC components are unavailable (the "
                    "temporalcv/dml imports failed at module load — see the "
                    "earlier RuntimeWarning). Refusing to silently report "
                    "naive iid standard errors; set use_hac=False to opt "
                    "into them explicitly."
                )
            se = np.std(psi) / np.sqrt(n_samples)

        self._ate_se = se
        self._propensity_scores = propensity_scores

        # Compute confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # Aggregate nuisance metrics
        nuisance_metrics = {
            "r2_propensity": float(np.mean(propensity_r2_list)),
            "r2_outcome": float(np.mean(outcome_r2_list)),
            "r2_propensity_min": float(np.min(propensity_r2_list)),
            "r2_outcome_min": float(np.min(outcome_r2_list)),
        }

        # Run monitoring
        monitoring_results = []
        if self._monitor is not None:
            monitoring_results = self._monitor.run_all_checks(
                propensity_scores=propensity_scores,
                r2_propensity=nuisance_metrics["r2_propensity"],
                r2_outcome=nuisance_metrics["r2_outcome"],
                current_effect=ate,
                baseline_effect=baseline_ate,
                current_se=se,
                baseline_se=baseline_ate_se,
            )

        # Create model version
        hyperparameters = {
            "n_folds": self.config.n_folds,
            "use_hac": self.config.use_hac,
            "hac_bandwidth": self.config.hac_bandwidth,
            "propensity_model": str(type(self.config.propensity_model).__name__),
            "outcome_model": str(type(self.config.outcome_model).__name__),
        }

        feature_names = (
            self.config.feature_columns
            if self.config.feature_columns
            else [f"X_{i}" for i in range(X_scaled.shape[1])]
        )

        self._current_version = DMLModelVersion.create(
            model_type="double_ml",
            nuisance_models=self._nuisance_models,
            feature_names=feature_names,
            treatment_name=self.config.treatment_column,
            outcome_name=self.config.outcome_column,
            hyperparameters=hyperparameters,
            metrics=nuisance_metrics,
            feature_transformer=self._scaler,
        )

        # Register model
        version_id = self._registry.register(self._current_version)

        self._fitted = True

        return PipelineResult(
            ate=ate,
            ate_se=se,
            ate_ci_lower=ci_lower,
            ate_ci_upper=ci_upper,
            nuisance_metrics=nuisance_metrics,
            monitoring_results=monitoring_results,
            model_version=version_id,
            metadata={
                "n_samples": n_samples,
                "n_features": X_scaled.shape[1],
                "n_folds": self.config.n_folds,
            },
        )

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """
        Predict treatment propensity for new data.

        Uses averaged predictions across all cross-fitting folds.

        Args:
            X: Covariate matrix (n_samples, n_features)

        Returns:
            Propensity score predictions (n_samples,)

        Raises:
            ValueError: If pipeline not fitted
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before prediction")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self._scaler.transform(X)
        n_samples = X_scaled.shape[0]

        # Average across folds
        propensity_sum = np.zeros(n_samples)
        for _fold_idx, (prop_model, _) in self._nuisance_models.items():
            if hasattr(prop_model, "predict_proba"):
                propensity_sum += prop_model.predict_proba(X_scaled)[:, 1]
            else:
                propensity_sum += prop_model.predict(X_scaled)

        return propensity_sum / len(self._nuisance_models)

    def evaluate_retrain_need(
        self,
        X_new: np.ndarray,
        T_new: np.ndarray,
        X_baseline: np.ndarray | None = None,
        T_baseline: np.ndarray | None = None,
    ) -> tuple[list[MonitoringResult], RetrainTrigger | None]:
        """
        Evaluate if retraining is needed based on new data.

        Args:
            X_new: New covariate data
            T_new: New treatment data
            X_baseline: Optional baseline covariates for comparison
            T_baseline: Optional baseline treatment for comparison

        Returns:
            Tuple of (monitoring_results, retrain_trigger_or_none)

        Raises:
            ValueError: If monitoring not enabled
        """
        if self._scheduler is None:
            raise ValueError("Monitoring must be enabled for retrain evaluation")

        if not self._fitted:
            raise ValueError("Pipeline must be fitted before evaluation")

        # Predict propensity on new data
        propensity_new = self.predict_propensity(X_new)

        # Get nuisance metrics
        r2_prop = self._current_version.metrics.get("r2_propensity", 0.5)
        r2_out = self._current_version.metrics.get("r2_outcome", 0.5)

        return self._scheduler.evaluate_and_monitor(
            propensity_scores=propensity_new,
            treatment_current=T_new,
            treatment_baseline=T_baseline,
            r2_propensity=r2_prop,
            r2_outcome=r2_out,
            X_current=X_new,
            X_baseline=X_baseline,
        )

    def promote_to_production(self) -> str:
        """
        Mark current model as the demo registry's production slot.

        Returns:
            Version ID assigned to the production slot
        """
        if self._current_version is None:
            raise ValueError("No model fitted to promote")

        # First promote to staging
        self._registry.promote_to_staging(self._current_version.version_id)

        # Then to production
        return self._registry.promote_to_production()

    def rollback(self, to_version: str | None = None) -> str:
        """
        Roll back to previous production-slot version.

        Args:
            to_version: Specific version to rollback to (None = previous)

        Returns:
            Version ID rolled back to
        """
        return self._registry.rollback(to_version)

    def get_production_model(self) -> DMLModelVersion | None:
        """Return current production model, or None if not set."""
        return self._registry.get_production()

    def list_versions(self) -> list[dict[str, Any]]:
        """List all registered model versions."""
        return self._registry.list_versions()

    def save(self, path: str | Path) -> Path:
        """
        Save pipeline state.

        Args:
            path: Directory to save to

        Returns:
            Path to saved pipeline
        """
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_data = {
            "n_folds": self.config.n_folds,
            "n_jobs": self.config.n_jobs,
            "random_state": self.config.random_state,
            "use_hac": self.config.use_hac,
            "hac_bandwidth": self.config.hac_bandwidth,
            "monitoring_enabled": self.config.monitoring_enabled,
            "feature_columns": self.config.feature_columns,
            "treatment_column": self.config.treatment_column,
            "outcome_column": self.config.outcome_column,
            "time_column": self.config.time_column,
            "entity_column": self.config.entity_column,
        }

        import json

        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Save scaler
        if self._scaler is not None:
            with open(path / "scaler.pkl", "wb") as f:
                pickle.dump(self._scaler, f)

        # Save fitted state
        state = {
            "fitted": self._fitted,
            "ate": self._ate,
            "ate_se": self._ate_se,
            "current_version_id": (
                self._current_version.version_id if self._current_version else None
            ),
        }
        with open(path / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        return path

    @classmethod
    def load(cls, path: str | Path) -> InsuranceDMLPipeline:
        """
        Load pipeline from saved state.

        Args:
            path: Directory containing saved pipeline

        Returns:
            Loaded InsuranceDMLPipeline instance
        """
        import json
        import pickle

        path = Path(path)

        # Load configuration
        with open(path / "config.json") as f:
            config_data = json.load(f)

        config = PipelineConfig(**config_data)
        pipeline = cls(config)

        # Load scaler
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                pipeline._scaler = pickle.load(f)

        # Load state
        with open(path / "state.json") as f:
            state = json.load(f)

        pipeline._fitted = state["fitted"]
        pipeline._ate = state["ate"]
        pipeline._ate_se = state["ate_se"]

        # Load current version if available
        if state["current_version_id"]:
            try:
                pipeline._current_version = pipeline._registry.get(state["current_version_id"])
                # Restore nuisance models
                models = pipeline._current_version.get_nuisance_models()
                if isinstance(models, dict):
                    pipeline._nuisance_models = models
            except KeyError:
                pass  # Version not in registry

        return pipeline

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        version = self._current_version.version_id if self._current_version else "none"
        return f"InsuranceDMLPipeline(status={status}, version={version})"
