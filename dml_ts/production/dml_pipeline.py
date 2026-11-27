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
from typing import Any, cast

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dml_ts.dml._utils import compute_r2, theta_via_fwl
from dml_ts.dml.double_ml import _influence_function_se

# Import our DML components
try:
    from temporalcv import (
        TimeSeriesCrossValidator,  # noqa: F401
        newey_west_se,
    )

    DML_AVAILABLE = True
except ImportError as _import_error:
    DML_AVAILABLE = False
    warnings.warn(
        f"Temporal CV and HAC are unavailable ({_import_error!r}).",
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
        propensity_model: Classifier for 0/1 treatment, or regressor for numeric dose
        outcome_model: Model for outcome regression
        use_hac: Use HAC standard errors for time series
        hac_bandwidth: Bandwidth for HAC (None = Andrews data-driven selection)
        monitoring_enabled: Enable companion monitoring utilities
        feature_columns: List of feature/covariate column names
        treatment_column: Treatment variable name
        outcome_column: Outcome variable name
        time_column: Enables forward splits on rows supplied in chronological order
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
        ate: Binary average treatment effect or continuous partially linear slope
        ate_se: Standard error of ATE
        ate_ci_lower: Lower bound of 95% CI
        ate_ci_upper: Upper bound of 95% CI
        cate: Conditional ATE (if computed)
        nuisance_metrics: Held-out R² for treatment and observed-arm outcome predictions
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
    Research DML pipeline for insurance/annuity pricing analysis.

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
        self._binary_treatment = False
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

        Raises:
            ValueError: If the inputs are inconsistent (see the checks below), or, after
                cross-fitting, if the monitor refuses its inputs: a ``baseline_ate`` that is
                NaN or infinite, or a ``baseline_ate_se`` that is negative or not finite.
                The pipeline is then left unfitted.
        """
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float).reshape(-1)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or len(X) != len(T) or len(T) != len(Y):
            raise ValueError("X, T and Y must have matching row counts.")
        if not all(np.isfinite(values).all() for values in (X, T, Y)):
            raise ValueError("X, T and Y must be finite.")
        levels = np.unique(T)
        if levels.size < 2:
            raise ValueError("Treatment must vary across observations.")
        binary = np.array_equal(levels, [0, 1])
        if self.config.use_hac and not DML_AVAILABLE:
            raise RuntimeError("use_hac=True requires temporalcv HAC components.")

        treatment_learner = self.config.propensity_model
        if treatment_learner is None:
            treatment_learner = LogisticRegression(max_iter=1000) if binary else RidgeCV()
        if binary and not hasattr(treatment_learner, "predict_proba"):
            raise ValueError("The binary treatment model must provide predict_proba.")
        if not binary and is_classifier(treatment_learner):
            raise ValueError("A continuous treatment requires a regression model.")
        outcome_learner = self.config.outcome_model
        if outcome_learner is None:
            outcome_learner = RidgeCV()
        if is_classifier(outcome_learner):
            raise ValueError("Use an outcome regressor to estimate conditional outcome means.")

        n_samples = len(T)
        fold_ids = np.full(n_samples, -1, dtype=int)
        treatment_hat = np.full(n_samples, np.nan)
        outcome_hat = np.full((n_samples, 2 if binary else 1), np.nan)
        models = {}
        propensity_r2_list = []
        outcome_r2_list = []
        cv = self._create_cross_validator(n_samples)
        for fold, (train, test) in enumerate(cv.split(X)):
            if binary and np.unique(T[train]).size != 2:
                raise ValueError(f"Training fold {fold} needs observations in both treatment arms.")
            prop_model = make_pipeline(StandardScaler(), clone(treatment_learner))
            prop_model.fit(X[train], T[train])
            treatment_hat[test] = self._predict_treatment(prop_model, X[test])

            outcome_models = []
            for column in range(outcome_hat.shape[1]):
                rows = train[T[train] == column] if binary else train
                out_model = make_pipeline(StandardScaler(), clone(outcome_learner))
                out_model.fit(X[rows], Y[rows])
                outcome_hat[test, column] = out_model.predict(X[test])
                outcome_models.append(out_model)
            if (
                not np.isfinite(treatment_hat[test]).all()
                or not np.isfinite(outcome_hat[test]).all()
            ):
                raise ValueError(f"Non-finite nuisance predictions in fold {fold}.")
            observed_hat = (
                outcome_hat[test, T[test].astype(int)] if binary else outcome_hat[test, 0]
            )
            fold_ids[test] = fold
            propensity_r2_list.append(compute_r2(T[test], treatment_hat[test]))
            outcome_r2_list.append(compute_r2(Y[test], observed_hat))
            models[fold] = (prop_model, tuple(outcome_models) if binary else outcome_models[0])

        scored = fold_ids >= 0
        y, treatment, m = Y[scored], T[scored], treatment_hat[scored]
        n_used = len(y)
        if n_used < 2:
            raise ValueError("At least two held-out observations are required for inference.")
        if binary:
            if np.unique(treatment).size != 2:
                raise ValueError("The scored sample must contain both treatment arms.")
            mu0, mu1 = outcome_hat[scored].T
            e = np.clip(m, 0.01, 0.99)
            pseudo_outcome = (
                mu1 - mu0 + treatment * (y - mu1) / e - (1 - treatment) * (y - mu0) / (1 - e)
            )
            ate = float(np.mean(pseudo_outcome))
            psi = pseudo_outcome - ate
            se = float(np.std(psi, ddof=1) / np.sqrt(n_used))
            observed_hat = np.where(treatment == 1, mu1, mu0)
        else:
            observed_hat = outcome_hat[scored, 0]
            y_residual = y - observed_hat
            t_residual = treatment - m
            ate, _ = theta_via_fwl(
                y_residual,
                t_residual,
                treatment=treatment,
                no_variation_msg="Continuous treatment has no residual variation on scored rows.",
            )
            se, psi = _influence_function_se(y_residual, t_residual, ate)

        bandwidth = None
        if self.config.use_hac:
            bw = self.config.hac_bandwidth if self.config.hac_bandwidth is not None else "andrews"
            hac = newey_west_se(psi, bandwidth=bw)
            se, bandwidth = float(hac.se), int(hac.bandwidth)
        if not np.isfinite(se) or se <= 0:
            raise ValueError("The score standard error must be finite and positive.")
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        nuisance_metrics = {
            "r2_propensity": compute_r2(treatment, m),
            "r2_outcome": compute_r2(y, observed_hat),
            "r2_propensity_min": float(np.min(propensity_r2_list)),
            "r2_outcome_min": float(np.min(outcome_r2_list)),
        }

        # Run monitoring
        monitoring_results = []
        if self._monitor is not None:
            monitoring_results = self._monitor.run_all_checks(
                propensity_scores=m if binary else None,
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
            "propensity_model": type(treatment_learner).__name__,
            "outcome_model": type(outcome_learner).__name__,
        }

        feature_names = (
            self.config.feature_columns
            if self.config.feature_columns
            else [f"X_{i}" for i in range(X.shape[1])]
        )

        version = DMLModelVersion.create(
            model_type="double_ml",
            nuisance_models=models,
            feature_names=feature_names,
            treatment_name=self.config.treatment_column,
            outcome_name=self.config.outcome_column,
            hyperparameters=hyperparameters,
            metrics=nuisance_metrics,
            metadata={
                "binary_treatment": binary,
                "fold_scaling": True,
                "custom_treatment_model": self.config.propensity_model is not None,
                "custom_outcome_model": self.config.outcome_model is not None,
            },
        )

        # Register model
        version_id = self._registry.register(version)
        self._current_version = version
        self._nuisance_models = models
        self._scaler = None  # Legacy saves can contain a single external scaler.
        self._ate, self._ate_se = ate, se
        self._binary_treatment = binary
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
                "n_features": X.shape[1],
                "n_used": n_used,
                "n_unscored": n_samples - n_used,
                "estimand": "binary_ate" if binary else "continuous_plr_slope",
                "hac_bandwidth": bandwidth,
                "n_folds": self.config.n_folds,
            },
        )

    @staticmethod
    def _predict_treatment(model: Any, x: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            (treated,) = np.flatnonzero(model.classes_ == 1)
            return np.asarray(model.predict_proba(x))[:, treated]
        return np.asarray(model.predict(x))

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of treatment or the conditional mean of a numeric dose.

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

        X_scaled = self._scaler.transform(X) if self._scaler is not None else X
        n_samples = len(X_scaled)

        # Average across folds
        propensity_sum = np.zeros(n_samples)
        for _fold_idx, (prop_model, _) in self._nuisance_models.items():
            propensity_sum += self._predict_treatment(prop_model, X_scaled)

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

        # A probability overlap check applies to a binary assignment.
        propensity_new = self.predict_propensity(X_new) if self._binary_treatment else None

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
            "model_registry_path": str(self.config.model_registry_path),
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

        if self._current_version is not None:
            self._current_version.save(path)
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

        # Load state
        with open(path / "state.json") as f:
            state = json.load(f)

        pipeline._fitted = state["fitted"]
        pipeline._ate = state["ate"]
        pipeline._ate_se = state["ate_se"]

        # Load current version if available
        if state["current_version_id"]:
            version_id = state["current_version_id"]
            if (path / version_id / "metadata.json").exists():
                version = DMLModelVersion.load(path / version_id)
            else:
                version = pipeline._registry.get(version_id)
            pipeline._current_version = version
            pipeline._nuisance_models = cast(
                dict[int, tuple[Any, Any]], version.get_nuisance_models()
            )
            first_treatment_model = pipeline._nuisance_models[0][0]
            pipeline._binary_treatment = version.metadata.get(
                "binary_treatment", hasattr(first_treatment_model, "predict_proba")
            )
            if version.metadata.get("custom_treatment_model"):
                config.propensity_model = clone(first_treatment_model[-1])
            if version.metadata.get("custom_outcome_model"):
                stored_outcome = pipeline._nuisance_models[0][1]
                if pipeline._binary_treatment:
                    stored_outcome = stored_outcome[0]
                config.outcome_model = clone(stored_outcome[-1])
            if not version.metadata.get("fold_scaling"):
                pipeline._scaler = version.get_feature_transformer()
                if pipeline._scaler is None and (path / "scaler.pkl").exists():
                    with open(path / "scaler.pkl", "rb") as f:
                        pipeline._scaler = pickle.load(f)

        return pipeline

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        version = self._current_version.version_id if self._current_version else "none"
        return f"InsuranceDMLPipeline(status={status}, version={version})"
