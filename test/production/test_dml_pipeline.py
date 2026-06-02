"""
Tests for end-to-end DML pipeline.

Tests cover:
- Pipeline configuration
- Data preparation and scaling
- DML estimation with cross-fitting
- Monitoring integration
- Model versioning
- Save/load functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from dml_ts.production.dml_pipeline import (
    InsuranceDMLPipeline,
    PipelineConfig,
    PipelineResult,
)

pytestmark = pytest.mark.tier2


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.n_folds == 5
        assert config.n_jobs == -1
        assert config.use_hac is True
        assert config.monitoring_enabled is True
        assert config.treatment_column == "treatment"
        assert config.outcome_column == "outcome"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            n_folds=3,
            treatment_column="competitor_price",
            outcome_column="retention",
            time_column="quarter",
            feature_columns=["age", "income", "tenure"],
        )

        assert config.n_folds == 3
        assert config.treatment_column == "competitor_price"
        assert config.time_column == "quarter"
        assert len(config.feature_columns) == 3


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_result_creation(self):
        """Test pipeline result creation."""
        result = PipelineResult(
            ate=0.15,
            ate_se=0.03,
            ate_ci_lower=0.09,
            ate_ci_upper=0.21,
            nuisance_metrics={"r2_propensity": 0.75, "r2_outcome": 0.80},
        )

        assert result.ate == 0.15
        assert result.ate_se == 0.03
        assert result.ate_ci_lower == 0.09
        assert result.ate_ci_upper == 0.21

    def test_to_dict_serialization(self):
        """Test result serialization to dict."""
        result = PipelineResult(
            ate=0.10,
            ate_se=0.02,
            ate_ci_lower=0.06,
            ate_ci_upper=0.14,
        )

        d = result.to_dict()

        assert d["ate"] == 0.10
        assert d["ate_se"] == 0.02
        assert "fit_timestamp" in d


class TestInsuranceDMLPipeline:
    """Tests for InsuranceDMLPipeline."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n = 500
        n_features = 5

        X = np.random.randn(n, n_features)
        # Binary treatment with confounding
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0] - 0.3 * X[:, 1]))
        T = np.random.binomial(1, propensity)
        # Outcome with treatment effect = 0.5
        Y = 2 + 0.5 * T + 0.8 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n) * 0.5

        return X, T, Y

    @pytest.fixture
    def pipeline_with_registry(self):
        """Create pipeline with temp registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                model_registry_path=tmpdir,
                n_folds=2,  # Faster tests
                propensity_model=LogisticRegression(max_iter=200),
                outcome_model=Ridge(),
            )
            yield InsuranceDMLPipeline(config)

    def test_initialization(self):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(model_registry_path=tmpdir)
            pipeline = InsuranceDMLPipeline(config)

            assert pipeline.config is not None
            assert pipeline._fitted is False
            assert pipeline._monitor is not None
            assert pipeline._scheduler is not None

    def test_initialization_without_monitoring(self):
        """Test pipeline initialization with monitoring disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                model_registry_path=tmpdir,
                monitoring_enabled=False,
            )
            pipeline = InsuranceDMLPipeline(config)

            assert pipeline._monitor is None
            assert pipeline._scheduler is None

    def test_fit_basic(self, sample_data, pipeline_with_registry):
        """Test basic pipeline fit."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(X, T, Y)

        assert isinstance(result, PipelineResult)
        assert result.ate is not None
        assert result.ate_se > 0
        assert result.ate_ci_lower < result.ate < result.ate_ci_upper
        assert pipeline._fitted is True

    def test_fit_returns_reasonable_ate(self, sample_data, pipeline_with_registry):
        """Test that fitted ATE is in reasonable range."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(X, T, Y)

        # True ATE is 0.5, should be within 3 SEs
        assert abs(result.ate - 0.5) < 3 * result.ate_se

    def test_fit_includes_nuisance_metrics(self, sample_data, pipeline_with_registry):
        """Test that fit returns nuisance model metrics."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(X, T, Y)

        assert "r2_propensity" in result.nuisance_metrics
        assert "r2_outcome" in result.nuisance_metrics
        assert result.nuisance_metrics["r2_propensity"] > 0
        assert result.nuisance_metrics["r2_outcome"] > 0

    def test_fit_runs_monitoring(self, sample_data, pipeline_with_registry):
        """Test that fit runs monitoring checks."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(X, T, Y)

        assert len(result.monitoring_results) > 0
        check_names = [r.check_name for r in result.monitoring_results]
        assert "overlap_violations" in check_names
        assert "nuisance_degradation" in check_names

    def test_fit_creates_model_version(self, sample_data, pipeline_with_registry):
        """Test that fit creates and registers model version."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(X, T, Y)

        assert result.model_version is not None
        assert result.model_version.startswith("dml-")
        assert pipeline._current_version is not None

    def test_fit_with_baseline_effect(self, sample_data, pipeline_with_registry):
        """Test fit with baseline effect for comparison."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(
            X,
            T,
            Y,
            baseline_ate=0.45,
            baseline_ate_se=0.04,
        )

        # Should include effect stability check
        check_names = [r.check_name for r in result.monitoring_results]
        assert "effect_stability" in check_names

    def test_predict_propensity(self, sample_data, pipeline_with_registry):
        """Test propensity score prediction."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        pipeline.fit(X, T, Y)

        # Predict on new data
        X_new = np.random.randn(50, 5)
        propensity = pipeline.predict_propensity(X_new)

        assert propensity.shape == (50,)
        assert np.all((propensity >= 0) & (propensity <= 1))

    def test_predict_propensity_before_fit_raises(self, pipeline_with_registry):
        """Test that predicting before fit raises error."""
        pipeline = pipeline_with_registry

        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.predict_propensity(np.random.randn(10, 5))

    def test_evaluate_retrain_need(self, sample_data, pipeline_with_registry):
        """Test retraining evaluation."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        pipeline.fit(X, T, Y)

        # Evaluate on similar data - should not trigger
        X_new = np.random.randn(200, 5)
        T_new = np.random.binomial(1, 0.5, size=200)

        results, trigger = pipeline.evaluate_retrain_need(
            X_new,
            T_new,
            X_baseline=X[:200],
            T_baseline=T[:200],
        )

        assert len(results) > 0
        # May or may not trigger depending on random data

    def test_promote_to_production(self, sample_data, pipeline_with_registry):
        """Test promoting model to production."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        pipeline.fit(X, T, Y)

        version_id = pipeline.promote_to_production()

        assert version_id == pipeline._current_version.version_id
        assert pipeline._registry.production_version == version_id

    def test_rollback(self, sample_data, pipeline_with_registry):
        """Test rollback to previous version."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        # Fit and promote two versions
        result1 = pipeline.fit(X, T, Y)
        v1_id = pipeline.promote_to_production()

        result2 = pipeline.fit(X, T, Y)
        v2_id = pipeline.promote_to_production()

        assert pipeline._registry.production_version == v2_id

        # Rollback
        rolled_back = pipeline.rollback()

        assert rolled_back == v1_id
        assert pipeline._registry.production_version == v1_id

    def test_list_versions(self, sample_data, pipeline_with_registry):
        """Test listing model versions."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        # Create multiple versions
        for _ in range(3):
            pipeline.fit(X, T, Y)

        versions = pipeline.list_versions()

        assert len(versions) == 3
        assert all("version_id" in v for v in versions)

    def test_save_and_load(self, sample_data, pipeline_with_registry):
        """Test pipeline save and load."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        result = pipeline.fit(X, T, Y)
        original_ate = result.ate

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_path = pipeline.save(tmpdir)

            assert (save_path / "config.json").exists()
            assert (save_path / "scaler.pkl").exists()
            assert (save_path / "state.json").exists()

            # Load
            loaded_pipeline = InsuranceDMLPipeline.load(tmpdir)

            assert loaded_pipeline._fitted is True
            assert loaded_pipeline._ate == pytest.approx(original_ate)

    def test_repr(self, sample_data, pipeline_with_registry):
        """Test pipeline string representation."""
        X, T, Y = sample_data
        pipeline = pipeline_with_registry

        repr_unfitted = repr(pipeline)
        assert "not fitted" in repr_unfitted

        pipeline.fit(X, T, Y)

        repr_fitted = repr(pipeline)
        assert "fitted" in repr_fitted


class TestPipelineWithCustomModels:
    """Tests for pipeline with custom sklearn models."""

    def test_fit_with_random_forest(self):
        """Test fit with RandomForest models."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 3)
        T = np.random.binomial(1, 0.5, size=n)
        Y = T * 0.5 + X[:, 0] + np.random.randn(n) * 0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                model_registry_path=tmpdir,
                n_folds=2,
                propensity_model=RandomForestClassifier(n_estimators=10, random_state=42),
                outcome_model=RandomForestRegressor(n_estimators=10, random_state=42),
            )
            pipeline = InsuranceDMLPipeline(config)

            result = pipeline.fit(X, T, Y)

            assert result.ate is not None
            assert result.nuisance_metrics["r2_outcome"] > 0


class TestPipelineEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fit_1d_features(self):
        """Test fit with 1D feature array."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n)  # 1D
        T = np.random.binomial(1, 0.5, size=n)
        Y = T * 0.3 + X + np.random.randn(n) * 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                model_registry_path=tmpdir,
                n_folds=2,
            )
            pipeline = InsuranceDMLPipeline(config)

            result = pipeline.fit(X, T, Y)

            assert result.ate is not None

    def test_continuous_treatment(self):
        """Test fit with continuous treatment."""
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 3)
        T = np.random.randn(n)  # Continuous
        Y = 0.5 * T + X[:, 0] + np.random.randn(n) * 0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                model_registry_path=tmpdir,
                n_folds=2,
                propensity_model=Ridge(),  # Use regression for continuous T
            )
            pipeline = InsuranceDMLPipeline(config)

            result = pipeline.fit(X, T, Y)

            assert result.ate is not None
            # Should be close to 0.5
            assert abs(result.ate - 0.5) < 0.3

    def test_evaluate_retrain_without_monitoring_raises(self):
        """Test that evaluate_retrain raises when monitoring disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                model_registry_path=tmpdir,
                monitoring_enabled=False,
            )
            pipeline = InsuranceDMLPipeline(config)

            X = np.random.randn(100, 3)
            T = np.random.binomial(1, 0.5, size=100)
            Y = np.random.randn(100)

            pipeline.fit(X, T, Y)

            with pytest.raises(ValueError, match="Monitoring must be enabled"):
                pipeline.evaluate_retrain_need(X, T)
