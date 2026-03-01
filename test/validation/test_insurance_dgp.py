"""Tests for Insurance/Annuity DGP Generator.

Tests cover:
1. Output shapes and types for all realism levels
2. True parameter correctness per complexity level
3. Reproducibility via seed control
4. Data quality (no NaN/Inf, valid ranges, panel structure)
5. Monte Carlo recovery validation (tier3)
6. Input validation

See: src/validation/insurance_dgp.py (667 lines)
"""

from dataclasses import fields
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.validation.insurance_dgp import (
    InsuranceDGPParams,
    InsuranceDGPResult,
    create_insurance_dgp,
    validate_dgp_recovery,
)

# Expected macro keys across all realism levels
EXPECTED_MACRO_KEYS = {"fed_funds", "treasury_10y", "inflation", "gdp_growth", "unemployment"}


# ============================================================================
# Class 1: Output Shapes and Types
# ============================================================================


@pytest.mark.tier1
class TestCreateInsuranceDGPShapes:
    """Validate output shapes and types for all realism levels."""

    @pytest.mark.parametrize("realism", ["simple", "moderate", "full"])
    def test_default_output_shapes(self, realism: str) -> None:
        """Default n_periods=120, n_products=10 -> n_total=1200."""
        result = create_insurance_dgp(realism=realism, seed=42)
        n_total = 120 * 10

        assert result.Y.shape == (n_total,)
        assert result.T.shape == (n_total,)
        assert result.X.shape == (n_total, 9)  # 4 firm + 5 macro
        assert result.X_macro.shape == (n_total, 5)
        assert result.time_index.shape == (n_total,)
        assert result.product_index.shape == (n_total,)

    def test_custom_dimensions(self) -> None:
        """Non-default panel dimensions produce correct shapes."""
        result = create_insurance_dgp(realism="simple", n_periods=60, n_products=5, seed=42)
        n_total = 60 * 5

        assert result.Y.shape == (n_total,)
        assert result.T.shape == (n_total,)
        assert result.X.shape == (n_total, 9)
        assert result.X_macro.shape == (n_total, 5)

    def test_small_panel(self) -> None:
        """Minimal panel (10 periods x 2 products) still works."""
        result = create_insurance_dgp(realism="moderate", n_periods=10, n_products=2, seed=42)
        n_total = 10 * 2

        assert result.Y.shape == (n_total,)
        assert result.X.shape == (n_total, 9)

    def test_output_dtypes(self) -> None:
        """All arrays have expected dtypes."""
        result = create_insurance_dgp(realism="moderate", seed=42)

        assert result.Y.dtype == np.float64
        assert result.T.dtype == np.float64
        assert result.X.dtype == np.float64
        assert result.X_macro.dtype == np.float64
        assert result.time_index.dtype == np.int64
        assert result.product_index.dtype == np.int64

    def test_description_string_populated(self) -> None:
        """Description is a non-empty string."""
        result = create_insurance_dgp(realism="full", seed=42)

        assert isinstance(result.description, str)
        assert len(result.description) > 0
        assert "full" in result.description

    def test_true_params_is_dataclass(self) -> None:
        """true_params is an InsuranceDGPParams instance."""
        result = create_insurance_dgp(realism="simple", seed=42)

        assert isinstance(result.true_params, InsuranceDGPParams)
        # Verify it's a proper dataclass with expected fields
        field_names = {f.name for f in fields(InsuranceDGPParams)}
        assert "tau" in field_names
        assert "beta_macro" in field_names

    def test_result_is_dataclass(self) -> None:
        """Result container is an InsuranceDGPResult instance."""
        result = create_insurance_dgp(realism="simple", seed=42)
        assert isinstance(result, InsuranceDGPResult)


# ============================================================================
# Class 2: Parameter Correctness
# ============================================================================


@pytest.mark.tier1
class TestInsuranceDGPParams:
    """Validate true_params correctness per complexity level."""

    def test_simple_params(self) -> None:
        """Simple realism: no AR, no seasonality, no regime shift."""
        result = create_insurance_dgp(realism="simple", seed=42)
        p = result.true_params

        assert p.tau == -0.8
        assert p.ar_coef == 0.0
        assert p.seasonal_amplitude == 0.0
        assert p.regime_shift == 0.0
        assert p.tau_by_product is None
        assert p.beta_product is None

    def test_moderate_params(self) -> None:
        """Moderate realism: AR errors, product FE, no seasonality."""
        result = create_insurance_dgp(realism="moderate", seed=42)
        p = result.true_params

        assert p.tau == -0.8
        assert p.ar_coef == 0.3
        assert p.seasonal_amplitude == 0.0
        assert p.regime_shift == 0.0
        assert p.tau_by_product is None
        assert p.beta_product is not None
        assert p.beta_product.shape == (10,)  # default n_products

    def test_full_params(self) -> None:
        """Full realism: AR, seasonality, regime shift, heterogeneous tau."""
        result = create_insurance_dgp(realism="full", n_products=10, seed=42)
        p = result.true_params

        assert p.ar_coef == 0.4
        assert p.seasonal_amplitude == 3.0
        assert p.regime_shift == -0.2
        assert p.tau_by_product is not None
        assert p.tau_by_product.shape == (10,)
        assert p.beta_product is not None
        assert p.beta_product.shape == (10,)

    def test_custom_true_tau(self) -> None:
        """Custom true_tau propagates to params."""
        result = create_insurance_dgp(realism="simple", true_tau=2.5, seed=42)
        assert result.true_params.tau == 2.5

    def test_beta_macro_keys(self) -> None:
        """All realism levels have 5 expected macro coefficient keys."""
        for realism in ("simple", "moderate", "full"):
            result = create_insurance_dgp(realism=realism, seed=42)
            assert set(result.true_params.beta_macro.keys()) == EXPECTED_MACRO_KEYS

    def test_regime_shift_period_full(self) -> None:
        """Full realism: regime_shift_period ~ 60 for default n_periods=120."""
        result = create_insurance_dgp(realism="full", n_periods=120, seed=42)
        assert result.true_params.regime_shift_period == 60

    def test_regime_shift_period_clamped(self) -> None:
        """Regime shift period is clamped to n_periods - 1 for small panels."""
        result = create_insurance_dgp(realism="full", n_periods=30, seed=42)
        assert result.true_params.regime_shift_period == 29  # min(60, 30-1)


# ============================================================================
# Class 3: Reproducibility
# ============================================================================


@pytest.mark.tier1
class TestInsuranceDGPReproducibility:
    """Verify seed-based reproducibility."""

    def test_same_seed_same_data(self) -> None:
        """Identical seeds produce identical data."""
        r1 = create_insurance_dgp(realism="moderate", seed=42)
        r2 = create_insurance_dgp(realism="moderate", seed=42)

        assert_allclose(r1.Y, r2.Y)
        assert_allclose(r1.T, r2.T)
        assert_allclose(r1.X, r2.X)
        assert_allclose(r1.X_macro, r2.X_macro)

    def test_different_seed_different_data(self) -> None:
        """Different seeds produce different data."""
        r1 = create_insurance_dgp(realism="moderate", seed=42)
        r2 = create_insurance_dgp(realism="moderate", seed=99)

        assert not np.allclose(r1.Y, r2.Y)
        assert not np.allclose(r1.T, r2.T)

    def test_none_seed_nondeterministic(self) -> None:
        """seed=None produces (likely) different data across calls."""
        r1 = create_insurance_dgp(realism="simple", seed=None)
        r2 = create_insurance_dgp(realism="simple", seed=None)

        # Astronomically unlikely to be identical
        assert not np.allclose(r1.Y, r2.Y)


# ============================================================================
# Class 4: Data Quality Properties
# ============================================================================


@pytest.mark.tier1
class TestInsuranceDGPDataProperties:
    """Sanity checks on generated data values."""

    @pytest.mark.parametrize("realism", ["simple", "moderate", "full"])
    def test_no_nan_in_output(self, realism: str) -> None:
        """No NaN values in any output array."""
        result = create_insurance_dgp(realism=realism, seed=42)

        assert not np.any(np.isnan(result.Y)), "NaN in Y"
        assert not np.any(np.isnan(result.T)), "NaN in T"
        assert not np.any(np.isnan(result.X)), "NaN in X"
        assert not np.any(np.isnan(result.X_macro)), "NaN in X_macro"

    @pytest.mark.parametrize("realism", ["simple", "moderate", "full"])
    def test_no_inf_in_output(self, realism: str) -> None:
        """No Inf values in any output array."""
        result = create_insurance_dgp(realism=realism, seed=42)

        assert not np.any(np.isinf(result.Y)), "Inf in Y"
        assert not np.any(np.isinf(result.T)), "Inf in T"
        assert not np.any(np.isinf(result.X)), "Inf in X"
        assert not np.any(np.isinf(result.X_macro)), "Inf in X_macro"

    def test_panel_index_structure(self) -> None:
        """time_index and product_index cover full panel grid."""
        n_periods, n_products = 24, 3
        result = create_insurance_dgp(
            realism="simple", n_periods=n_periods, n_products=n_products, seed=42
        )

        # Every time period 0..n_periods-1 appears
        assert set(result.time_index) == set(range(n_periods))
        # Every product 0..n_products-1 appears
        assert set(result.product_index) == set(range(n_products))
        # Total observations = n_periods * n_products
        assert len(result.time_index) == n_periods * n_products

    def test_macro_variables_reasonable_range(self) -> None:
        """Macro variables stay in economically plausible ranges."""
        result = create_insurance_dgp(realism="moderate", n_periods=120, seed=42)
        # X_macro columns: fed_funds, treasury_10y, inflation, gdp_growth, unemployment
        # Extract first n_periods rows (one product's worth)
        macro = result.X_macro[:120, :]

        fed_funds = macro[:, 0]
        treasury = macro[:, 1]
        inflation = macro[:, 2]
        gdp = macro[:, 3]
        unemployment = macro[:, 4]

        # Generous bounds — DGP uses mean-reverting AR processes
        assert fed_funds.min() > -5, f"fed_funds too low: {fed_funds.min()}"
        assert fed_funds.max() < 15, f"fed_funds too high: {fed_funds.max()}"
        assert unemployment.min() >= 3.0, f"unemployment below floor: {unemployment.min()}"
        assert unemployment.max() <= 10.0, f"unemployment above cap: {unemployment.max()}"
        assert treasury.min() > -3, f"treasury too low: {treasury.min()}"

    def test_treatment_has_variation(self) -> None:
        """Treatment variable is not degenerate (has positive std)."""
        for realism in ("simple", "moderate", "full"):
            result = create_insurance_dgp(realism=realism, seed=42)
            assert np.std(result.T) > 0, f"Degenerate T for realism={realism}"


# ============================================================================
# Class 5: Input Validation
# ============================================================================


@pytest.mark.tier1
class TestInsuranceDGPValidation:
    """Input validation guards."""

    def test_invalid_realism_raises(self) -> None:
        """Invalid realism string raises ValueError."""
        with pytest.raises(ValueError, match="realism must be"):
            create_insurance_dgp(realism="extreme")  # type: ignore[arg-type]

    def test_zero_n_periods_raises(self) -> None:
        """n_periods=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_periods must be >= 1"):
            create_insurance_dgp(n_periods=0)

    def test_negative_n_products_raises(self) -> None:
        """n_products=-1 raises ValueError."""
        with pytest.raises(ValueError, match="n_products must be >= 1"):
            create_insurance_dgp(n_products=-1)


# ============================================================================
# Class 6: Monte Carlo Recovery (tier3)
# ============================================================================


@pytest.mark.tier3
class TestValidateDGPRecovery:
    """Integration: DML recovers true treatment effect via Monte Carlo."""

    def test_simple_recovery(self) -> None:
        """Simple DGP: PanelDML recovers true tau with low bias."""
        results = validate_dgp_recovery(
            realism="simple",
            n_sims=20,
            seed=42,
            n_periods=60,
            n_products=5,
        )

        assert "bias" in results
        assert "rmse" in results
        assert "coverage" in results
        assert "n_successful" in results
        assert abs(results["bias"]) < 1.0, f"Bias too large: {results['bias']}"

    def test_moderate_recovery(self) -> None:
        """Moderate DGP: bias within 1.0 despite AR errors and product FE."""
        results = validate_dgp_recovery(
            realism="moderate",
            n_sims=20,
            seed=42,
            n_periods=60,
            n_products=5,
        )

        assert abs(results["bias"]) < 1.0, f"Bias too large: {results['bias']}"
        assert results["n_successful"] >= 10

    def test_full_recovery(self) -> None:
        """Full DGP: relaxed threshold (regime shifts + GARCH increase variance)."""
        results = validate_dgp_recovery(
            realism="full",
            n_sims=30,
            seed=42,
            n_periods=60,
            n_products=5,
        )

        # Relaxed: regime shifts and GARCH widen the bias distribution
        assert abs(results["bias"]) < 2.0, f"Bias too large: {results['bias']}"
        assert results["n_successful"] >= 15

    def test_recovery_too_many_failures_raises(self) -> None:
        """RuntimeError when >50% of simulations fail."""
        with patch("src.dml.PanelDML") as mock_panel:
            # Force all fits to raise
            mock_panel.return_value.fit.side_effect = RuntimeError("forced failure")

            with pytest.raises(RuntimeError, match="Too many simulation failures"):
                validate_dgp_recovery(
                    realism="simple",
                    n_sims=10,
                    seed=42,
                    n_periods=30,
                    n_products=3,
                )
