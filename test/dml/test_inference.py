"""Tests for the causal inference helpers (dml_ts.dml.inference).

Carried over from the retired test_hac.py — the HAC machinery itself now
lives in temporalcv (tested there + pinned by test/golden); only the
causal-layer hac_inference helper remains here.
"""

import pytest

from dml_ts.dml.inference import hac_inference

pytestmark = pytest.mark.tier1


class TestHACInference:
    """Test inference helper function."""

    def test_basic_inference(self) -> None:
        """Test basic inference computation."""
        result = hac_inference(theta=2.0, se_hac=0.5, alpha=0.05)

        assert result["theta"] == 2.0
        assert result["se"] == 0.5
        assert result["ci_lower"] < 2.0 < result["ci_upper"]
        assert result["t_stat"] == 4.0
        assert result["p_value"] < 0.001  # Highly significant

    def test_ci_coverage(self) -> None:
        """Test CI has correct width."""
        result = hac_inference(theta=1.0, se_hac=1.0, alpha=0.05)

        # 95% CI should be approximately ±1.96
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 3.8 < ci_width < 4.0  # ≈ 2 * 1.96

    def test_different_alpha(self) -> None:
        """Test different significance levels."""
        result_95 = hac_inference(theta=1.0, se_hac=0.5, alpha=0.05)
        result_99 = hac_inference(theta=1.0, se_hac=0.5, alpha=0.01)

        # 99% CI should be wider
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]
        assert width_99 > width_95

    def test_nonsignificant(self) -> None:
        """Test non-significant result."""
        result = hac_inference(theta=0.1, se_hac=1.0, alpha=0.05)

        # Should not be significant at 5%
        assert result["p_value"] > 0.05
        # CI should contain 0
        assert result["ci_lower"] < 0 < result["ci_upper"]

    def test_zero_se_raises(self) -> None:
        """Degenerate SE must fail loud, not report p=0 (infinitely significant)."""
        with pytest.raises(ValueError, match="positive finite"):
            hac_inference(theta=2.0, se_hac=0.0)

    def test_negative_and_nonfinite_se_raise(self) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            hac_inference(theta=2.0, se_hac=-0.5)
        with pytest.raises(ValueError, match="positive finite"):
            hac_inference(theta=2.0, se_hac=float("nan"))
