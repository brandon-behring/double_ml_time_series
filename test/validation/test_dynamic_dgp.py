"""Tests for the dynamic-treatment DGP with known per-period blips."""

import numpy as np
import pytest

from dml_ts.validation import DynamicTreatmentDGP, DynamicTreatmentDGPResult


@pytest.mark.tier1
class TestDynamicDGPStructure:
    """Shape, layout, and validation of the generated data."""

    def test_panel_shapes_and_groups(self):
        d = DynamicTreatmentDGP(n_periods=4, n_units=50, p=3, random_state=0).generate()
        assert isinstance(d, DynamicTreatmentDGPResult)
        assert d.Y.shape == (200,) and d.X.shape == (200, 3) and d.groups.shape == (200,)
        assert d.n_periods == 4 and d.n_units == 50 and d.theta_t.shape == (4,)
        y_p, t_p, x_p = d.panels()
        assert y_p.shape == (50, 4) and t_p.shape == (50, 4) and x_p.shape == (50, 4, 3)
        # every unit contributes exactly m consecutive periods
        assert (np.bincount(d.groups) == 4).all()
        assert (d.time_index[:4] == [0, 1, 2, 3]).all()

    def test_cumulative_is_sum_of_blips(self):
        d = DynamicTreatmentDGP(
            n_periods=3, theta_t=[1.0, 2.0, 3.0], n_units=10, random_state=0
        ).generate()
        assert d.cumulative_effect == pytest.approx(6.0)
        assert np.allclose(d.theta_t, [1.0, 2.0, 3.0])

    def test_series_shapes(self):
        d = DynamicTreatmentDGP(
            n_periods=2, p=2, mode="series", series_length=300, random_state=0
        ).generate()
        assert d.groups is None and d.Y.shape == (300,) and d.X.shape == (300, 2)
        assert d.mode == "series"

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError, match="n_periods"):
            DynamicTreatmentDGP(n_periods=1)
        with pytest.raises(ValueError, match="theta_t"):
            DynamicTreatmentDGP(n_periods=3, theta_t=[1.0, 2.0])
        with pytest.raises(ValueError, match="mode"):
            DynamicTreatmentDGP(n_periods=3, mode="bogus")

    def test_to_econml_layout(self):
        d = DynamicTreatmentDGP(n_periods=3, n_units=20, random_state=0).generate()
        y, t, x, g = d.to_econml()
        assert len(y) == 60 and len(t) == 60 and x.shape == (60, d.p) and g.max() == 19

    def test_series_result_rejects_panel_helpers(self):
        d = DynamicTreatmentDGP(
            n_periods=2, mode="series", series_length=200, random_state=0
        ).generate()
        with pytest.raises(ValueError):
            d.panels()
        with pytest.raises(ValueError):
            d.to_econml()


@pytest.mark.tier2
class TestDynamicDGPProperties:
    """Statistical properties: the blips are exactly the stated ground truth."""

    def test_exact_blip_algebra_zero_outcome_noise(self):
        # Independent treatment variation + zero outcome noise => final outcome is an
        # exact linear function of the treatments, recoverable to machine precision.
        d = DynamicTreatmentDGP(
            n_periods=3,
            theta_t=[1.0, 2.0, 3.0],
            n_units=400,
            p=2,
            treatment_noise=1.0,
            outcome_noise=0.0,
            state_noise=1.0,
            state_feedback=False,
            random_state=0,
        ).generate()
        y_p, t_p, x_p = d.panels()
        design = np.column_stack([t_p, x_p[:, -1, :]])
        coef, *_ = np.linalg.lstsq(design, y_p[:, -1], rcond=None)
        np.testing.assert_allclose(coef[:3], [1.0, 2.0, 3.0], atol=1e-8)

    def test_state_feedback_inflates_total_blip(self):
        # With treatment-dependent state the total blip exceeds the direct coefficient
        # for tau < m (closed form), while the last period is unchanged.
        d = DynamicTreatmentDGP(
            n_periods=3,
            theta_t=1.0,
            p=2,
            state_feedback=True,
            state_transition=0.5,
            treatment_state_coef=0.8,
            confounding_coef=1.0,
            random_state=0,
        ).generate()
        assert d.theta_t[-1] == pytest.approx(1.0)
        assert d.theta_t[0] > 1.0 and d.theta_t[1] > 1.0

    def test_naive_regression_is_biased_under_feedback(self):
        # Confounding requires the sequential structure: a no-control regression is biased.
        d = DynamicTreatmentDGP(
            n_periods=3,
            theta_t=1.0,
            n_units=4000,
            p=2,
            state_feedback=True,
            treatment_state_coef=0.8,
            confounding_coef=1.0,
            treatment_noise=1.0,
            outcome_noise=0.5,
            random_state=1,
        ).generate()
        y_p, t_p, _ = d.panels()
        design = np.column_stack([t_p, np.ones(d.n_units)])
        naive, *_ = np.linalg.lstsq(design, y_p[:, -1], rcond=None)
        assert np.abs(naive[:3] - d.theta_t).max() > 0.1
