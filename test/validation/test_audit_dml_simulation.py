"""Check the paired causal-method comparison at its data, score and inference boundaries."""

import importlib

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from dml_ts.dml.double_ml import double_ml

sim = importlib.import_module("docs.audits.2026-11-19_time_series_dml.adversarial.simulation")
pytestmark = pytest.mark.tier2


def test_folds_and_dgp_hold_target_and_scored_rows_fixed():
    iid, full = sim.design("iid", 6)
    forward, later = sim.design("serial", 6)
    np.testing.assert_array_equal(full, np.arange(sim.N))
    np.testing.assert_array_equal(later, np.arange(300, sim.N))
    for train, test in forward:
        assert train.max() + 6 == test.min()
    for train, test in iid + forward:
        assert not np.intersect1d(train, test).size
    for regime in sim.REGIMES:
        x, d0, y0 = sim.generate(6, "additive", regime)
        x1, d1, y1 = sim.generate(6, "interaction", regime)
        added = 2 * ((x[:, 0] > 0) & (x[:, 1] > 0))
        np.testing.assert_array_equal(x, x1)
        np.testing.assert_allclose(d1 - d0, added, atol=1e-14)
        np.testing.assert_allclose(y1 - y0, (1 + sim.TRUTH) * added, atol=1e-14)


def test_scalar_dml_matches_public_entrypoint_and_fwl_matches_joint_ols():
    x, d, y = sim.generate(13, "interaction", "iid")
    splits, used = sim.design("iid", 13)
    with threadpool_limits(limits=1):
        actual = sim.estimate("dml_trees", x, d, y, "iid", splits, used)
        published = double_ml(y, d, x, model=sim.tree_model(), n_folds=3, random_state=13)
    assert actual == pytest.approx((published.theta, published.se))
    for method, features in [("ols", x), ("spline", sim.spline_features(x))]:
        theta, _ = sim.estimate(method, x, d, y, "iid", splits, used)
        joint = np.column_stack([d, np.ones(len(d)), features])
        assert theta == pytest.approx(np.linalg.lstsq(joint, y, rcond=None)[0][0])


def test_monte_carlo_uncertainty_pairs_only_shared_successes():
    rows = [
        {
            "rep": i,
            "scenario": "linear",
            "regime": "iid",
            "method": method,
            "n_scored": 40,
            "theta": None if method == "ols" and i == 2 else theta,
            "se": 0.15,
            "seconds": 0.01,
        }
        for method in sim.METHODS
        for i, theta in enumerate([0.8, 1.1, 1.4])
    ]
    frame = pd.DataFrame(rows)
    frame.loc[frame.method == "dml_trees", "theta"] += 0.1
    summary, paired = sim.summaries(frame)
    assert summary.loc[summary.method == "ols", "failed"].item() == 1
    comparison = paired.loc[paired.baseline == "ols"].iloc[0]
    assert comparison.paired_n == 2
    gain = np.array([(-0.1) ** 2 - (-0.2) ** 2, 0.2**2 - 0.1**2])
    assert comparison.mse_difference == pytest.approx(gain.mean(), abs=1e-14)
    assert comparison.mse_mcse == pytest.approx(gain.std(ddof=1) / np.sqrt(2))


def test_crossfit_ablation_prediction_membership_and_original_reference():
    from pathlib import Path

    study = importlib.import_module(
        "docs.audits.2026-11-19_time_series_dml.adversarial.simulation_crossfit"
    )
    x, d, y = sim.generate(19, "interaction", "iid")
    splits, used = sim.design("iid", 19)
    assignments = list(study.fold_assignments(splits))
    held_out = np.concatenate([test for _, test, _ in assignments])
    recycled = np.concatenate([included for _, _, included in assignments])
    np.testing.assert_array_equal(np.sort(held_out), used)
    np.testing.assert_array_equal(np.sort(recycled), used)
    for train, test, included in assignments:
        assert len(train) == 800 and len(test) == len(included) == 400
        assert not np.intersect1d(test, train).size
        assert np.isin(included, train).all()
    with threadpool_limits(limits=1):
        results = pd.DataFrame(study.evaluate((19, "interaction")))
        result = results.set_index("method")
        frozen_theta, frozen_se = sim.estimate("dml_trees", x, d, y, "iid", splits, used)
        assert result.loc["dml_trees", "theta"] == pytest.approx(frozen_theta)
        assert result.loc["dml_trees", "se"] == pytest.approx(frozen_se)
        yh, dh = study.predictions(study.fit_trees(x, d, y), x).T
        assert result.loc["full_in_sample", "theta"] == pytest.approx(
            np.dot(y - yh, d - dh) / np.dot(d - dh, d - dh)
        )
        all_shapes = pd.DataFrame(
            [row for shape in sim.SCENARIOS for row in study.evaluate((19, shape))]
        )
    study.validate_reference(all_shapes, Path(study.__file__).with_name("simulation-draws.csv.gz"))
    with pytest.raises(ValueError, match="schedule"):
        study.validate_reference(results, Path(study.__file__).with_name("simulation-draws.csv.gz"))
