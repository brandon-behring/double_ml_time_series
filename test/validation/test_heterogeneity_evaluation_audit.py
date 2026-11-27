"""Check the randomized evaluation score printed in both editions of Chapter 9."""

import importlib
import json

import numpy as np
import pytest

audit = importlib.import_module("docs.audits.2026-11-21_evaluation_methods.heterogeneity.verify")
pytestmark = pytest.mark.tier1


def test_randomized_null_uses_treatments_and_outcomes_independent_of_frozen_prediction():
    effect_score, project = audit.projection_functions()
    rng = np.random.default_rng(7)
    prediction, y = rng.normal(size=(2, 300))
    t = rng.binomial(1, 0.5, len(y))
    legacy = audit.legacy_fit(prediction)
    actual = project(prediction, effect_score(y, t, 0.5))
    assert legacy.params[1] == pytest.approx(1.0)
    assert legacy.pvalues[1] < 1e-10
    assert actual.params[1] == pytest.approx(0.00694897754229)
    assert actual.bse[1] == pytest.approx(0.121573576605)
    reverse = project(prediction, effect_score(y, 1 - t, 0.5))
    assert reverse.params[1] == pytest.approx(-actual.params[1])


def test_known_unequal_assignment_recovers_subgroup_blp_and_frozen_policy_value():
    effect_score, project = audit.projection_functions()
    score = np.repeat([-1.0, 2.0], 8)
    p = np.repeat([0.25, 0.75], 8)
    t = np.r_[[1] * 2 + [0] * 6, [1] * 6 + [0] * 2]
    y = 4 + t * score
    response = effect_score(y, t, p)
    fit = project(score, response)
    assert fit.params == pytest.approx([0.5, 1.0])
    cost = 1.0
    true_policy = score > cost
    shifted_policy = (score + 10) > cost  # Same ranking, different decisions.
    assert np.mean(true_policy * (response - cost)) == pytest.approx(0.5)
    assert np.mean(shifted_policy * (response - cost)) == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="varying prediction"):
        project(np.ones(len(y)), response)
    with pytest.raises(ValueError, match="0 < p < 1"):
        effect_score(y, t, np.where(t == 1, 1, p))


def test_calibration_receipt_and_heldout_insurance_control():
    record = json.loads((audit.HERE / "blp-calibration.json").read_text())
    assert record["legacy_homogeneous"]["count"] == record["repetitions"] == 2000
    measured = record["corrected"]
    for scenario in ("homogeneous", "unrelated_modifier"):
        assert abs(measured[scenario]["fraction"] - 0.05) < 4 * measured[scenario]["mcse"]
    assert measured["aligned_modifier"]["fraction"] > 0.8
    output = json.loads((audit.HERE / "chapter-replay.json").read_text())
    assert len(audit.chapter_blocks()) == output["executed_blocks"]
    assert output["insurance"]["moderate"]["true_sd"] == 0.0
    assert output["insurance"]["full"]["true_sd"] > 0.0
