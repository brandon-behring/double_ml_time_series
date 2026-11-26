"""Validation of nuisance conditional means, effect-risk targets, and causal timing."""

from importlib import import_module

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from dml_ts import double_ml
from dml_ts.dml.dynamic_g_estimation import _crossfit_residualize
from dml_ts.dml.robinson import robinson_estimator

prefix = "docs.audits.2026-11-21_evaluation_methods.experiments"
examples = import_module(prefix + ".counterexamples")
factorial = import_module(prefix + ".learner_pairs")
pytestmark = pytest.mark.tier2


def test_equal_marginal_losses_can_lead_to_different_effects():
    first, second = examples.paired_errors()
    for key in ("outcome_mse", "treatment_mse", "outcome_r2", "treatment_r2", "se"):
        assert first[key] == pytest.approx(second[key])
    assert (first["estimate"], second["estimate"]) == (1, 0)
    targets = examples.differing_targets()
    assert targets["A"]["excess_oracle_r_loss"] < targets["B"]["excess_oracle_r_loss"]
    assert targets["A"]["population_ate_error"] > targets["B"]["population_ate_error"]


def test_public_classifier_matches_explicit_probabilities_and_differs_from_hard_labels():
    example = examples.binary_example()
    for key in ("estimate", "se", "brier"):
        assert example["native_classifier"][key] == pytest.approx(example["probabilities"][key])
    assert example["hard_labels"]["estimate"] == pytest.approx(0.75)
    assert example["native_classifier"]["estimate"] == pytest.approx(1, abs=0.01)
    assert example["native_classifier"]["brier"] < example["hard_labels"]["brier"]


def test_conditional_means_with_nonstandard_class_labels_on_both_sides():
    n = 120
    x = np.arange(n)[:, None]
    d = np.tile([2, 2, 2, 5, 5, 5], n // 6)
    y = np.tile([-2, -2, 4, -2, 4, 4], n // 6)
    result = double_ml(y, d, x, outcome_model=DummyClassifier(), treatment_model=DummyClassifier())
    for train, test in KFold(5, shuffle=True, random_state=42).split(x):
        np.testing.assert_allclose((d - result.T_residual)[test], np.mean(d[train]))
        np.testing.assert_allclose((y - result.Y_residual)[test], np.mean(y[train]))
    robinson = robinson_estimator(y, d, x, model=DummyClassifier())
    np.testing.assert_allclose(d - robinson.T_residual, np.mean(d))
    np.testing.assert_allclose(y - robinson.Y_residual, np.mean(y))


@pytest.mark.parametrize("model", [LogisticRegression(), DummyClassifier()])
def test_dynamic_class_probabilities_use_numeric_levels_for_each_period(model):
    rng = np.random.default_rng(20)
    x = rng.normal(size=(120, 2))
    targets = np.column_stack([rng.choice([-3, 4], 120), rng.choice([2, 8], 120)])
    splits = list(KFold(3).split(x))
    actual = _crossfit_residualize(x, targets, model, splits)
    for train, test in splits:
        for j in range(2):
            fit = model.fit(x[train], targets[train, j])
            expected = fit.predict_proba(x[test]) @ fit.classes_
            np.testing.assert_allclose(targets[test, j] - actual[test, j], expected)


def test_hard_only_classifier_requires_probabilities():
    x = np.arange(32)[:, None]
    with pytest.raises(ValueError, match="predict_proba"):
        double_ml(x[:, 0], np.tile([0, 1], 16), x, model="ridge", treatment_model=LinearSVC())


def test_next_block_outcomes_do_not_influence_selection_or_frozen_forecast():
    n = 1200
    rng = np.random.default_rng(4)
    blocks = factorial.BASE.folds(n)["past_buffer"]
    scored = np.concatenate([te for _, te in blocks])
    ys = {name: rng.normal(size=n) for name in factorial.FAMILIES}
    ds = {name: rng.normal(size=n) for name in factorial.FAMILIES}
    truth = np.ones(n)
    orig = factorial.run_selectors(5, "serial", scored, blocks, ys, ds, truth)
    for name in factorial.FAMILIES:
        ys[name][blocks[2][1]] += 10 * ds[name][blocks[2][1]]
    after = factorial.run_selectors(5, "serial", scored, blocks, ys, ds, truth)
    for a, b in zip(orig[:6], after[:6], strict=True):
        assert a[:10] == b[:10]
        assert a[11:] == b[11:]
        future = blocks[2][1]
        change = 10 * ds[a[5]][future] @ ds[a[6]][future] / (ds[a[6]][future] @ ds[a[6]][future])
        assert b[10] - a[10] == pytest.approx(change)
