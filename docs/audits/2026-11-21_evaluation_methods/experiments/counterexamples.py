"""Public PLR, binary probability and causal-risk counterexamples with known answers."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

from dml_ts import double_ml

HERE = Path(__file__).resolve().parent


class FixedPrediction(RegressorMixin, BaseEstimator):
    """Deterministic conditional-mean candidate used to isolate score algebra."""

    def __init__(self, slope=1.0, intercept=0.0):
        self.slope = slope
        self.intercept = intercept

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.intercept + self.slope * X[:, 0]


class LogisticOutput(RegressorMixin, BaseEstimator):
    """Replay either output of the same fitted classifier as a numeric regressor."""

    def __init__(self, probability):
        self.probability = probability

    def fit(self, X, y):
        self.model_ = LogisticRegression().fit(X, y)
        return self

    def predict(self, X):
        if self.probability:
            return self.model_.predict_proba(X) @ self.model_.classes_
        return self.model_.predict(X)


def paired_errors():
    x, d, u = np.tile(list(product([-1.0, 1.0], repeat=3)), (20, 1)).T
    y = d + u
    rows = []
    for sign in (1.0, -1.0):
        result = double_ml(
            y,
            d,
            x[:, None],
            outcome_model=FixedPrediction(sign),
            treatment_model=FixedPrediction(),
            n_folds=4,
        )
        rows.append(
            {
                "outcome_prediction_sign": sign,
                "estimate": result.theta,
                "se": result.se,
                "outcome_mse": float(np.mean(result.Y_residual**2)),
                "treatment_mse": float(np.mean(result.T_residual**2)),
                "outcome_r2": result.outcome_r2_cv,
                "treatment_r2": result.treatment_r2_cv,
            }
        )
    return rows


def binary_example():
    x = np.repeat([-1.0, 1.0], 480)[:, None]
    d = np.r_[np.repeat([1, 0], [120, 360]), np.repeat([1, 0], [360, 120])].astype(float)
    u = np.tile([-1.0, 1.0], len(d) // 2)
    y = d + u
    rows = {}
    for label, model in [
        ("hard_labels", LogisticOutput(False)),
        ("probabilities", LogisticOutput(True)),
        ("native_classifier", LogisticRegression()),
    ]:
        result = double_ml(
            y, d, x, n_folds=4, outcome_model=FixedPrediction(0.25, 0.5), treatment_model=model
        )
        rows[label] = {
            "estimate": result.theta,
            "se": result.se,
            "brier": float(np.mean(result.T_residual**2)),
        }
    return rows


def differing_targets():
    p = np.array([0.5, 0.01])
    truth = np.array([0.0, 4.0])
    rows = {}
    for label, candidate in {"A": [0.0, 0.0], "B": [1.0, 4.0]}.items():
        error = np.asarray(candidate) - truth
        rows[label] = {
            "population_ate_error": float(abs(error.mean())),
            "population_cate_rmse": float(np.sqrt(np.mean(error**2))),
            "excess_oracle_r_loss": float(np.mean(p * (1 - p) * error**2)),
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "counterexamples.json")
    args = parser.parse_args()
    results = {
        "error_alignment": paired_errors(),
        "binary": binary_example(),
        "overlap": differing_targets(),
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
