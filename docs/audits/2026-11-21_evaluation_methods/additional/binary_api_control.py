"""Compare EconML's binary and continuous treatment interfaces with identical nuisances."""

import argparse
import json
from pathlib import Path

import numpy as np
from econml.dml import LinearDML
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


def run():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 4))
    treatment = rng.binomial(1, 0.5, size=len(x))
    outcome = x[:, 0] + 2 * treatment + rng.normal(size=len(x))
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, treatment))
    output = {}
    for discrete, model in [
        (False, DummyRegressor(strategy="mean")),
        (True, DummyClassifier(strategy="prior")),
    ]:
        fit = LinearDML(
            model_y=LinearRegression(),
            model_t=model,
            discrete_treatment=discrete,
            cv=folds,
            random_state=42,
        )
        fit.fit(outcome, treatment, X=x, cache_values=True)
        low, high = fit.ate_interval(X=x)
        label = "classifier" if discrete else "regressor"
        output[label] = {"ate": float(fit.ate(X=x)), "lower": float(low), "upper": float(high)}
        if not discrete:
            residuals = fit.residuals_[1].ravel()
        else:
            output["largest_treatment_residual_difference"] = float(
                np.max(np.abs(residuals - fit.residuals_[1].ravel()))
            )
    output["default_forest_max_features"] = {
        "classifier": RandomForestClassifier().max_features,
        "regressor": RandomForestRegressor().max_features,
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2, allow_nan=False) + "\n")
