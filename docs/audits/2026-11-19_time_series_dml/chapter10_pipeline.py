"""Reproduce the synthetic retention example in Chapter 10 from the repository root."""

import csv
import sys
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from dml_ts.production import InsuranceDMLPipeline, PipelineConfig


def simulated_retention(seed=42):
    """Return covariates, price, price-design outcome, offer, and offer-design outcome."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (400, 2))
    price = x[:, 0] + rng.uniform(-1, 1, len(x))  # units of ten dollars
    retained = rng.binomial(1, 0.50 + 0.10 * price + 0.10 * x[:, 1])
    offer = rng.binomial(1, 0.50 + 0.20 * x[:, 0])
    renewed = rng.binomial(1, 0.35 + 0.20 * offer + 0.10 * x[:, 1])
    return x, price, retained, offer, renewed


def main():
    x, price, retained, offer, renewed = simulated_retention()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("specification", "target", "estimate", "se", "ci_lower", "ci_upper", "n_used"))
    cases = (
        ("Price in tens of dollars", 0.10, price, retained, LinearRegression()),
        ("Price in dollars", 0.01, 10 * price, retained, LinearRegression()),
        ("Offer", 0.20, offer, renewed, LogisticRegression()),
        ("Offer with outcome offset", 0.20, offer, renewed + 100, LogisticRegression()),
    )
    with TemporaryDirectory() as registry:
        for name, truth, treatment, outcome, learner in cases:
            config = PipelineConfig(
                n_folds=4,
                time_column="period",
                hac_bandwidth=4,
                propensity_model=learner,
                outcome_model=LinearRegression(),
                model_registry_path=registry,
            )
            result = InsuranceDMLPipeline(config).fit(x, treatment, outcome)
            writer.writerow(
                (
                    name,
                    truth,
                    result.ate,
                    result.ate_se,
                    result.ate_ci_lower,
                    result.ate_ci_upper,
                    result.metadata["n_used"],
                )
            )


if __name__ == "__main__":
    main()
