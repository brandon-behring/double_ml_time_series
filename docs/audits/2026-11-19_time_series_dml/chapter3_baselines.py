"""Reproduce Chapter 3 Experiment 3.1 OLS rows on identical linear DGP draws."""

import numpy as np
from scipy.stats import binomtest, t

from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.ols_baseline import NaiveOLS, OLSWithControls


def chapter_dgp():
    return DGPGenerator(
        n=1000,
        p=5,
        true_effect=2.0,
        confounding_strength=1.0,
        treatment_model="linear",
        outcome_model="linear",
        random_state=42,
    )


def coverage_summary(covers):
    interval = binomtest(sum(covers), len(covers)).proportion_ci(method="wilson")
    return f"{np.mean(covers):.0%} (Wilson 95% interval {interval.low:.1%} to {interval.high:.1%})"


def main():
    trials = 100
    reported = {
        model.__name__: model(n_simulations=trials, random_state=42).validate(chapter_dgp())
        for model in (NaiveOLS, OLSWithControls)
    }
    # Replay the same draws with a reference calculation. Only the controlled
    # OLS SE and its t critical value change in the corrected coverage count.
    dgp = chapter_dgp()
    naive_errors, controlled_errors, corrected_covers, se_ratios = [], [], [], []
    for _ in range(trials):
        data = dgp.generate()
        treatment, outcome = data.T, data.Y
        naive = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])
        naive_errors.append(naive - dgp.true_effect)
        predictors = np.column_stack([treatment, data.X])
        design = np.column_stack([np.ones(len(outcome)), predictors])
        coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
        controlled_error = coefficients[1] - dgp.true_effect
        controlled_errors.append(controlled_error)
        residuals = outcome - design @ coefficients
        rss = residuals @ residuals
        degrees = len(outcome) - design.shape[1]
        corrected_se = np.sqrt(rss / degrees * np.linalg.inv(design.T @ design)[1, 1])
        reported_se = np.sqrt(rss / (degrees + 1) * np.linalg.inv(predictors.T @ predictors)[0, 0])
        corrected_covers.append(abs(controlled_error) <= t.ppf(0.975, degrees) * corrected_se)
        se_ratios.append(reported_se / corrected_se)

    for name, errors in [("NaiveOLS", naive_errors), ("OLSWithControls", controlled_errors)]:
        result = reported[name]
        np.testing.assert_allclose(
            [np.mean(errors), np.mean(np.square(errors))], [result.bias, result.mse]
        )
        mcse = np.std(errors, ddof=1) / np.sqrt(trials)
        reference_p = 2 * t.sf(abs(np.mean(errors)) / mcse, trials - 1)
        np.testing.assert_allclose(result.bias_p_value, reference_p, rtol=1e-12)
        count = round(result.coverage * trials)
        print(
            f"{name}: bias {result.bias:.6f}, bias MCSE {mcse:.6f}, RMSE {result.mse**0.5:.6f}, "
            f"coverage {count}/{trials}, bias p {result.bias_p_value:.6f}, {result.status}"
        )
    print("Controlled OLS with intercept in covariance:", coverage_summary(corrected_covers))
    print(f"Median package / corrected slope SE: {np.median(se_ratios):.6f}")


if __name__ == "__main__":
    main()
