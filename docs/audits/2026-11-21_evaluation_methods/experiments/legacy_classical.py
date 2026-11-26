"""Compare the forest ATE stress grid with the correctly specified additive linear model."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, t
from statsmodels.stats.multitest import multipletests

from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator

HERE = Path(__file__).resolve().parent
TRUTH = 2.0


def classical_interval(data, check):
    """FWL estimate and exact normal-error Student interval conditional on the design."""
    controls = np.column_stack((np.ones(len(data.Y)), data.X))
    treatment = data.T - controls @ np.linalg.lstsq(controls, data.T, rcond=None)[0]
    outcome = data.Y - controls @ np.linalg.lstsq(controls, data.Y, rcond=None)[0]
    denominator = treatment @ treatment
    slope = float(treatment @ outcome / denominator)
    df = len(outcome) - controls.shape[1] - 1
    residuals = outcome - slope * treatment
    se = np.sqrt((residuals @ residuals) / (df * denominator))
    half_width = float(t.ppf(0.975, df) * se)
    interval = (slope, slope - half_width, slope + half_width)
    if check:
        model = sm.OLS(data.Y, np.column_stack((controls, data.T))).fit()
        np.testing.assert_allclose(interval, [model.params[-1], *model.conf_int()[-1]], atol=1e-10)
    return interval


def main(output, jobs):
    grid = pd.read_csv(HERE / "legacy-coverage-stress.csv")
    comparisons = []
    for cell in grid.itertuples():
        config = {
            "n": cell.n,
            "p": cell.p,
            "true_effect": TRUTH,
            "confounding_strength": cell.confounding,
            "random_state": 42,
        }
        generator = DGPGenerator(**config)
        fits = np.array(
            [classical_interval(generator.generate(), i == 0) for i in range(cell.n_simulations)]
        )
        point, lower, upper = fits.T
        hits = (lower <= TRUTH) & (TRUTH <= upper)
        comparisons.append(
            {
                "scenario_num": cell.scenario_num,
                "n": cell.n,
                "p": cell.p,
                "confounding": cell.confounding,
                "draws": len(hits),
                "forest_hits": cell.n_covered,
                "ols_hits": int(hits.sum()),
                "ols_coverage_p": binomtest(int(hits.sum()), len(hits), 0.95).pvalue,
                "ols_bias": np.mean(point - TRUTH),
                "ols_rmse": np.sqrt(np.mean((point - TRUTH) ** 2)),
                "ols_mean_interval_width": np.mean(upper - lower),
            }
        )
        if cell.scenario_num != 14:
            continue
        forest = BiasValidation(
            n_simulations=cell.n_simulations, random_state=42, n_jobs=jobs
        ).validate(DGPGenerator(**config))
        forest_lower, forest_upper = forest.ci_estimates.T
        forest_hits = (forest_lower <= TRUTH) & (TRUTH <= forest_upper)
        np.testing.assert_equal(forest_hits.sum(), cell.n_covered)
        np.testing.assert_allclose([forest.bias, forest.mse], [cell.bias, cell.mse], atol=1e-12)
        pd.DataFrame(
            {
                "draw": np.arange(len(hits)),
                "forest_ate": forest.point_estimates,
                "forest_lower": forest_lower,
                "forest_upper": forest_upper,
                "forest_hit": forest_hits.astype(int),
                "ols_ate": point,
                "ols_lower": lower,
                "ols_upper": upper,
                "ols_hit": hits.astype(int),
            }
        ).to_csv(output / "legacy-worst-cell-draws.csv", index=False)
    summary = pd.DataFrame(comparisons)
    summary["ols_coverage_holm_p"] = multipletests(summary["ols_coverage_p"], method="holm")[1]
    summary.to_csv(output / "legacy-classical.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("DML_VALIDATION_DRAW_JOBS", 1))
    )
    args = parser.parse_args()
    main(args.output, args.workers)
