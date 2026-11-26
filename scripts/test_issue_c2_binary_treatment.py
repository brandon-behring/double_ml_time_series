"""Compare two binary-treatment nuisance specifications on paired iid simulations."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dml_ts.validation.dgp_generator import DGPGenerator  # noqa: E402

SCENARIOS = [
    (500, 5, 2.0, 0.5),
    (500, 10, 2.0, 1.0),
    (200, 10, 2.0, 2.0),
    (500, 5, 0.5, 1.0),
    (1000, 5, 2.0, 0.5),
]


def fit_draw(data, random_state, discrete_treatment, folds):
    from econml.dml import LinearDML

    options = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "random_state": random_state,
        "n_jobs": 1,
    }
    model_t = (
        RandomForestClassifier(max_features="sqrt", **options)
        if discrete_treatment
        else RandomForestRegressor(max_features=1.0, **options)
    )
    estimator = LinearDML(
        model_y=RandomForestRegressor(max_features=1.0, **options),
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        cv=folds,
        random_state=random_state,
    )
    estimator.fit(Y=data.Y, T=data.T, X=data.X)
    estimate = float(estimator.ate(X=data.X))
    lower, upper = estimator.ate_interval(X=data.X, alpha=0.05)
    return estimate, float(lower), float(upper)


def paired_draw(data, seed):
    # Hold both the data and nuisance train/test assignments fixed across methods.
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(data.X, data.T))
    return [fit_draw(data, seed, discrete, folds) for discrete in [False, True]]


def summarize(estimates, lower, upper, truth):
    error = estimates - truth
    covered = (lower <= truth) & (truth <= upper)
    interval = binomtest(int(covered.sum()), len(covered)).proportion_ci(0.95, method="exact")
    return {
        "bias": error.mean(),
        "abs_bias": abs(error.mean()),
        "bias_mcse": error.std(ddof=1) / np.sqrt(len(error)),
        "mse": np.mean(error**2),
        "rmse": np.sqrt(np.mean(error**2)),
        "coverage": covered.mean(),
        "covered": int(covered.sum()),
        "coverage_mc_lower": interval.low,
        "coverage_mc_upper": interval.high,
    }


def paired_differences(draws, truth):
    estimate, lower, upper = draws.transpose(2, 0, 1)
    hit = ((lower <= truth) & (truth <= upper)).astype(float)
    contrasts = {
        "signed_error": estimate[:, 1] - estimate[:, 0],
        "squared_error": (estimate[:, 1] - truth) ** 2 - (estimate[:, 0] - truth) ** 2,
        "coverage": hit[:, 1] - hit[:, 0],
    }
    result = {}
    for metric, delta in contrasts.items():
        result[metric + "_difference"] = delta.mean()
        result[metric + "_difference_mcse"] = delta.std(ddof=1) / np.sqrt(len(delta))
    return result


def run_dml_with_config(dgp, n_simulations, discrete_treatment, config_name, random_state=42):
    """Evaluate one configuration on a supplied generator, preserving direct callers."""
    if n_simulations < 2:
        raise ValueError("At least two draws are required to estimate Monte Carlo uncertainty")
    values = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for _ in range(n_simulations):
        sample = dgp.generate()
        folds = list(splitter.split(sample.X, sample.T))
        values.append(fit_draw(sample, random_state, discrete_treatment, folds))
    return {
        **summarize(*np.asarray(values).T, truth=dgp.true_effect),
        "configuration": config_name,
        "discrete_treatment": discrete_treatment,
        "n_simulations": n_simulations,
        "dgp_n": dgp.n,
        "dgp_p": dgp.p,
        "dgp_true_effect": dgp.true_effect,
        "dgp_confounding": dgp.confounding_strength,
    }


def main(output_dir=ROOT / "results/impact_analysis", n_simulations=100, n_jobs=1):
    if n_simulations < 2 or n_jobs < 1:
        raise ValueError("Use at least two simulations and one fit worker")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 42
    summaries, comparisons, records = [], [], []
    for n, p, truth, strength in SCENARIOS:
        dgp = DGPGenerator(
            n=n, p=p, true_effect=truth, confounding_strength=strength, random_state=seed
        )
        # Advance one generator in the caller. Worker ordering cannot change the data.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            fits = Parallel(n_jobs=min(n_jobs, n_simulations))(
                delayed(paired_draw)(dgp.generate(), seed) for _ in range(n_simulations)
            )
        draws = np.asarray(fits)
        design = {"dgp_n": n, "dgp_p": p, "dgp_true_effect": truth, "dgp_confounding": strength}
        for j, label in enumerate(["regression nuisance", "classifier nuisance"]):
            estimates, lower, upper = draws[:, j, :].T
            summaries.append(
                {
                    **design,
                    "configuration": label,
                    "discrete_treatment": bool(j),
                    "n_simulations": n_simulations,
                    **summarize(estimates, lower, upper, truth),
                }
            )
            records.extend(
                {
                    **design,
                    "draw": i,
                    "configuration": label,
                    "estimate": e,
                    "lower": lo,
                    "upper": hi,
                }
                for i, (e, lo, hi) in enumerate(draws[:, j, :])
            )
        comparisons.append(
            {**design, "n_simulations": n_simulations, **paired_differences(draws, truth)}
        )
    summary = pd.DataFrame(summaries)
    comparison = pd.DataFrame(comparisons)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary.to_csv(output_dir / f"issue_c2_impact_{stamp}.csv", index=False)
    comparison.to_csv(output_dir / f"issue_c2_paired_{stamp}.csv", index=False)
    pd.DataFrame(records).to_csv(
        output_dir / f"issue_c2_draws_{stamp}.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(summary.to_string(index=False))
    print("\nClassifier minus regression on paired draws and identical stratified folds")
    print(comparison.to_string(index=False))
    return summary, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/impact_analysis")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    main(args.output_dir, args.draws, args.jobs)
