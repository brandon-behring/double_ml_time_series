"""Isolate changes in the 401(k) fit using the repository's original learners."""

import argparse
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LogisticRegressionCV

PAPER_CONTROLS = ["age", "inc", "fsize", "educ", "db", "marr", "twoearn", "pira", "hown"]
PAPER_REFERENCE = {"RF": 9247, "Lasso": 8187}


def learners(name, seed):
    if name == "RF":
        options = {"n_estimators": 500, "min_samples_leaf": 10, "random_state": seed}
        return RandomForestRegressor(**options), RandomForestClassifier(**options)
    return LassoCV(cv=5, random_state=seed), LogisticRegressionCV(
        cv=5, penalty="l1", solver="saga", random_state=seed, max_iter=1000
    )


def fit(data, name, specification, seed):
    # Preserve the source fixture column order when only removing the two wealth fields.
    controls = [c for c in data if c not in {"e401", "p401", "net_tfa"}]
    if specification != "repo_X11":
        controls = [c for c in controls if c in PAPER_CONTROLS]
    controls = data[controls].to_numpy()
    x = None if specification == "paper_W9" else controls
    w = controls if specification == "paper_W9" else None
    outcome_model, treatment_model = learners(name, seed)
    model = LinearDML(
        model_y=outcome_model,
        model_t=treatment_model,
        discrete_treatment=True,
        cv=5,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", category=ConvergenceWarning)
        model.fit(data.net_tfa.to_numpy(), data.e401.to_numpy(), X=x, W=w)
        pop = model.ate_inference(X=x)
        conditional_lower, conditional_upper = model.effect_interval(X=x, alpha=0.05)
        lower, upper = pop.conf_int_mean(alpha=0.05)
    return {
        "method": name,
        "specification": specification,
        "n": len(data),
        "seed": seed,
        "ate": float(np.squeeze(model.ate(X=x))),
        "pop_se": float(np.squeeze(pop.stderr_mean)),
        "pop_lo": float(np.squeeze(lower)),
        "pop_hi": float(np.squeeze(upper)),
        "mean_cate_lo": float(np.mean(conditional_lower)),
        "mean_cate_hi": float(np.mean(conditional_upper)),
        "journal_fivefold_plr": PAPER_REFERENCE[name],
        "convergence_warnings": sum(issubclass(x.category, ConvergenceWarning) for x in recorded),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--method", choices=["RF", "Lasso"], required=True)
    parser.add_argument("--spec", choices=["repo_X11", "paper_X9", "paper_W9"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sha = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    df = pd.read_csv(args.fixture)
    assert len(df) == 9915 and len(df.columns) == 14 and set(PAPER_CONTROLS) <= set(df)
    row = fit(df, args.method, args.spec, args.seed)
    print(pd.DataFrame([row | {"fixture_sha256": sha}]).to_csv(index=False), end="")


if __name__ == "__main__":
    main()
