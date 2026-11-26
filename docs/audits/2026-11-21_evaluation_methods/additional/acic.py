"""Check ordinary PLR and a cross-fitted ATE score on the ACIC development files."""

import argparse
import hashlib
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dml_ts.dml.double_ml import double_ml

ARCHIVES = {
    "TestDatasets_lowD_Dec28.zip": "281c7723026ab81f5abbbe9ee4c9bc17e2c4cc95baccf8cb5c0e4d3f866d39d6",
    "TestDatasets_highD_Dec28.zip": "9734a8489dbb780c7d373bd853b1862a0a8c233d12826a45b44b07530e69d45c",
}


def evaluate(data, *, detailed=False):
    """Fit fixed fivefold estimators and optionally score held-out nuisance baselines."""
    y = data.Y.to_numpy(dtype=float)
    a = data.A.to_numpy(dtype=float)
    x = data.drop(columns=["Y", "A"]).to_numpy(dtype=float)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    plr = double_ml(y, a, x, outcome_model=ridge, treatment_model=logit, random_state=42)
    pred = {key: np.full(len(y), np.nan) for key in ["y", "p", "mu0", "mu1"]}
    if detailed:
        pred["baseline_y"] = np.empty_like(y)
        pred["baseline_p"] = np.empty_like(a)
    for train, test in KFold(5, shuffle=True, random_state=42).split(x):
        if detailed:
            pred["baseline_y"][test] = y[train].mean()
            pred["baseline_p"][test] = a[train].mean()
        pred["y"][test] = ridge.fit(x[train], y[train]).predict(x[test])
        pred["p"][test] = logit.fit(x[train], a[train]).predict_proba(x[test])[:, 1]
        for arm in (0, 1):
            group = train[a[train] == arm]
            pred[f"mu{arm}"][test] = ridge.fit(x[group], y[group]).predict(x[test])
    v = a - pred["p"]
    manual = np.dot(v, y - pred["y"]) / np.dot(v, v)
    np.testing.assert_allclose(manual, plr.theta, rtol=1e-9, atol=1e-11)
    p = np.clip(pred["p"], 0.01, 0.99)
    cate = pred["mu1"] - pred["mu0"]
    score = cate + a * (y - pred["mu1"]) / p - (1 - a) * (y - pred["mu0"]) / (1 - p)
    se = score.std(ddof=1) / np.sqrt(len(y))
    result = {
        "n": len(y),
        "covariates": x.shape[1],
        "treated": int(a.sum()),
        "arm_difference": y[a == 1].mean() - y[a == 0].mean(),
        "plr": plr.theta,
        "plr_manual_delta": plr.theta - manual,
        "residual_treatment_variance": np.mean(v**2),
        "outcome_mse": np.mean((y - pred["y"]) ** 2),
        "propensity_brier": np.mean(v**2),
        "propensity_min": np.min(pred["p"]),
        "propensity_max": np.max(pred["p"]),
        "probabilities_clipped": int((pred["p"] != p).sum()),
        "aipw": score.mean(),
        "aipw_se_iid": se,
    }
    if detailed:
        result["outcome_baseline_mse"] = np.mean((y - pred["baseline_y"]) ** 2)
        result["propensity_baseline_brier"] = np.mean((a - pred["baseline_p"]) ** 2)
        for arm in (0, 1):
            observed = a == arm
            result[f"mu{arm}_factual_mse"] = np.mean(
                (y[observed] - pred[f"mu{arm}"][observed]) ** 2
            )
            probability = p[observed] if arm == 1 else 1 - p[observed]
            weights = 1 / probability
            result[f"ess{arm}"] = weights.sum() ** 2 / np.dot(weights, weights)
    return result, cate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for name, digest in ARCHIVES.items():
        source = args.cache / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected archive digest for {name}")
        with ZipFile(source) as archive, threadpool_limits(1):
            observed = sorted(
                f
                for f in archive.namelist()
                if not f.startswith("__MACOSX/") and re.search(r"testdataset\d+\.csv$", f)
            )
            if len(observed) != 8:
                raise ValueError("Expected eight development files per track")
            for filename in observed:
                data = pd.read_csv(archive.open(filename))
                result, cate = evaluate(data)
                truth = pd.read_csv(archive.open(filename.removesuffix(".csv") + "_cf.csv"))
                truth = truth.rename(columns={"EY1_i": "EY1", "EY0_i": "EY0"})
                if len(truth) != len(data) or truth.ATE.nunique() != 1:
                    raise ValueError("Truth file does not match the observed sample")
                tau = truth.EY1 - truth.EY0
                rows.append(
                    {
                        "dataset": Path(filename).stem,
                        **result,
                        "population_ate": truth.ATE.iloc[0],
                        "sample_mean_cate": tau.mean(),
                        "aipw_minus_pate": result["aipw"] - truth.ATE.iloc[0],
                        "cate_rmse": np.sqrt(np.mean((cate - tau) ** 2)),
                    }
                )
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
