"""Independently check the official truth join, every coverage verdict and 32 raw-data fits."""

import argparse
from pathlib import Path
from statistics import NormalDist
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

Z95 = NormalDist().inv_cdf(0.975)
TOLERANCE = 1e-9


def arm_prediction(x, y, training, held):
    """Closed-form standardized ridge, with a free intercept and unit ridge penalty."""
    scale = StandardScaler().fit(x[training])
    design = scale.transform(x[training])
    center = design.mean(axis=0)
    design -= center
    outcome_mean = y[training].mean()
    beta = np.linalg.solve(
        design.T @ design + np.eye(design.shape[1]), design.T @ (y[training] - outcome_mean)
    )
    return outcome_mean + (scale.transform(x[held]) - center) @ beta


def independent_interval(data):
    y = data["Y"].to_numpy(dtype=float)
    a = data["A"].to_numpy()
    x = data.drop(columns=["Y", "A"]).to_numpy(dtype=float)
    n = len(a)
    # KFold's shuffled membership, with fitting/scoring rows in original row order.
    membership = np.random.RandomState(42).permutation(n)
    signals = np.empty(n)
    for held_unsorted in np.array_split(membership, 5):
        held = np.sort(held_unsorted)
        training = np.ones(n, dtype=bool)
        training[held] = False
        scale = StandardScaler().fit(x[training])
        propensity = LogisticRegression(C=1, max_iter=2000).fit(
            scale.transform(x[training]), a[training]
        )
        p = np.clip(propensity.predict_proba(scale.transform(x[held]))[:, 1], 0.01, 0.99)
        mu0 = arm_prediction(x, y, training & (a == 0), held)
        mu1 = arm_prediction(x, y, training & (a == 1), held)
        correction = np.where(a[held] == 1, (y[held] - mu1) / p, -(y[held] - mu0) / (1 - p))
        signals[held] = mu1 - mu0 + correction
    ate = signals.sum() / n
    se = np.linalg.norm(signals - ate) / np.sqrt(n * (n - 1))
    return np.array([ate, se, ate - Z95 * se, ate + Z95 * se])


def check(data_zip, truth_zip, study):
    with ZipFile(truth_zip) as archive:
        truth = pd.read_csv(archive.open("lowDim_trueATE.csv")).set_index("filename")
    draws = pd.read_csv(study / "draws.csv").set_index("dataset")
    aggregate = pd.read_csv(study / "by-design.csv").set_index("design")
    if any(not table.index.is_unique for table in (truth, draws, aggregate)):
        raise ValueError("Duplicate dataset or design key")
    sizes = truth.groupby("DGPid").size()
    if len(sizes) != 32 or not sizes.eq(100).all() or set(draws.index) != set(truth.index):
        raise ValueError("Incomplete design or dataset membership")
    truth = truth.loc[draws.index]
    np.testing.assert_array_equal(draws.design, truth.DGPid)
    np.testing.assert_allclose(draws.population_ate, truth.trueATE, atol=TOLERANCE, rtol=TOLERANCE)
    if not draws.status.isin(["ok", "failed"]).all():
        raise ValueError("Unknown fit status")
    ok = draws.status.eq("ok")
    if not np.isfinite(draws.loc[ok, ["aipw", "aipw_se_iid", "ci_lower", "ci_upper"]]).all().all():
        raise ValueError("Nonfinite fit reported as successful")
    valid = draws.loc[ok]
    if valid.aipw_se_iid.lt(0).any():
        raise ValueError("Negative standard error on a successful fit")
    for endpoint, sign in [(valid.ci_lower, -1), (valid.ci_upper, 1)]:
        np.testing.assert_allclose(
            endpoint, valid.aipw + sign * Z95 * valid.aipw_se_iid, rtol=TOLERANCE, atol=TOLERANCE
        )
    hits = ok & truth.trueATE.ge(draws.ci_lower) & truth.trueATE.le(draws.ci_upper)
    np.testing.assert_array_equal(hits.astype(int), draws.covered)
    if set(aggregate.index) != set(sizes.index):
        raise ValueError("Unexpected set of reported designs")
    aggregate = aggregate.loc[sizes.index]
    np.testing.assert_array_equal(aggregate.planned, sizes)
    np.testing.assert_array_equal(aggregate.covered, hits.groupby(truth.DGPid).sum())
    max_difference = 0.0
    with ZipFile(data_zip) as archive, threadpool_limits(limits=1):
        if {name + ".csv" for name in truth.index} != set(archive.namelist()):
            raise ValueError("Contest ZIP members differ from the answer key")
        for _, group in truth.groupby("DGPid"):
            dataset = min(group.index, key=lambda name: int(name.removeprefix("low")))
            expected = draws.loc[dataset, ["aipw", "aipw_se_iid", "ci_lower", "ci_upper"]].to_numpy(
                dtype=float
            )
            actual = independent_interval(pd.read_csv(archive.open(dataset + ".csv")))
            np.testing.assert_allclose(
                actual, expected, rtol=TOLERANCE, atol=TOLERANCE, err_msg=dataset
            )
            max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))
    print(
        f"{len(draws)} official keys and coverage flags matched. {len(aggregate)} design totals matched."
    )
    print(f"Verified Wald interval arithmetic for {len(valid)} successful fits.")
    print(f"Reconstructed {len(aggregate)} prespecified estimates, SEs and interval bounds.")
    print(f"Maximum absolute difference {max_difference:.12g}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("truth_zip", type=Path)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    check(args.data_zip, args.truth_zip, args.study)
