"""Reproduce Drink Less Tables 2 and 6 with participant-clustered inference."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch

URL = "https://osf.io/download/mtcfa/"
SHA = "d7abef054a6734a6c1deec561ef5a015d3378f9030ec777c34c7e7f8395be73c"
CONTROLS = [
    "age",
    "AUDIT_score",
    "days_since_download",
    "gender",
    "employment_type",
    "after_9pm_day_before",
    "before_8pm",
]


def exposure_effect(frame, outcome, probability):
    """Solve the authors' multiplicative estimating equation for the pooled notification."""
    a = frame["treatment"].to_numpy()
    y = frame[outcome].to_numpy()
    z = np.column_stack([np.ones(len(a)), frame[CONTROLS]])
    design = np.column_stack([z, a - 0.5])
    weight = np.where(a == 1, 0.5 / probability, 0.5 / (1 - probability))
    instruments = weight[:, None] * design

    def score_terms(theta):
        baseline = np.exp(z @ theta[:-1])
        adjusted_outcome = y * np.exp(-a * theta[-1])
        return adjusted_outcome, baseline

    def equation(theta):
        adjusted_outcome, baseline = score_terms(theta)
        return instruments.T @ (adjusted_outcome - baseline)

    def jacobian(theta):
        adjusted_outcome, baseline = score_terms(theta)
        return -instruments.T @ np.column_stack([baseline[:, None] * z, a * adjusted_outcome])

    solution = root(equation, np.zeros(design.shape[1]), jac=jacobian, tol=1e-10)
    if not solution.success or np.max(np.abs(equation(solution.x))) > 1e-5:
        raise ValueError(f"Estimating equation failed to converge: {solution.message}")
    adjusted_outcome, baseline = score_terms(solution.x)
    scores = instruments * (adjusted_outcome - baseline)[:, None]
    groups = pd.factorize(frame["ID"])[0]
    clustered = np.zeros((groups.max() + 1, scores.shape[1]))
    np.add.at(clustered, groups, scores)
    influence = np.linalg.inv(jacobian(solution.x))[-1]
    log_se = np.linalg.norm(clustered @ influence)
    beta = solution.x[-1]
    ci = np.exp(beta + np.array([-1, 1]) * norm.ppf(0.975) * log_se)
    return {
        "ratio": np.exp(beta),
        "lower": ci[0],
        "upper": ci[1],
        "log_se": log_se,
        "independent_row_log_se": np.linalg.norm(scores @ influence),
    }


def cross_fitted_risk_difference(frame):
    """One-hour AIPW risk difference using the trial's known 0.6 assignment probability."""
    x = frame[CONTROLS].to_numpy()
    a = frame["treatment"].to_numpy()
    y = frame["primary_outcome"].to_numpy()
    person = pd.factorize(frame["ID"], sort=True)[0]
    fold = person % 5
    predictions = np.empty((len(frame), 2))
    for k in range(5):
        test = fold == k
        for arm in (0, 1):
            train = (~test) & (a == arm)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            model.fit(x[train], y[train])
            predictions[test, arm] = model.predict_proba(x[test])[:, 1]
    mu0, mu1 = predictions.T
    score = mu1 - mu0 + a * (y - mu1) / 0.6 - (1 - a) * (y - mu0) / 0.4
    estimate = score.mean()
    totals = np.bincount(person, weights=score - estimate)
    se = np.linalg.norm(totals) / len(score)
    return {
        "risk_difference": estimate,
        "se": se,
        "lower": estimate - norm.ppf(0.975) * se,
        "upper": estimate + norm.ppf(0.975) * se,
    }


def run(cache_dir):
    path = fetch(URL, cache_dir, "drink-less.rds", SHA)
    # Base R handles the original RDS classes and message encodings.
    read_rds = """
    d <- readRDS(commandArgs(trailingOnly=TRUE)[1])
    d$message <- NULL
    write.csv(d, stdout(), row.names=FALSE)
    """
    converted = subprocess.run(
        ["Rscript", "--vanilla", "-e", read_rds, str(path)], capture_output=True, check=True
    )
    data = pd.read_csv(io.BytesIO(converted.stdout))
    if (
        not data.groupby("ID")["days_since_download"]
        .apply(lambda s: sorted(s) == list(range(1, 31)))
        .all()
        or data["ID"].nunique() != 349
    ):
        raise ValueError("Expected 349 participants each observed on days 1–30")
    arms = data.groupby("subversion")["primary_outcome"].agg(["size", "sum"])
    if arms.to_dict("index") != {
        "X": {"size": 3146, "sum": 384},
        "Y": {"size": 3112, "sum": 407},
        "Z": {"size": 4212, "sum": 153},
    }:
        raise ValueError("Notification allocations or observed one-hour opens changed")
    np.testing.assert_array_equal(data["prob_A"], np.where(data["treatment"], 0.6, 0.4))
    estimates = {
        "one_hour": exposure_effect(data, "primary_outcome", data["prob_A"].to_numpy()),
        "24_hours": exposure_effect(data, "outcome_24hour", data["prob_A"].to_numpy()),
        "one_hour_design_probability": exposure_effect(data, "primary_outcome", 0.6),
        "one_hour_aipw": cross_fitted_risk_difference(data),
    }
    for name, target in [("one_hour", [3.523, 2.918, 4.255]), ("24_hours", [1.260, 1.187, 1.337])]:
        measured = [estimates[name][key] for key in ("ratio", "lower", "upper")]
        np.testing.assert_allclose(measured, target, atol=0.0005, rtol=0)
    return {"participants": 349, "decision_points": len(data), "estimates": estimates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2))
