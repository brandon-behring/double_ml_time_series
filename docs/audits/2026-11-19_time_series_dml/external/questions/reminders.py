"""Compare the absolute one-hour effect of Drink Less reminders on days 1–7 and 24–30."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch
from sequential.drink_less import CONTROLS, SHA, URL, exposure_effect

PROBABILITY = 0.60
WINDOWS = ("early", "late")


def load_trial(cache_dir):
    path = fetch(URL, cache_dir, "drink-less.rds", SHA)
    # Ask base R to decode only the analysis columns in the authors' file.
    args = ["ID", "treatment", "primary_outcome", "prob_A", *CONTROLS]
    code = """
    args <- commandArgs(trailingOnly=TRUE)
    trial <- readRDS(args[1])
    write.csv(trial[args[-1]], stdout(), row.names=FALSE)
    """
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", code, str(path), *args], check=True, capture_output=True
    )
    frame = pd.read_csv(io.BytesIO(result.stdout))
    if (
        frame.isna().any().any()
        or not frame[["treatment", "primary_outcome"]].isin([0, 1]).all().all()
    ):
        raise ValueError("Incomplete or invalid assignments, outcomes or baseline controls")
    days = frame.groupby("ID")["days_since_download"]
    if (
        len(frame) != 10470
        or days.ngroups != 349
        or not days.apply(lambda d: sorted(d) == list(range(1, 31))).all()
    ):
        raise ValueError("The expected panel records every participant on every day from 1 to 30")
    np.testing.assert_allclose(frame["prob_A"], np.where(frame["treatment"], 0.6, 0.4))
    return frame


def augmented_score(assignment, outcome, mu0, mu1):
    """Known-probability AIPW. With zero outcome predictions this is Horvitz–Thompson."""
    return (
        mu1
        - mu0
        + assignment * (outcome - mu1) / PROBABILITY
        - (1 - assignment) * (outcome - mu0) / (1 - PROBABILITY)
    )


def predicted_risks(window):
    # Use identical participant folds in both weeks. Histories enter the model
    # only through the seven variables known before that evening's assignment.
    fold = pd.factorize(window["ID"], sort=True)[0] % 5
    a = window["treatment"].to_numpy()
    y = window["primary_outcome"].to_numpy()
    x = window[CONTROLS].to_numpy()
    risk = np.empty((len(window), 2))
    for k in range(5):
        held_out = fold == k
        for arm in (0, 1):
            train = (~held_out) & (a == arm)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
            model.fit(x[train], y[train])
            risk[held_out, arm] = model.predict_proba(x[held_out])[:, 1]
    return risk.T


def paired_summary(by_person):
    """Joint covariance of two mean scores with participants as sampling units."""
    effects = by_person.mean(axis=0)
    covariance = np.cov(by_person, rowvar=False, ddof=1) / len(by_person)
    critical = t.ppf(0.975, len(by_person) - 1)
    output = {"covariance": covariance.tolist()}
    for name, contrast in [("early", [1, 0]), ("late", [0, 1]), ("late_minus_early", [-1, 1])]:
        contrast = np.asarray(contrast)
        effect = effects @ contrast
        se = np.sqrt(max(0.0, contrast @ covariance @ contrast))
        output[name] = {
            "risk_difference": effect,
            "se": se,
            "lower": effect - critical * se,
            "upper": effect + critical * se,
        }
    return output


def run(cache_dir):
    trial = load_trial(cache_dir)
    # Reproduce the article's different, full-month relative-risk estimand first.
    published = exposure_effect(trial, "primary_outcome", trial["prob_A"].to_numpy())
    np.testing.assert_allclose(
        [published[k] for k in ("ratio", "lower", "upper")],
        [3.523, 2.918, 4.255],
        atol=0.0005,
        rtol=0,
    )
    days = trial["days_since_download"]
    frame = trial[days.between(1, 7) | days.between(24, 30)].copy()
    frame["window"] = np.where(frame["days_since_download"] <= 7, "early", "late")
    counts = {}
    for name in WINDOWS:
        mask = frame["window"].eq(name)
        window = frame.loc[mask]
        a = window["treatment"].to_numpy()
        y = window["primary_outcome"].to_numpy()
        mu0, mu1 = predicted_risks(window)
        frame.loc[mask, "ipw"] = augmented_score(a, y, 0.0, 0.0)
        frame.loc[mask, "aipw"] = augmented_score(a, y, mu0, mu1)
        counts[name] = {
            label: {
                "decisions": int(sum(a == arm)),
                "opens": int(y[a == arm].sum()),
                "observed_risk": float(y[a == arm].mean()),
            }
            for arm, label in [(0, "none"), (1, "any")]
        }
    if not (frame.groupby(["ID", "window"]).size() == 7).all():
        raise ValueError("Expected seven decisions per participant in each window")

    estimates = {}
    for method in ["ipw", "aipw"]:
        by_person = frame.groupby(["ID", "window"])[method].mean().unstack()
        estimates[method] = paired_summary(by_person[list(WINDOWS)].to_numpy())
    return {
        "paper_table2_relative_risk": published,
        "participants": frame["ID"].nunique(),
        "scored_decisions": len(frame),
        "assignment_probability": PROBABILITY,
        "counts": counts,
        "estimates": estimates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2))
