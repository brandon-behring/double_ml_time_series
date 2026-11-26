"""Reproduce Indonesian teacher-pay Table V and compare two follow-ups of the same lottery."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

SOURCE = "https://dataverse.harvard.edu/api/access/datafile/3050669?format=original"
DIGEST = "f19a537f3dd5c0048b48d52cfff5d86dc4fd9d2c8b4cc50e40ac72723f80032e"
DIRECTORY = "20170903 Replication files DFN QJE"
BASELINE = ["ITT_score0", "missing_ITT_score0", "schoolscore0", "missing_schoolscore0"]
IDS = ["school_id", "triplet_id", "student_id", "year"]


def fit_wave(data, controls):
    """Weighted within-triplet slope and school contributions to its sampling error."""
    weights = data.weight.to_numpy(dtype=float)
    values = data[["ITT_score", "treatment", *controls]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or min(weights) <= 0:
        raise ValueError("Regression requires finite observations and positive sampling weights")
    strata, names = pd.factorize(data.triplet_id)
    totals = np.bincount(strata, weights)
    means = np.column_stack([np.bincount(strata, weights * col) / totals for col in values.T])
    centered = values - means[strata]
    y, x = centered[:, 0], centered[:, 1:]
    weighted_x = weights[:, None] * x
    gram = x.T @ weighted_x
    rank = np.linalg.matrix_rank(gram)
    control_rank = np.linalg.matrix_rank(gram[1:, 1:]) if controls else 0
    if rank == control_rank:
        raise ValueError("Within-triplet regression lacks independent treatment variation")
    # The school-score missing indicator is constant within each archived triplet.
    bread = np.linalg.pinv(gram, hermitian=True)
    coef = bread @ (weighted_x.T @ y)
    residual = y - x @ coef
    score = (weighted_x @ bread[:, 0]) * residual
    schools = pd.Series(score).groupby(data.school_id.to_numpy()).sum()
    n, g = len(data), len(schools)
    correction = g / (g - 1) * (n - 1) / (n - len(names) - rank)
    schools *= np.sqrt(correction)
    treated = data[["school_id", "treatment"]].drop_duplicates().treatment.sum()
    raw_y = values[:, 0]
    r_squared = 1 - np.average(residual**2, weights=weights) / np.average(
        (raw_y - np.average(raw_y, weights=weights)) ** 2, weights=weights
    )
    return {
        "effect_sd": float(coef[0]),
        "se_sd": float(np.linalg.norm(schools)),
        "r_squared": float(r_squared),
        "observations": n,
        "students": data.student_id.nunique(),
        "schools": g,
        "treated_schools": int(treated),
        "triplets": len(names),
    }, schools


def compare_waves(data, controls=BASELINE):
    """Allow the same school's estimation errors to covary across years."""
    if data[IDS].isna().any().any() or set(data.year.unique()) != {2, 3}:
        raise ValueError("Need identified schools, students and triplets in both follow-up years")
    membership = data[["school_id", "triplet_id", "treatment"]].drop_duplicates()
    if membership.school_id.duplicated().any() or set(membership.treatment) != {0, 1}:
        raise ValueError("Each school must have one randomized assignment and triplet across waves")
    first, later = (data.loc[data.year.eq(wave)] for wave in (2, 3))
    early, early_scores = fit_wave(first, controls)
    end, later_scores = fit_wave(later, controls)
    difference = early["effect_sd"] - end["effect_sd"]
    covariance = float(early_scores.mul(later_scores, fill_value=0).sum())
    se = float(np.linalg.norm(early_scores.sub(later_scores, fill_value=0)))
    critical = float(t.ppf(0.975, min(early["schools"], end["schools"]) - 1))
    return {
        "year_2": early,
        "year_3": end,
        "shared_schools": len(early_scores.index.intersection(later_scores.index)),
        "shared_students": len(set(first.student_id) & set(later.student_id)),
        "covariance_sd2": covariance,
        "early_minus_later_sd": difference,
        "difference_se_sd": se,
        "difference_ci95_sd": [difference - critical * se, difference + critical * se],
        "zero_minus_later_sd": -end["effect_sd"],
        "zero_ci95_sd": [
            -end["effect_sd"] - t.ppf(0.975, end["schools"] - 1) * end["se_sd"],
            -end["effect_sd"] + t.ppf(0.975, end["schools"] - 1) * end["se_sd"],
        ],
    }


def run(cache_dir):
    path = fetch(SOURCE, cache_dir, "indonesia-teacher-pay-3050669.zip", DIGEST)
    with ZipFile(path) as archive:
        program = archive.read(f"{DIRECTORY}/do/20170903 DFN Replication TABLES.do").decode(
            "latin1"
        )
        table = program.split("*Table 5", 1)[1].split("esttab", 1)[0]
        for year in (2, 3):
            if f"year=={year} [pweight=weight], cluster(school_id) absorb(triplet_id)" not in table:
                raise ValueError("Original Table V specification changed")
        with archive.open(f"{DIRECTORY}/dta/REPLICATION/students.dta") as source:
            data = pd.read_stata(
                source,
                convert_categoricals=False,
                columns=[*IDS, *BASELINE, "SD", "weight", "ITT_score", "treatment"],
            )
    data = data.loc[data.SD.isin([0, 1])]
    published = compare_waves(data)
    for wave, n, slope, se, r2 in [
        (2, 279066, -0.005, 0.024, 0.28),
        (3, 274993, 0.010, 0.026, 0.24),
    ]:
        result = published[f"year_{wave}"]
        if result["observations"] != n:
            raise ValueError("Table V student-subject sample size changed")
        np.testing.assert_allclose(
            [result["effect_sd"], result["se_sd"]], [slope, se], atol=0.0005, rtol=0
        )
        np.testing.assert_allclose(result["r_squared"], r2, atol=0.005, rtol=0)
    common = set(data.loc[data.year.eq(2), "school_id"]) & set(
        data.loc[data.year.eq(3), "school_id"]
    )
    return {
        "table_v": published,
        "common_school_sensitivity": compare_waves(data.loc[data.school_id.isin(common)]),
        "no_baseline_sensitivity": compare_waves(data, controls=[]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
