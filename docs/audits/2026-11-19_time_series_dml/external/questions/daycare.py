"""Center-level reanalysis of Table 1 of Gneezy and Rustichini (2000)."""

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402

from dml_ts.dml.fwl import fwl_estimate  # noqa: E402

PAPER = (
    "https://raw.githubusercontent.com/OpenIntroStat/openintro/"
    "001a16cb7f599472c22b61331060bb1981d0cc0f/data-raw/daycare_fines/fine.pdf"
)
SHA256 = "06b583380fc9bf1c876747acea4ba92952a57f6253add3bacd8e4a51cf2ef022"


def published_counts(text):
    """Read only the ten integer rows in the published count table, in center order."""
    table = text.split("Test group:", 1)[1].split("The columns under", 1)[0]
    rows = [
        list(map(int, line.split()))
        for line in table.splitlines()
        if re.fullmatch(r"\s*\d+(?:\s+\d+){21}\s*", line)
    ]
    if len(rows) != 10 or [row[0] for row in rows] != list(range(1, 11)):
        raise ValueError("Expected all ten centers and 20 weeks from published Table 1")
    return pd.DataFrame([row[2:] for row in rows], index=range(1, 11), columns=range(1, 21))


def randomization(changes, assigned, alpha=0.05):
    """Enumerate the center assignments under the sharp null; invert constant-shift tests."""
    changes = np.asarray(changes, dtype=float)
    assigned = np.asarray(assigned, dtype=bool)
    n = len(assigned)
    treated = int(assigned.sum())
    if not 0 < treated < n or changes.shape != (n,) or not np.isfinite(changes).all():
        raise ValueError("Expected one complete change per center and two assignment arms")
    allocations = list(itertools.combinations(range(n), treated))
    treatment_masks = np.zeros((len(allocations), n), dtype=bool)
    for i, allocation in enumerate(allocations):
        treatment_masks[i, list(allocation)] = True
    weights = treatment_masks / treated - (~treatment_masks) / (n - treated)
    observed = float(changes[assigned].mean() - changes[~assigned].mean())
    permutations = weights @ changes
    extreme = int(np.count_nonzero(np.abs(permutations) >= abs(observed) - 1e-12))

    # Under a hypothesized constant shift tau, remove tau from the observed treated
    # centers and permute. Each assignment's two-sided acceptance interval contains
    # the observed difference. At either endpoint at least ceil(alpha * C(n,k))
    # assignments must still be as extreme. The observed assignment always ties.
    b = weights @ assigned.astype(float)
    interior = np.abs(b) < 1 - 1e-12
    c = permutations[interior] - b[interior] * observed
    endpoints = np.column_stack((c / (b[interior] - 1), c / (b[interior] + 1))) + observed
    left = np.full(len(allocations), -np.inf)
    right = np.full(len(allocations), np.inf)
    left[interior], right[interior] = endpoints.min(axis=1), endpoints.max(axis=1)
    needed = int(np.ceil(alpha * len(allocations)))
    lower, upper = np.sort(left)[needed - 1], np.sort(right)[-needed]
    return {
        "did": observed,
        "assignments": len(allocations),
        "as_extreme": extreme,
        "two_sided_p": extreme / len(allocations),
        "constant_effect_interval": [float(lower), float(upper)],
    }


def phase_summary(counts, weeks):
    pre = counts.loc[:, 1:4].mean(axis=1).to_numpy()
    post = counts.loc[:, weeks].mean(axis=1).to_numpy()
    assigned = np.asarray(counts.index <= 6)
    result = randomization(post - pre, assigned)
    result["post_mean_fined"] = post[assigned].mean()
    result["post_mean_control"] = post[~assigned].mean()
    result["unadjusted_arm_difference"] = result["post_mean_fined"] - result["post_mean_control"]
    # One observation per randomized center. All covariate information predates the fine.
    fit = fwl_estimate(post, assigned.astype(float), pre[:, None], method="ols")
    reference = sm.OLS(post, np.column_stack((np.ones(len(pre)), assigned, pre))).fit(
        cov_type="HC3", use_t=True
    )
    np.testing.assert_allclose(fit.theta, reference.params[1], atol=1e-10, rtol=0)
    result["baseline_adjusted_fwl"] = float(fit.theta)
    result["adjusted_hc3_interval"] = reference.conf_int()[1].tolist()
    return result


def study(cache_dir):
    path = fetch(PAPER, cache_dir, "gneezy-rustichini-2000.pdf", SHA256)
    published = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    counts = published_counts(published)
    main = phase_summary(counts, list(range(5, 17)))
    after = phase_summary(counts, list(range(17, 21)))
    np.testing.assert_allclose(
        [main["did"], after["did"]], [9.21527777778, 12.45833333333], atol=1e-8, rtol=0
    )
    if main["assignments"] != 210 or main["as_extreme"] != 1:
        raise AssertionError("Fine-period assignment distribution disagrees with Table 1")
    return {
        "centers": len(counts),
        "reported_weeks": len(counts.columns),
        "calendar_weeks": 21,
        "during": main,
        "after_repeal": after,
        "exclude_short_week_11": phase_summary(counts, [w for w in range(5, 17) if w != 11]),
        "center_1_checks": {
            "pre": float(counts.loc[1, 1:4].mean()),
            "first_four_fine_weeks": float(counts.loc[1, 5:8].mean()),
            "after": float(counts.loc[1, 17:20].mean()),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(study(parser.parse_args().cache_dir), indent=2, allow_nan=False))
