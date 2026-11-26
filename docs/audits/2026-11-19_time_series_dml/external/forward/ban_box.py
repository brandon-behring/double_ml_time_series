"""Replay Agan–Starr Table V and forecast a randomized callback contrast across waves."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402

HARVARD = "https://dataverse.harvard.edu/api/access/datafile"
SOURCE = {
    "dta": (3036350, "2358d56b73398b04dfd480bfdba9ed98bdb1e1897c22d6c188bf57689020ed2c"),
    "do": (3036351, "b89f989a3207fc15174f90d642f2982f182e5db3d553510ddb89f97c7872cefc"),
}
# Pre window, policy effective date, and post window, as dated in section II.
DATES = {
    "New Jersey": ("2015-01-31", "2015-02-28", "2015-03-01", "2015-05-04", "2015-06-12"),
    "New York City": ("2015-06-10", "2015-08-30", "2015-10-27", "2015-11-30", "2016-03-31"),
}


def stata_day(date):
    return (pd.Timestamp(date) - pd.Timestamp("1960-01-01")).days


def published(data):
    columns = [
        "post_remover_white",
        "post_white",
        "post_remover",
        "remover_white",
        "remover",
        "white",
        "post",
        "ged",
        "empgap",
    ]
    use = data.loc[data.balanced.eq(1) & data.remover.ne(-1)]
    fit = sm.OLS(use.response, sm.add_constant(use[columns])).fit(
        cov_type="cluster", cov_kwds={"groups": use.chain_id}
    )
    # Figure in Table V uses proportion units.
    if len(use) != 11188:
        raise ValueError("Original balanced estimation sample does not match Table V")
    np.testing.assert_allclose(
        [fit.params.iloc[1], fit.bse.iloc[1]], [0.039, 0.020], atol=0.00051, rtol=0
    )
    return {
        "applications": len(use),
        "chains": int(use.chain_id.nunique()),
        "effect_pp": 100 * fit.params.iloc[1],
        "se_pp": 100 * fit.bse.iloc[1],
    }


def eligible_waves(data, dates):
    start, end, law, start_late, end_late = map(stata_day, dates)
    early = data.post.eq(0) & data.app_date_d.between(start, end)
    later = data.post.eq(1) & data.app_date_d.between(start_late, end_late)
    # Data contain positive callbacks but also dates for other kinds of responses.
    elapsed = data.response_date_d - data.app_date_d
    known = data.response.eq(0) | (data.response.eq(1) & elapsed.between(0, 56))
    early_observed = early & known & (data.app_date_d + 56 < start_late)
    later_observed = later & known
    audit = {
        "outside_paper_dates": int((~(early | later)).sum()),
        "positive_with_unusable_date": int(((early | later) & ~known).sum()),
        "early_not_mature_at_post": int((early & known & ~early_observed).sum()),
        "early_mature_by_law": int((early & known & (data.app_date_d + 56 < law)).sum()),
    }
    return data.loc[early_observed | later_observed], audit


def comparison(data):
    """Estimate both name gaps in a saturated model and allow chain shocks across waves."""
    y, white, post = (data[c].to_numpy(float) for c in ("response", "white", "post"))
    if len(set(zip(white, post, strict=True))) != 4:
        raise ValueError("Both name signals required in each wave")
    design = np.column_stack([1 - post, post, white * (1 - post), white * post])
    fit = sm.OLS(y, design).fit()
    chain_scores = pd.DataFrame(design * fit.resid[:, None]).groupby(data.chain_id.to_numpy()).sum()
    n_chains = len(chain_scores)
    correction = (n_chains / (n_chains - 1)) * (len(y) - 1) / fit.df_resid
    critical = t.ppf(0.975, n_chains - 1)
    output = {}
    for key, contrast in {
        "early": [0, 0, 1, 0],
        "later": [0, 0, 0, 1],
        "early_minus_later": [0, 0, 1, -1],
        "zero_minus_later": [0, 0, 0, -1],
    }.items():
        v = np.asarray(contrast)
        point = float(100 * (v @ fit.params))
        influence = chain_scores.to_numpy() @ (fit.normalized_cov_params @ v)
        se = float(100 * np.linalg.norm(influence) * np.sqrt(correction))
        output[key] = {"estimate": point, "ci95": [point - critical * se, point + critical * se]}
    earlier, later = data[data.post.eq(0)], data[data.post.eq(1)]
    for name, frame in (("early", earlier), ("later", later)):
        means = frame.response.astype(float).groupby(frame.white).mean()
        np.testing.assert_allclose(
            output[name]["estimate"], 100 * (means[1] - means[0]), atol=1e-10
        )
        output[name]["applications"] = len(frame)
        output[name]["chains"] = int(frame.chain_id.nunique())
    output["shared_chains"] = len(set(earlier.chain_id) & set(later.chain_id))
    return output


def run(cache):
    paths = {
        ext: fetch(f"{HARVARD}/{i}?format=original", cache, f"ban-the-box-{i}.{ext}", checksum)
        for ext, (i, checksum) in SOURCE.items()
    }
    if "Table V: Effects of Ban the Box" not in paths["do"].read_text():
        raise ValueError("Original analysis source missing Table V")
    # Stata day counts preserve erroneous years outside pandas' nanosecond range.
    data = pd.read_stata(paths["dta"], convert_dates=False, convert_categoricals=False)
    out = {"table_v": published(data), "waves": {}}
    for place, dates in DATES.items():
        site = data[data.nj.eq(int(place == "New Jersey"))]
        eligible, audit = eligible_waves(site, dates)
        out["waves"][place] = {
            "chronology": audit,
            "documented_dates": comparison(eligible),
            "archive_wave_flag": comparison(site),
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
