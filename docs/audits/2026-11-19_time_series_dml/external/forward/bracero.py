"""Replay the bracero exclusion wage estimate and test three earlier placebo dates."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

HARVARD = "https://dataverse.harvard.edu/api/access/datafile"
SOURCES = {
    "farm.dta": (3240617, "fc8354297a9d194a87e8afe7d18dd8c6197719bec614ec727fbcb8d8b6141ee5"),
    "cpi.dta": (3240623, "cfbbf1765300d4b2e26ff428da42a9296fc915065ed7792c8f623564701846e2"),
    "author.do": (3240615, "c0ef20d367551ba6bba97442d891e3d748dc77b97980576577525767adc60789"),
}


def exposure_1955(farm):
    """Average valid monthly fractions within state, following the author's Stata code."""
    months = farm.loc[farm.Year.eq(1955)].dropna(subset=["State_FIPS", "Month"])
    denominator = months.TotalHiredSeasonal_final.where(months.TotalHiredSeasonal_final > 0)
    fractions = months.Mexican_final.fillna(0) / denominator
    return fractions.groupby(months.State).mean().rename("exposure")


def wage_panel(farm, cpi):
    exposure = exposure_1955(farm)
    wages = (
        farm.loc[farm.Month.isin([1, 4, 7, 10])]
        .dropna(subset=["State_FIPS", "HourlyComposite_final"])
        .copy()
    )
    wages["time_m"] = (wages.Year - 1960) * 12 + wages.Month - 1
    wages = wages.merge(cpi, on=["State_FIPS", "time_m"], validate="one_to_one")
    wages["wage"] = wages.HourlyComposite_final.astype(float) * 0.1966401 / wages.cpi
    wages["quarter"] = pd.PeriodIndex.from_fields(
        year=wages.Year, quarter=1 + (wages.Month.astype(int) - 1) // 3, freq="Q"
    )
    return wages.join(exposure, on="State").dropna(subset=["wage", "exposure"])


def wage_contrast(panel, cutoff_year):
    """State and calendar-quarter fixed effects, with Stata's state-cluster correction."""
    rows = panel.sort_values(["State", "quarter"]).reset_index(drop=True)
    if rows.duplicated(["State", "quarter"]).any():
        raise ValueError("Duplicate state-quarter")
    state = rows.State
    clusters = state.nunique()
    quarters = rows.quarter.nunique()
    if clusters < 2 or quarters < 2:
        raise ValueError("At least two states and two quarters are required")
    post = rows.quarter.dt.year.ge(cutoff_year)
    variables = pd.DataFrame({"x": post * rows.exposure.astype(float), "y": rows.wage})
    within = variables - variables.groupby(state).transform("mean")
    time_effects = pd.get_dummies(rows.quarter, drop_first=True, dtype=float)
    controls = time_effects - time_effects.groupby(state).transform("mean")
    weights, _, rank, _ = np.linalg.lstsq(controls, within, rcond=None)
    x, y = (within.to_numpy() - controls.to_numpy() @ weights).T
    xx = x @ x
    if xx < np.finfo(float).eps * max(1, len(rows)) or rank != quarters - 1:
        raise ValueError("State and time effects leave insufficient variation")
    estimate = x @ y / xx
    scores = pd.Series(x * (y - estimate * x)).groupby(state).sum().to_numpy()
    # xtreg, fe counts the intercept, time contrasts and slope in CR1. State effects
    # are nested within the clustering variable, so they consume no additional df here.
    k = quarters + 1
    correction = clusters / (clusters - 1) * (len(rows) - 1) / (len(rows) - k)
    se = np.sqrt(correction * (scores @ scores)) / xx
    radius = t.ppf(0.975, clusters - 1) * se
    p_value = 2 * t.sf(abs(estimate) / se, clusters - 1) if se > 0 else float(estimate == 0)
    return {
        "estimate": estimate,
        "se": se,
        "ci95": [estimate - radius, estimate + radius],
        "p": p_value,
        "state_quarters": len(rows),
        "states": clusters,
        "quarters": quarters,
        "first_quarter": str(rows.quarter.min()),
        "last_quarter": str(rows.quarter.max()),
    }


def placebo_grid(panel):
    """Two complete years on each side of each January pseudo-intervention."""
    results = {}
    for year in (1958, 1959, 1960):
        window = panel.loc[panel.quarter.dt.year.between(year - 2, year + 1)]
        if window.quarter.nunique() != 16:
            raise ValueError("Incomplete calendar window for placebo grid")
        results[str(year)] = wage_contrast(window, year)
    adjusted = multipletests([row["p"] for row in results.values()], method="holm")[1]
    for row, p_holm in zip(results.values(), adjusted, strict=True):
        row["p_holm"] = p_holm
    return results


def run(cache):
    paths = {
        name: fetch(
            f"{HARVARD}/{file_id}?format=original",
            cache,
            f"bracero-{name}",
            digest,
            user_agent="Mozilla/5.0",
        )
        for name, (file_id, digest) in SOURCES.items()
    }
    if "xtreg realwage_hourly treatment_frac i.time_q_plus, fe vce(cluster State_FIPS)" not in (
        paths["author.do"].read_text()
    ):
        raise ValueError("Archive does not contain the expected Table 1 specification")
    farm, cpi = (
        pd.read_stata(paths[name], convert_categoricals=False, convert_dates=False)
        for name in ("farm.dta", "cpi.dta")
    )
    panel = wage_panel(farm, cpi)
    published = wage_contrast(panel, 1965)
    if (published["states"], published["state_quarters"]) != (46, 4324):
        raise ValueError("Estimation sample differs from published Table 1, column 1")
    np.testing.assert_allclose(
        [published["estimate"], published["se"]], [-0.0356, 0.0426], atol=0.00005, rtol=0
    )
    descriptive = panel.groupby("State").exposure.first()
    return {
        "units": "1965 US dollars per hour per unit of 1955 exposure",
        "table_1_hourly": published,
        "exposure_1955": {
            "zero_states": int(descriptive.eq(0).sum()),
            "low_states": int(((descriptive > 0) & (descriptive < 0.2)).sum()),
            "high_states": descriptive[descriptive >= 0.2].index.tolist(),
        },
        "placebos": placebo_grid(panel),
        "exploratory_1959_six_years": wage_contrast(
            panel.loc[panel.quarter.dt.year.between(1956, 1961)], 1959
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
