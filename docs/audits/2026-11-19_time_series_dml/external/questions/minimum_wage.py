"""Reproduce restaurant employment results and evaluate a short-run wage-change model."""

import argparse
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

EXTERNAL = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXTERNAL.parents[3]), str(EXTERNAL)]
from benchmark_io import fetch  # noqa: E402

from dml_ts.dml._panel import panel_standard_error  # noqa: E402
from dml_ts.dml.fwl import fwl_estimate  # noqa: E402

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/"
SOURCES = {
    "njmin.zip": (
        "https://davidcard.berkeley.edu/data_sets/njmin.zip",
        "41a29aec6b489658b20030d044a2b20a5f5a04b871cde41df057afd6013ba587",
    ),
    "dlr-minimumwage.zip": (
        DATAVERSE + "2304888",
        "8c02c15bf1ac880ce678a52e5a99b1d5f4887a2edfad840d5f6cdab81e146303",
    ),
    "jnr-cz-panel.dta": (
        DATAVERSE + "9261285?format=original",
        "f533ede4ab2190ba1e8b4c12d4fd4c4de7ba78df26b7d78ff9b8629eee3b153c",
    ),
    "jnr-cz-pairs.dta": (
        DATAVERSE + "9261204?format=original",
        "61c4f4fa9be6f6addd1503153e1df682838db95ff44a6cd420a83589cd502a7b",
    ),
}

EMP = "lnemp_rest_both"
PAY = "lnAWW_rest_both"
WAGE = "lnMW"
VARIABLES = [EMP, PAY, WAGE, "lnpop", "lnemp_TOT", "lnAWW_TOT"]
IDS = ["county", "period", "state_fips", "nonmissing_rest_both"]


def store_change(frame):
    groups = frame.groupby("STATE").change
    return {
        "estimate": float(groups.mean()[1] - groups.mean()[0]),
        "se": float(np.sqrt((groups.var() / groups.count()).sum())),
        "pa": int(groups.count()[0]),
        "nj": int(groups.count()[1]),
    }


def store_att(sample):
    """Cross-fitted normalized doubly robust DiD score on a two-period store panel."""
    d = sample.sort_values(["STATE", "SHEET"])
    if d.duplicated(["STATE", "SHEET"]).any():
        raise ValueError("Each store must contribute exactly one signed change")
    x = pd.get_dummies(d[["CHAIN", "CO_OWNED", "baseline_fte"]], columns=["CHAIN"], dtype=float)
    a = d.STATE.to_numpy(dtype=int)
    change = d.change.to_numpy()
    mu0, p = np.empty(len(d)), np.empty(len(d))
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=1992)
    for train, test in splits.split(x, a):
        control = train[a[train] == 0]
        reg = RandomForestRegressor(n_estimators=64, min_samples_leaf=12, random_state=13, n_jobs=1)
        propensity = RandomForestClassifier(
            n_estimators=64, min_samples_leaf=20, random_state=13, n_jobs=1
        )
        mu0[test] = reg.fit(x.iloc[control], change[control]).predict(x.iloc[test])
        p[test] = propensity.fit(x.iloc[train], a[train]).predict_proba(x.iloc[test])[:, 1]
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("Store propensity failed overlap on held-out observations")
    untreated = a == 0
    odds = p[untreated] / (1 - p[untreated])
    residual = change - mu0
    return {
        "constant_nuisances": float(change[~untreated].mean() - change[untreated].mean()),
        "forest": float(
            residual[~untreated].mean() - np.average(residual[untreated], weights=odds)
        ),
        "min_propensity": float(p.min()),
        "max_propensity": float(p.max()),
        "effective_control_stores": float(odds.sum() ** 2 / (odds @ odds)),
        "n": len(d),
    }


def store_study(path):
    # Column positions come from the archive's fixed-width codebook.
    with ZipFile(path) as archive:
        specs, names = [], []
        for line in archive.read("codebook").decode("latin1").splitlines():
            bits = line.split()
            if len(bits) > 3 and bits[1].isdigit() and bits[2].isdigit() and bits[0].isupper():
                names.append(bits[0])
                specs.append((int(bits[1]) - 1, int(bits[2])))
        with archive.open("public.dat") as source:
            data = pd.read_fwf(source, colspecs=specs, names=names, na_values=".")
    if data.duplicated(["STATE", "SHEET"]).any():
        raise ValueError("A state/store ID occurs twice in the original survey")
    fte = data.EMPFT + data.NMGRS + 0.5 * data.EMPPT
    fte2 = data.EMPFT2 + data.NMGRS2 + 0.5 * data.EMPPT2
    data["change"] = fte2 - fte
    data["baseline_fte"] = fte
    base = store_change(data)
    temporary = data.STATUS2.isin([2, 4, 5])
    closed = store_change(data.assign(change=fte2.mask(temporary, 0) - fte))
    complete = data.dropna(subset=["change"])
    regression = complete.loc[
        (complete.STATUS2 == 3) | complete[["WAGE_ST", "WAGE_ST2"]].notna().all(axis=1)
    ]
    chain = pd.get_dummies(regression.CHAIN, dtype=float).to_numpy()[:, :3]
    controls = np.column_stack([chain, regression.CO_OWNED])
    res = fwl_estimate(regression.change.to_numpy(), regression.STATE.to_numpy(), controls)
    np.testing.assert_allclose(
        [base["estimate"], closed["estimate"], res.theta, res.se],
        [2.75, 2.51, 2.30, 1.20],
        atol=0.005,
        rtol=0,
    )
    return {
        "archive_stores": len(data),
        "unique_sheet_numbers": data.SHEET.nunique(),
        "matched": base,
        "temporary_zero": closed,
        "matched_store_att": store_att(complete),
        "table4_chain_ownership": {"estimate": res.theta, "se": res.se, "n": len(regression)},
    }


def legacy_segment(pair):
    """Literal numeric substr()/min()/max() calls in both Stata scripts."""
    a = pd.to_numeric(pair.str[:2], errors="coerce")
    b = pd.to_numeric(pair.str[6:8], errors="coerce")
    ids = pd.concat([a, b], axis=1)
    # Stata min/max ignore missing operands.
    return pd.MultiIndex.from_arrays([ids.min(axis=1), ids.max(axis=1)]).factorize()[0]


def crosswalk_segment(a, b):
    ids = np.sort(np.column_stack([a, b]), axis=1)
    if not np.isfinite(ids).all():
        raise ValueError("Missing state in crosswalk")
    return pd.MultiIndex.from_arrays(ids.T).factorize()[0]


def historical_sample(data):
    # Each variable has its own complete observations for the first areg.
    # Keep double precision through both projection stages.
    data = data.copy()
    values = data[VARIABLES].astype(float)
    within = values - values.groupby(data.county).transform("mean")
    data[VARIABLES] = within
    return data.loc[data.nonmissing_rest_both == 66].copy()


def historical_fit(sample, outcome, controls, effects, segment=None, treatment=WAGE):
    """Authors' sequential linear projections and Stata's areg CR1 df convention."""
    columns = [outcome, treatment, *controls]
    d = sample.dropna(subset=columns)
    v = d[columns]
    by = [d[k] for k in effects]
    group = v.groupby(by)
    residual = (v - group.transform("mean")).to_numpy(dtype=float)
    y, x = residual[:, 0], residual[:, 1:]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    # Validate the coefficient with this repository's estimator (ordinary native
    # FWL SE does not include clustering or absorbed indicator degrees of freedom).
    native = fwl_estimate(y, x[:, 0], x[:, 1:], add_intercept=False, method="ols")
    np.testing.assert_allclose(native.theta, beta[0], rtol=0, atol=1e-9)
    score = (x @ np.linalg.inv(x.T @ x)[:, 0]) * (y - x @ beta)
    n = len(y)
    if segment is None:
        se, _ = panel_standard_error(n * score, d.state_fips, d.period, "unit_cluster")
        groups = {}
    else:
        se, _ = panel_standard_error(n * score, d.state_fips, d[segment], "two_way_cluster")
        groups = {"segments": d[segment].nunique()}
    df_factor = (n - 1) / (n - x.shape[1] - group.ngroups)
    return {
        "estimate": float(beta[0]),
        "se": se * np.sqrt(df_factor),
        "n": n,
        "states": d.state_fips.nunique(),
        **groups,
    }


def published_pretrend(raw):
    """DLR Table 3 employment contrast, lead-four coefficient conditional on lead twelve."""
    paired = "pair_id" in raw
    panel_id = "pair_id_county" if paired else "countyreal"
    panel = raw.dropna(subset=[panel_id, "period"]).copy()
    if panel.duplicated([panel_id, "period"]).any():
        raise ValueError("Ambiguous county-quarter identifier in lead matching")
    wages = panel.set_index([panel_id, "period"])[WAGE]
    for lead in (4, 12):
        # Look up actual event time in the wider wage source before sample restriction.
        key = pd.MultiIndex.from_arrays([panel[panel_id], panel.period + lead])
        panel[f"lead{lead}"] = wages.reindex(key).to_numpy()
    panel = panel.loc[(panel.nonmissing_rest_both == 66) & panel.period.between(25, 90)].copy()
    cols = [EMP, WAGE, "lnpop", "lead4", "lead12"]
    values = panel[cols].astype(float)
    panel[cols] = values - values.groupby(panel.countyreal).transform("mean")
    panel["legacy"] = legacy_segment(panel.pair_id) if paired else 0
    fit = historical_fit(
        panel,
        EMP,
        ["lead12", WAGE, "lnpop"],
        ["pair_id", "period"] if paired else ["period"],
        "legacy" if paired else None,
        treatment="lead4",
    )
    printed = [0.041, 0.134] if paired else [-0.124, 0.070]
    np.testing.assert_allclose([fit["estimate"], fit["se"]], printed, atol=0.0005, rtol=0)
    return fit


def historical_results(national, border, zone, crosswalk):
    national = historical_sample(national)
    border = historical_sample(border)
    zone = historical_sample(zone).merge(crosswalk, on="pair_id", validate="many_to_one")
    border["legacy"] = legacy_segment(border.pair_id)
    zone["legacy"] = legacy_segment(zone.pair_id)
    zone["actual"] = crosswalk_segment(zone.state_0, zone.state_1)
    complete = border.loc[border.groupby("pair_id").county.transform("nunique") == 2]
    cz = zone.loc[zone.pairtimes == 2]
    pair_period = ["pair_id", "period"]
    controls = ["lnpop", "lnemp_TOT"]
    results = {
        "national_quarter": historical_fit(national, EMP, ["lnpop"], ["period"]),
        "border_quarter": historical_fit(border, EMP, ["lnpop"], ["period"], "legacy"),
        "border_pair_quarter": historical_fit(border, EMP, ["lnpop"], pair_period, "legacy"),
        "border_private": historical_fit(border, EMP, controls, pair_period, "legacy"),
        "border_complete_private": historical_fit(complete, EMP, controls, pair_period, "legacy"),
        "border_complete_earnings": historical_fit(complete, PAY, [], pair_period, "legacy"),
        "cz_complete": historical_fit(cz, EMP, ["lnpop"], pair_period, "legacy"),
        "cz_private": historical_fit(cz, EMP, controls, pair_period, "legacy"),
        "cz_geographic_clusters": historical_fit(cz, EMP, controls, pair_period, "actual"),
        "cz_earnings": historical_fit(cz, PAY, ["lnAWW_TOT"], pair_period, "legacy"),
        "cz_earnings_geographic": historical_fit(cz, PAY, ["lnAWW_TOT"], pair_period, "actual"),
    }
    # Round-off intervals in the two source papers. The later paper restricts
    # to complete pairs before dropping missing controls (affects the SE).
    for key, target in {
        "national_quarter": -0.211,
        "border_quarter": -0.137,
        "border_pair_quarter": 0.057,
        "border_private": 0.016,
        "border_complete_private": 0.016,
        "border_complete_earnings": 0.200,
        "cz_complete": -0.128,
        "cz_private": -0.141,
        "cz_earnings": 0.208,
    }.items():
        np.testing.assert_allclose(
            results[key]["estimate"], target, rtol=0, atol=0.0005, err_msg=key
        )
    for key, target in {
        "border_complete_private": 0.076,
        "border_complete_earnings": 0.050,
        "cz_complete": 0.070,
        "cz_private": 0.070,
        "cz_earnings": 0.064,
    }.items():
        np.testing.assert_allclose(results[key]["se"], target, rtol=0, atol=0.0005, err_msg=key)
    return results


def pair_changes(raw):
    """Quarterly first differences of high-state minus low-state log-level contrasts."""
    data = raw.loc[
        raw.nonmissing_rest_both == 66,
        ["pair_id", "period", "county", "state_fips", WAGE, EMP, PAY, "lnpop"],
    ]
    counts = data.groupby("pair_id").county.nunique()
    d = data.loc[data.pair_id.isin(counts.index[counts == 2])].sort_values(
        ["pair_id", "period", "state_fips"]
    )
    groups = d.groupby(["pair_id", "period"])
    if not ((groups.size() == 2) & (groups.state_fips.nunique() == 2)).all():
        raise ValueError("Each complete pair/quarter must contain two distinct states")
    levels = [WAGE, EMP, PAY, "lnpop"]
    # A stable orientation prevents the sign of a treatment contrast depending on input row order.
    high, low = (
        groups.tail(1).set_index(["pair_id", "period"]),
        groups.head(1).set_index(["pair_id", "period"]),
    )
    contrast = (high[levels] - low[levels]).sort_index()
    periods = contrast.index.get_level_values("period").to_series(index=contrast.index)
    if not periods.groupby(level=0).diff().dropna().eq(1).all():
        raise ValueError("The retained pair panels contain a time gap")
    change = contrast.groupby(level=0).diff()
    lag = change.groupby(level=0)
    features = lag.shift(2).add_prefix("lag2_")
    frame = (
        features.assign(
            wage=change[WAGE],
            employment=change[EMP],
            earnings=change[PAY],
            pre_employment=lag[EMP].shift(1),
        )
        .dropna()
        .reset_index()
    )
    for quarter in range(1, 4):
        frame[f"season{quarter}"] = (frame.period % 4 == quarter).astype(float)
    return frame


def forward_folds(periods):
    """Four expanding test blocks with a two-quarter gap before each validation start."""
    blocks = np.array_split(np.unique(periods), 5)
    if min(map(len, blocks)) < 3:
        raise ValueError("At least fifteen quarters are required")
    for block in blocks[1:]:
        test = np.isin(periods, block)
        yield np.asarray(periods < block[0] - 2), test


def short_run(frame):
    features = [c for c in frame if c.startswith(("lag2_", "season"))]
    x = frame[features].to_numpy()
    names = ["wage", "employment", "earnings", "pre_employment"]
    targets = frame[names].to_numpy(dtype=float)
    predictions = {key: np.full_like(targets, np.nan) for key in ("linear", "forest")}
    design = []
    for train, test in forward_folds(frame.period.to_numpy()):
        design.append(
            {
                "train_end": int(frame.period[train].max()),
                "test_start": int(frame.period[test].min()),
                "test_end": int(frame.period[test].max()),
            }
        )
        # Independent models for each nuisance. Training never includes a row
        # from the same calendar quarter, even if a county occurs in multiple pairs.
        for j in range(len(names)):
            for key, model in [
                ("linear", LinearRegression()),
                (
                    "forest",
                    RandomForestRegressor(
                        n_estimators=32, min_samples_leaf=40, max_depth=5, random_state=37, n_jobs=1
                    ),
                ),
            ]:
                predictions[key][test, j] = model.fit(x[train], targets[train, j]).predict(x[test])
    test = np.isfinite(predictions["linear"][:, 0])
    result = {
        "n": int(test.sum()),
        "pairs": frame.loc[test, "pair_id"].nunique(),
        "quarters": frame.loc[test, "period"].nunique(),
        "folds": design,
        "observed_wage_changes": int((np.abs(targets[test, 0]) > 1e-6).sum()),
    }
    for key, pred in predictions.items():
        r = targets[test] - pred[test]
        d = r[:, 0]
        information = np.mean(d**2)
        if information <= 0 or not np.isfinite(r).all():
            raise ValueError("Held-out score lacks residual treatment variation")
        fits = {}
        for j, outcome in enumerate(names[1:], start=1):
            theta = d @ r[:, j] / (d @ d)
            influence = d * (r[:, j] - theta * d) / information
            se, bandwidth = panel_standard_error(
                influence, frame.pair_id[test], frame.period[test], "date_hac"
            )
            fits[outcome] = {
                "estimate": float(theta),
                "date_hac_se": se,
                "bandwidth": bandwidth,
                "prediction_rmse": float(np.sqrt(np.mean(r[:, j] ** 2))),
            }
        result[key] = {"residual_wage_rms": float(np.sqrt(information)), **fits}
    return result


def run(cache):
    files = {name: fetch(url, cache, name, digest) for name, (url, digest) in SOURCES.items()}
    with ZipFile(files["dlr-minimumwage.zip"]) as archive:

        def read_dlr(name, cols):
            with archive.open(f"minwage_dubeRESTAT/QCEWindustry_minwage_{name}.dta") as file:
                return pd.read_stata(file, columns=cols, convert_categoricals=False)

        national = read_dlr("all", IDS + VARIABLES + ["countyreal"])
        border = read_dlr("contig", IDS + VARIABLES + ["pair_id", "pair_id_county", "countyreal"])
    zone = pd.read_stata(
        files["jnr-cz-panel.dta"], columns=IDS + VARIABLES + ["pair_id"], convert_categoricals=False
    )
    lookup = pd.read_stata(files["jnr-cz-pairs.dta"], convert_categoricals=False)
    return {
        "card_krueger": store_study(files["njmin.zip"]),
        "dlr_jnr": historical_results(national, border, zone, lookup),
        "dlr_table3_pretrends": {
            "national": published_pretrend(national),
            "border": published_pretrend(border),
        },
        "short_run": {
            "border": short_run(pair_changes(border)),
            "commuting_zone": short_run(pair_changes(zone)),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.getenv("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache" / "dml-ts" / "external")
        ),
    )
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
