"""Replay Ferraro–Price Table 2A and Chetty–Looney–Kroft Table 6(1)."""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Direct execution must exercise the checkout even if another dml_ts wheel is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from dml_ts.dml.double_ml import double_ml  # noqa: E402
from dml_ts.dml.fwl import fwl_estimate  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "water_moments.json"
WATER_URL = "https://dataverse.harvard.edu/api/access/datafile/2446186?format=original"
WATER_SHA = "7e42f75dcbd6ff42bc463d9e2039a71d85d1a0683181a5c02910c02055ad908e"
BEER_URL = "https://rajchetty.com/wp-content/uploads/2021/04/CLK_data_code_AER.zip"
BEER_SHA = "b96c51c96f4a9489fe55cf4b6d377e76018b4019822f6c7b6127ee2cd1c3270e"
BEER_STATA_SHA = "86b1461f17a0b3ea78c7713d386694c4f33239595f66f46e9a440a1b2172b0fa"
ARMS = ["tips", "weak", "peer"]
DESIGN = ["intercept", *ARMS, "baseline2006", "spring2007"]
WATER_COEFS = [-0.24, -0.99, -1.74]
WATER_SES = [0.19, 0.17, 0.17]
WATER_COUNTS = {"tips": 11675, "weak": 11675, "peer": 11676, "control": 71643}


def water_sample(path):
    raw = pd.read_stata(path, convert_categoricals=False)
    if raw.shape != (158731, 33):
        raise ValueError(f"Unexpected raw Cobb County file dimensions {raw.shape}")
    trial = raw.loc[
        raw.group.notna() & raw.unusable.ne(1) & ~raw.prem.isin([105321, 129672, 148705])
    ].copy()
    for name, months in {
        "volume": ["jul07_3x", "aug07_3x", "sep07_3x", "oct07_3x"],
        "baseline2006": ["jun06", "jul06", "aug06", "sep06", "oct06", "nov06"],
        "spring2007": ["apr07_3x", "may07_3x"],
    }.items():
        trial[name] = trial[months].sum(axis=1, min_count=len(months))
    trial = trial.dropna(subset=["volume", "baseline2006", "spring2007"]).copy()
    trial["arm"] = trial.group.map({1: "weak", 2: "peer", 3: "tips", 4: "control"})
    if not trial.prem.is_unique or trial.arm.value_counts().to_dict() != WATER_COUNTS:
        raise ValueError("Unexpected trial units or treatment mapping")
    trial["intercept"] = 1.0
    for arm in ARMS:
        trial[arm] = trial.arm.eq(arm)
    return trial


def water_moments(trial):
    """Sufficient sums for OLS and HC1, without releasing individual households."""
    values = trial[[*DESIGN, "volume"]].to_numpy(dtype=float)
    x, y = values[:, :-1], values[:, -1]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    scores = x * (y - x @ beta)[:, None]
    return {
        "columns": [*DESIGN, "volume"],
        "n": len(trial),
        "arm_counts": trial.arm.value_counts().to_dict(),
        "crossproducts": (values.T @ values).tolist(),
        "score_crossproducts": (scores.T @ scores).tolist(),
    }


def fit_water_moments(moments, peer_scale=1.0):
    """Factor the joint Gram matrix to preserve FWL slopes in seven rows."""
    if moments["columns"] != [*DESIGN, "volume"] or moments["arm_counts"] != WATER_COUNTS:
        raise ValueError("Unrecognized water fixture design")
    gram = np.asarray(moments["crossproducts"])
    n = moments["n"]
    k = len(DESIGN)
    if gram[0, 0] != n or n != sum(WATER_COUNTS.values()):
        raise ValueError("Unexpected number of premises")
    scale = np.ones(k)
    scale[DESIGN.index("peer")] = peer_scale
    factor = np.linalg.cholesky(gram).T
    y = factor[:, -1]
    x = factor[:, :-1] * scale
    inverse = np.linalg.inv(x.T @ x)
    meat = np.asarray(moments["score_crossproducts"]) * np.outer(scale, scale)
    hc1 = np.sqrt(np.diag(inverse @ meat @ inverse.T) * n / (n - k))
    fitted = {}
    for arm in ARMS:
        j = DESIGN.index(arm)
        # Column zero already represents the intercept in the factored design.
        fit = fwl_estimate(y, x[:, j], np.delete(x, j, axis=1), add_intercept=False, method="ols")
        # Compression preserves RSS and the residualized treatment norm. Restore
        # the denominator n-k used by the full-sample classical variance.
        full_classical_se = fit.se * np.sqrt((len(y) - k) / (n - k))
        fitted[arm] = {"slope": fit.theta, "HC1_SE": hc1[j], "homoskedastic_SE": full_classical_se}
    return fitted


def replay_water(path, fixture_output=None):
    trial = water_sample(path)
    moments = water_moments(trial)
    compact = fit_water_moments(moments)
    y = trial.volume.to_numpy(dtype=float)
    x = trial[DESIGN].to_numpy(dtype=float)
    hc1 = sm.OLS(y, x).fit(cov_type="HC1")
    peer = DESIGN.index("peer")
    native = fwl_estimate(
        y,
        x[:, peer],
        trial[["tips", "weak", "baseline2006", "spring2007"]].to_numpy(dtype=float),
        method="ols",
    )
    np.testing.assert_allclose(native.theta, hc1.params[peer], rtol=1e-9)
    np.testing.assert_allclose(hc1.params[1:4], WATER_COEFS, rtol=0, atol=0.005)
    np.testing.assert_allclose(hc1.bse[1:4], WATER_SES, rtol=0, atol=0.005)
    for j, arm in enumerate(ARMS, start=1):
        np.testing.assert_allclose(
            [compact[arm]["slope"], compact[arm]["HC1_SE"]], [hc1.params[j], hc1.bse[j]]
        )
    np.testing.assert_allclose(compact["peer"]["homoskedastic_SE"], native.se)
    if fixture_output is None:
        saved = json.loads(FIXTURE.read_text())
        for key in ["crossproducts", "score_crossproducts"]:
            np.testing.assert_allclose(saved[key], moments[key], atol=1e-6, rtol=1e-10)
    else:
        fixture_output.write_text(json.dumps(moments, indent=2) + "\n")
    pair = trial.loc[trial.arm.isin(["control", "peer"])]
    dml = double_ml(
        pair.volume.to_numpy(dtype=float),
        pair.peer.to_numpy(dtype=float),
        pair[["baseline2006", "spring2007"]].to_numpy(dtype=float),
        model="ridge",
        n_folds=5,
        random_state=42,
    )
    return {
        "Table 2 model A": {"n": len(trial), "arms": compact},
        "native_FWL_peer": {"slope": native.theta, "homoskedastic_SE": native.se},
        "two_arm_DML_extension": {"n": len(pair), "slope": dml.theta, "IF_SE": dml.se},
    }


def replay_beer(path):
    with ZipFile(path) as archive:
        dta = archive.read("alcohol_data.dta")
    if hashlib.sha256(dta).hexdigest() != BEER_STATA_SHA:
        raise ValueError("Unexpected alcohol_data.dta checksum")
    raw = pd.read_stata(io.BytesIO(dta), convert_categoricals=False)
    rows = raw.loc[~raw.state.isin([15, 54])].sort_values(["state", "year"]).copy()
    if rows.duplicated(["state", "year"]).any():
        raise ValueError("Duplicate state-years")
    for key in ["c_beer", "beer_tax", "salestax", "population"]:
        rows[key] = rows[key].astype(float)
    logs = pd.DataFrame(
        {
            "volume": np.log(rows.c_beer / rows.population),
            "excise": np.log1p(rows.beer_tax / 100),
            "sales": np.log1p(rows.salestax / 100),
            "pop": np.log(rows.population),
        }
    )
    # Stata's d. operator requires consecutive years within the same state.
    diffs = logs.groupby(rows.state).diff()
    sample = (
        rows[["state", "year"]].join(diffs).loc[rows.groupby("state").year.diff().eq(1)].dropna()
    )
    if len(sample) != 1607 or sample.state.nunique() != 49:
        raise ValueError("Unexpected beer sample or cluster count")
    dates = pd.get_dummies(sample.year, prefix="year", drop_first=True, dtype=float)
    x = sm.add_constant(sample[["excise", "sales", "pop"]].join(dates))
    model = sm.OLS(sample.volume, x)
    ols = model.fit()
    cluster = model.fit(
        cov_type="cluster", cov_kwds={"groups": sample.state, "use_correction": True}
    )
    native = fwl_estimate(
        sample.volume.to_numpy(),
        sample.excise.to_numpy(),
        x.drop(columns=["excise", "const"]).to_numpy(),
        method="ols",
    )
    terms = ["excise", "sales"]
    np.testing.assert_allclose([native.theta, native.se], [ols.params.excise, ols.bse.excise])
    np.testing.assert_allclose(ols.params[terms], [-0.88, -0.20], rtol=0, atol=0.005)
    np.testing.assert_allclose(ols.bse[terms], [0.17, 0.30], rtol=0, atol=0.005)
    # Independent CR1 reference for this archive and the full year-dummy design.
    np.testing.assert_allclose(cluster.bse[terms], [0.132105, 0.293508], rtol=0, atol=5e-7)
    return {
        "Table 6 column 1": {
            "state_years": len(sample),
            "states": sample.state.nunique(),
            "slopes": ols.params[terms].to_dict(),
            "conventional_SE": ols.bse[terms].to_dict(),
            "state_CR1_SE": cluster.bse[terms].to_dict(),
            "printed_SE": [0.17, 0.30],
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["water", "beer"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--write-fixture", type=Path, help="write the derived water aggregate JSON")
    args = parser.parse_args()
    if args.write_fixture is not None and args.study != "water":
        parser.error("--write-fixture is available for water only")
    sys.path.insert(0, str(HERE.parent))
    from benchmark_io import fetch

    if args.study == "water":
        data = fetch(WATER_URL, args.cache_dir, "water.dta", WATER_SHA)
        results = replay_water(data, args.write_fixture)
    else:
        data = fetch(BEER_URL, args.cache_dir, "CLK_data_code_AER.zip", BEER_SHA)
        results = replay_beer(data)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
