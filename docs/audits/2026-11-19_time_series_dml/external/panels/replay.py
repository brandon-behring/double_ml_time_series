"""Check published fixed-effects, clustered OLS and IV estimates on the authors' archives."""

import argparse
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from threadpoolctl import threadpool_limits

# Direct execution must exercise the checkout even if another dml_ts wheel is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from dml_ts.dml.fwl import fwl_estimate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

WOLFERS_URL = "https://users.nber.org/~jwolfers/data/DivorceDataAppendix(Stata9).zip"
WOLFERS_SHA = "7fc6fdce65688e6d522e224a547e62c321ce6bdfe77adbf64f7df344653b4bd4"
DAHL_URL = "https://econweb.ucsd.edu/~gdahl/movies-code/movies.dta.zip"
DAHL_SHA = "f57e7320c4a2e85523267fb412f890ef48c2a82c7858758f40d0d9f2122be81b"
AUDIENCES = ["rjd810viol", "rjd57viol", "rjd04viol"]
INSTRUMENTS = ["rh1d810viol", "rh1d57viol", "rh1d04viol"]
WEATHER = "maxa maxb maxc heatindexb heatindexc heatindexd mina minb minc wdspb wdspc anyrain anysnow".split()


def printed(actual, published, digits, label):
    """The source reports decimal-rounded values with no additional numerical tolerance."""
    np.testing.assert_allclose(actual, published, rtol=0, atol=0.5 * 10**-digits, err_msg=label)


def wolfers(cache_dir):
    path = fetch(WOLFERS_URL, cache_dir, "wolfers-divorce-stata9.zip", WOLFERS_SHA)
    with ZipFile(path) as archive:
        (member,) = [
            name for name in archive.namelist() if name.endswith("/Divorce-Wolfers-AER.dta")
        ]
        with archive.open(member) as source:
            data = pd.read_stata(source, convert_categoricals=False)
    fields = ["st", "year", "div_rate", "stpop", "years_unilateral", "time", "timesq"]
    data = data.loc[data.year.between(1956, 1988), fields].dropna()
    assert (len(data), data.st.nunique(), data.year.nunique()) == (1631, 51, 33)
    assert not data.duplicated(["st", "year"]).any()
    assert (data.stpop > 0).all()
    assert sorted(data.years_unilateral.unique()) == [-99, 1, 3, 5, 7, 9, 11, 13, 15]

    event = pd.get_dummies(data.years_unilateral, drop_first=True, dtype=float).to_numpy()
    state = pd.get_dummies(data.st, drop_first=True, dtype=float).to_numpy()
    year = pd.get_dummies(data.year, drop_first=True, dtype=float).to_numpy()
    weights = data.stpop.to_numpy(dtype=float)
    y = data.div_rate.to_numpy(dtype=float)
    design = np.column_stack([event, np.ones(len(data)), state, year])

    # Table 2, columns (1)–(3), in event-time order (first two years through 15+).
    published_beta = [
        [0.27, 0.21, 0.16, 0.16, -0.12, -0.32, -0.46, -0.51],
        [0.34, 0.32, 0.30, 0.32, 0.08, -0.10, -0.20, -0.21],
        [0.30, 0.29, 0.29, 0.35, 0.16, 0.05, 0.03, 0.25],
    ]
    published_se = [
        [0.08, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
        [0.06, 0.07, 0.08, 0.08, 0.09, 0.10, 0.11, 0.12],
        [0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.17, 0.20],
    ]
    fits = []
    for column, trend in enumerate([None, "time", "timesq"]):
        if trend:
            design = np.column_stack([design, state * data[[trend]].to_numpy()])
        model = sm.WLS(y, design, weights=weights).fit()
        estimates, ses = model.params[:8], model.bse[:8]
        printed(estimates, published_beta[column], 2, f"Wolfers Table 2({column + 1}) estimates")
        printed(ses, published_se[column], 2, f"Wolfers Table 2({column + 1}) SEs")
        printed(model.rsquared_adj, [0.9310, 0.9732, 0.9822][column], 4, "adjusted R-squared")
        # Check the native estimator on the same weighted design, including its intercept.
        scale = np.sqrt(weights)
        weighted_design = design * scale[:, None]
        for index in (0, 7):
            native = fwl_estimate(
                y * scale,
                weighted_design[:, index],
                np.delete(weighted_design, index, axis=1),
                method="ols",
                add_intercept=False,
            )
            np.testing.assert_allclose(
                [native.theta, native.se],
                [estimates[index], ses[index]],
                rtol=0,
                atol=1e-8,
                err_msg=f"native FWL, specification {column + 1}, event bin {index + 1}",
            )
        fits.append(
            {"estimate": estimates.tolist(), "se": ses.tolist(), "adj_r2": model.rsquared_adj}
        )
    return {"n": len(data), "states": data.st.nunique(), "table_2": fits}


def dahl_fit(sample, controls, iv, first_stage=False):
    """Partial out calendar controls and calculate weekend-clustered CR1 uncertainty."""
    y = sample.lnassault.to_numpy(dtype=float)
    x = sample[AUDIENCES].to_numpy(dtype=float)
    z = sample[INSTRUMENTS].to_numpy(dtype=float) if iv else x
    c = np.column_stack([np.ones(len(sample)), sample[controls].to_numpy(dtype=float)])
    # Several complete indicator sets overlap. Retain only their numerical column space.
    q, _, _ = qr(c, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(c)
    q = q[:, :rank]
    y, x, z = [a - q @ (q.T @ a) for a in (y, x, z)]
    n = len(y)
    groups = sample.wkdgroup.to_numpy()
    n_groups = sample.wkdgroup.nunique()
    df_factor = n_groups / (n_groups - 1) * (n - 1) / (n - rank - x.shape[1])

    def fit(outcome, regressors, instruments):
        moment = instruments.T @ regressors
        beta = np.linalg.solve(moment, instruments.T @ outcome)
        score = instruments * (outcome - regressors @ beta)[:, None]
        sums = pd.DataFrame(score).groupby(groups).sum().to_numpy()
        influence = np.linalg.solve(moment, sums.T)
        se = np.sqrt(np.sum(influence**2, axis=1) * df_factor)
        return beta, se

    estimates, ses = fit(y, x, z)
    result = {"n": n, "weekends": n_groups, "estimate": estimates.tolist(), "se": ses.tolist()}
    if first_stage:
        stages = [fit(audience, z, z) for audience in x.T]
        result["first_stage"] = {
            "estimate": [beta.tolist() for beta, _ in stages],
            "se": [se.tolist() for _, se in stages],
        }
    if not iv:
        native = fwl_estimate(
            sample.lnassault.to_numpy(dtype=float),
            sample[AUDIENCES[0]].to_numpy(dtype=float),
            sample[AUDIENCES[1:] + controls].to_numpy(dtype=float),
            method="ols",
        )
        np.testing.assert_allclose(native.theta, estimates[0], rtol=0, atol=1e-10)
    return result


def dahl(cache_dir):
    path = fetch(DAHL_URL, cache_dir, "dahl-dellavigna-movies.zip", DAHL_SHA)
    with ZipFile(path) as archive:
        with archive.open("fulliblockday.dta") as member:
            daily = pd.read_stata(member, convert_categoricals=False)
        with archive.open("fulliblock.dta") as member:
            windows = pd.read_stata(member, convert_categoricals=False)
    cols = ["lnassault", "wkdgroup", *AUDIENCES]
    calendar = [c for c in daily if c.startswith(("hh", "hm", "hp", "dd", "mm", "yy", "nn"))]
    complete = calendar + WEATHER
    table_2 = {}
    for col, controls, beta, se in [
        (
            1,
            [c for c in daily if c.startswith("yy")],
            [0.0324, 0.0246, 0.0082],
            [0.0053, 0.0030, 0.0029],
        ),
        (6, complete, [-0.0091, -0.0079, -0.0035], [0.0026, 0.0022, 0.0024]),
        (7, complete, [-0.0106, -0.0102, -0.0050], [0.0031, 0.0028, 0.0029]),
    ]:
        fields = cols + controls + (INSTRUMENTS if col == 7 else [])
        sample = daily.loc[daily.wkd == 1, fields].dropna()
        result = dahl_fit(sample, controls, iv=col == 7)
        assert (result["n"], result["weekends"]) == (1563, 522)
        printed(result["estimate"], beta, 4, f"Dahl Table II({col})")
        printed(result["se"], se, 4, f"Dahl Table II({col}) SE")
        table_2[col] = result

    # Table III A columns are time blocks; entries within each block are the three audiences.
    published_beta = [
        [-0.0050, -0.0106, -0.0033],
        [-0.0030, -0.0001, 0.0016],
        [-0.0130, -0.0109, -0.0063],
        [-0.0192, -0.0205, -0.0060],
    ]
    published_se = [
        [0.0066, 0.0060, 0.0060],
        [0.0050, 0.0045, 0.0046],
        [0.0049, 0.0040, 0.0043],
        [0.0060, 0.0052, 0.0054],
    ]
    table_3 = {}
    for block in (1, 2, 3, 4):
        sample = windows.loc[
            (windows.wkd == 1) & (windows.iblock == block), cols + complete + INSTRUMENTS
        ].dropna()
        result = dahl_fit(sample, complete, iv=True, first_stage=block == 1)
        counts = (1562, 521) if block == 4 else (1563, 522)
        assert (result["n"], result["weekends"]) == counts
        printed(result["estimate"], published_beta[block - 1], 4, f"Dahl Table III A({block}) IV")
        printed(result["se"], published_se[block - 1], 4, f"Dahl Table III A({block}) SE")
        table_3[block] = result
    stages = table_3[1]["first_stage"]
    printed(
        stages["estimate"],
        [
            [0.9145, -0.0399, -0.0480],
            [-0.1431, 0.8532, -0.1363],
            [-0.1694, -0.1817, 0.8138],
        ],
        4,
        "Dahl Table III B first-stage coefficients",
    )
    printed(
        stages["se"],
        [
            [0.0196, 0.0101, 0.0097],
            [0.0210, 0.0255, 0.0195],
            [0.0281, 0.0296, 0.0309],
        ],
        4,
        "Dahl Table III B first-stage SEs",
    )
    return {"table_2": table_2, "table_3": table_3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["wolfers", "dahl"])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get("DML_TS_BENCHMARK_CACHE", "~/.cache/dml-ts/benchmarks")
        ).expanduser(),
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        result = {"wolfers": wolfers, "dahl": dahl}[args.study](args.cache_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
