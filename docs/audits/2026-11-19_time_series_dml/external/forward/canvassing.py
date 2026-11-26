"""Test a public 1999 canvassing effect forecast against six randomized 2001 elections."""

import argparse
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

EARLY_PDF = "https://www.uvm.edu/~dguber/POLS234/articles/gerber.pdf"
EARLY_HASH = "2db5cf4b815b519ea9ca5ad8deaea716eb0506ad586901e5b0fac955f7c5c5d9"
YALE = "https://dataverse.yale.edu/api/access/datafile"
FILES = {
    27228: "3c01c9669077aec486a56944dcf79e0b83040eb3b63069e84452ea11e3fee610",
    27483: "2da76eb3687e100a748dbc92f69fa858d1168b63e189e6736ba0cb1cb86a515e",
}
# Green, Gerber and Nickerson (2003), Table 2. Columns 1 and 2 and SE of column 2, pp.
PUBLISHED = {
    "Bridgeport": (-0.3, 4.0, 1.5),
    "Columbus": (-0.4, 1.4, 1.1),
    "Detroit": (0.2, 2.4, 1.4),
    "Minneapolis": (-0.4, 1.9, 1.6),
    "Raleigh": (0.4, 0.1, 1.4),
    "St Paul": (3.2, 4.6, 2.1),
    "All sites": (0.4, 2.1, 0.6),
}


def published_forecast(pdf_text):
    """Extract the publicly available difference in assignment groups, PNAS Table 1."""
    match = re.search(
        r"Percentage voting, %\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+Number of persons\s+23,921\s+4,509",
        pdf_text,
    )
    if match is None:
        raise ValueError("Published arm shares and denominators missing in Table 1")
    control, treated = map(float, match.groups())
    return treated - control


def corrected_early(frame):
    """Re-estimate using the later archive's corrected assignment and three factorial controls."""
    use = frame.loc[frame.vote98.isin([0, 1])].copy()
    controls = pd.get_dummies(
        use[["persons", "mailings", "phongotv", "placebo"]].astype("category"),
        drop_first=True,
        dtype=float,
    )
    design = sm.add_constant(pd.concat([use.persngrp, controls], axis=1))
    fit = sm.OLS(use.vote98, design).fit(cov_type="cluster", cov_kwds={"groups": use.id1})
    native = fwl_estimate(
        use.vote98.to_numpy(float), use.persngrp.to_numpy(float), controls.to_numpy(), method="ols"
    )
    np.testing.assert_allclose(native.theta, fit.params["persngrp"], atol=1e-12)
    return {
        "estimate": float(100 * fit.params["persngrp"]),
        "se": float(100 * fit.bse["persngrp"]),
        "rows": len(use),
        "households": int(use.id1.nunique()),
        "missing_turnout": len(frame) - len(use),
        "unadjusted": float(
            100
            * (
                use.loc[use.persngrp.eq(1), "vote98"].mean()
                - use.loc[use.persngrp.eq(0), "vote98"].mean()
            )
        ),
    }


def site_contrast(frame, outcome):
    """Regress turnout on randomized assignment and walk lists using within-list residuals."""
    use = frame[["city", "turf", "treatmen", outcome]].dropna()
    groups = use.groupby(["city", "turf"])
    within = use[["treatmen", outcome]] - groups[["treatmen", outcome]].transform("mean")
    treatment, y = within.treatmen.to_numpy(), within[outcome].to_numpy()
    information = treatment @ treatment
    if information <= 0:
        raise ValueError("No within-list variation in treatment")
    coefficient = float(treatment @ y / information)
    residual = y - treatment * coefficient
    df = len(use) - groups.ngroups - 1
    if df <= 0:
        raise ValueError("Insufficient residual degrees of freedom")
    # The paper uses homoskedastic SE. Inference for new forecast errors uses HC1.
    published_se = np.sqrt((residual @ residual) / df / information)
    robust_se = np.linalg.norm(treatment * residual) / information * np.sqrt(len(use) / df)
    return {
        "estimate": 100 * coefficient,
        "paper_se": 100 * published_se,
        "hc1_se": 100 * robust_se,
        "rows": len(use),
        "walk_lists": groups.ngroups,
        "supported_lists": int((groups.treatmen.nunique() == 2).sum()),
    }


def prediction_error(prediction, observed, early_se=0):
    """Two-sample error; early_se=0 treats an already-issued prediction as fixed."""
    delta = prediction - observed["estimate"]
    width = 1.959963984540054 * np.hypot(observed["hc1_se"], early_se)
    return {"estimate": float(delta), "ci95": [float(delta - width), float(delta + width)]}


def run(cache):
    pdf = fetch(EARLY_PDF, cache, "canvass-pnas-1999.pdf", EARLY_HASH)
    pdf_text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    public = published_forecast(pdf_text)
    np.testing.assert_allclose(public, 2.25, atol=1e-10)
    frames = {
        i: pd.read_stata(
            fetch(f"{YALE}/{i}?format=original", cache, f"canvass-{i}", checksum),
            convert_categoricals=False,
        )
        for i, checksum in FILES.items()
    }
    corrected = corrected_early(frames[27483])
    data = frames[27228]
    data["city"] = data.city.str.title()
    if len(data) != 18933 or data.duplicated(["city", "family"]).any():
        raise ValueError("Expected 18,933 distinct city–household records")
    groups = dict(tuple(data.groupby("city")))
    if set(groups) != set(PUBLISHED) - {"All sites"}:
        raise ValueError("Unexpected city labels")
    groups["All sites"] = data
    results = {}
    for name, group in groups.items():
        late = site_contrast(group, "voted01")
        placebo = site_contrast(group, "voted00")["estimate"]
        np.testing.assert_allclose(
            [placebo, late["estimate"], late["paper_se"]], PUBLISHED[name], atol=0.0501, rtol=0
        )
        results[name] = {
            "later": late,
            "pretreatment_placebo": placebo,
            "public_minus_later": prediction_error(public, late),
            "corrected_minus_later": prediction_error(corrected["estimate"], late, corrected["se"]),
            "zero_minus_later": prediction_error(0, late),
        }
    return {"public_forecast_pp": public, "corrected_forecast_pp": corrected, "cities_pp": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
