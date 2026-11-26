"""Public interrupted-time-series and cumulative local-projection benchmarks."""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

SICILY_URL = (
    "https://raw.githubusercontent.com/gasparrini/2017_lopezbernal_IJE_codedata/"
    "85456263850d03ae39439e032929347c40ff3a26/sicily.csv"
)
SICILY_HASH = "63a867b3fc8748425ff27a28202c19b4c2741e69a1ab753ad8e77c4e56457dda"
RAMEY_URL = "https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_replication_codes.zip"
RAMEY_HASH = "e268f8d4abc997227401d5b5f218d0cfa3c527cb9091f6e715baa5a7a6391d07"


def sicily(cache_dir):
    data = pd.read_csv(fetch(SICILY_URL, cache_dir, "sicily.csv", SICILY_HASH))
    month = pd.DatetimeIndex(
        pd.to_datetime({"year": data.year, "month": data.month, "day": 1})
    ).to_period("M")
    expected = pd.period_range("2002-01", "2006-11", freq="M")
    np.testing.assert_array_equal(month, expected)
    np.testing.assert_array_equal(data.time, np.arange(1, 60))
    np.testing.assert_array_equal(data.smokban, (month >= pd.Period("2005-01", "M")).astype(int))
    np.testing.assert_equal(int(data.aces.sum()), 48914)
    controls = pd.DataFrame({"intercept": 1.0, "ban": data.smokban, "time": data.time})
    for harmonic in (1, 2):
        angle = 2 * harmonic * np.pi * data.month / 12
        controls[f"sin{harmonic}"] = np.sin(angle)
        controls[f"cos{harmonic}"] = np.cos(angle)
    fit = sm.GLM(data.aces, controls, offset=np.log(data.stdpop), family=sm.families.Poisson()).fit(
        scale="X2"
    )
    rr = np.exp(fit.params["ban"])
    lower, upper = np.exp(fit.conf_int().loc["ban"])
    # Figure 4 reports the ratio and the two confidence limits to three decimal places.
    np.testing.assert_allclose([rr, lower, upper], [0.885, 0.839, 0.933], atol=0.0005, rtol=0)
    folds = []
    for train, test in TimeSeriesSplit(n_splits=4).split(data):
        folds.append(
            {
                "train_months": len(train),
                "train_treated": int(data.smokban.iloc[train].sum()),
                "test_months": len(test),
                "test_treated": int(data.smokban.iloc[test].sum()),
            }
        )
    np.testing.assert_equal([fold["train_treated"] for fold in folds], [0, 0, 1, 12])
    np.testing.assert_equal([fold["test_treated"] for fold in folds], [0, 1, 11, 11])
    return {
        "forward_support": folds,
        "months": len(data),
        "events": int(data.aces.sum()),
        "rate_ratio": float(rr),
        "lower": float(lower),
        "upper": float(upper),
    }


def cumulative_multiplier(series, horizon):
    """Project the sum from t through t+h, instrumenting spending with news at t."""
    controls = pd.concat(
        [series.shift(lag).add_suffix(f"_lag{lag}") for lag in range(1, 5)], axis=1
    )
    totals = series[["y", "g"]].rolling(horizon + 1).sum().shift(-horizon)
    joint = pd.concat([controls, series.news, totals], axis=1).dropna()
    design = sm.add_constant(joint[controls.columns]).to_numpy()
    news = joint.news.to_numpy()
    residual_news = news - design @ np.linalg.lstsq(design, news, rcond=None)[0]
    theta = (residual_news @ joint.y) / (residual_news @ joint.g)
    return {"observations": len(joint), "multiplier": float(theta)}


def archived_multipliers(package):
    """Read the comparison cells on Slack-newsy in the pinned authors' output workbook."""
    member = "Ramey_Zubairy_replication_codes/Multiplier-Standard-Errors.xlsx"
    with ZipFile(BytesIO(package.read(member))) as workbook:
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet2.xml"))
    values = {cell.attrib["r"]: cell.findtext("{*}v") for cell in sheet.iterfind(".//{*}c")}
    result = {}
    for name, row, horizon in [("two_year", 11, 7), ("four_year", 19, 15)]:
        np.testing.assert_equal(int(values[f"A{row}"]), horizon)
        result[name] = {"multiplier": float(values[f"B{row}"]), "se": float(values[f"C{row}"])}
    return result


def ramey(cache_dir):
    archive = fetch(RAMEY_URL, cache_dir, "ramey-zubairy.zip", RAMEY_HASH)
    with (
        ZipFile(archive) as source,
        source.open("Ramey_Zubairy_replication_codes/rzdatnew.csv") as member,
    ):
        raw = pd.read_csv(member).query("quarter >= 1889")
        workbook = archived_multipliers(source)
    np.testing.assert_array_equal(raw.quarter, 1889 + np.arange(508) / 4)
    series = pd.DataFrame(
        {
            "y": raw.rgdp / raw.rgdp_pott6,
            "g": raw.ngov / (raw.pgdp * raw.rgdp_pott6),
            "news": raw.news / (raw.pgdp * raw.rgdp_pott6).shift(1),
        }
    )
    two, four, extra = [cumulative_multiplier(series, h) for h in [7, 15, 8]]
    np.testing.assert_equal([two["observations"], four["observations"]], [493, 485])
    np.testing.assert_allclose(
        [two["multiplier"], four["multiplier"]], [0.66, 0.71], atol=0.005, rtol=0
    )
    if round(extra["multiplier"], 2) != 0.67:
        raise AssertionError("Off-by-one control failed to distinguish eight and nine quarters")
    return {
        "quarters": len(raw),
        "two_year": two,
        "four_year": four,
        "nine_quarter_control": extra,
        "archived_workbook": workbook,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["sicily", "ramey"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"sicily": sicily, "ramey": ramey}[args.study](args.cache_dir), indent=2))
