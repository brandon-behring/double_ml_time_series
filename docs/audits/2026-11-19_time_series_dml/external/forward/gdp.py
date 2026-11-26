"""Evaluate next-quarter GDP forecasts at actual SPF deadlines and release vintages."""

import argparse
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from temporalcv import TimeSeriesCrossValidator, newey_west_se

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from benchmark_io import fetch  # noqa: E402

FED = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
SPF = FED + "survey-of-professional-forecasters/"
SURVEY = SPF + "data-files/files/Median_RGDP_Growth.xlsx"
RELEASES = FED + "real-time-data/data-files/xlsx/routput_first_second_third.xlsx"
DEADLINES = SPF + "spf-release-dates.txt"
ALFRED = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=GDPC1&vintage_date="
EVALUATION_VINTAGE = "2026-10-29"
SUPPLEMENT = {"2025Q3": "2025-12-23", "2025Q4": "2026-02-20"}


def sheet(workbook, member):
    """Read the small set of nonempty cells used in the Philadelphia Fed workbooks."""
    with ZipFile(io.BytesIO(workbook)) as book:
        strings = [
            "".join(item.itertext()) for item in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ]
        root = ET.fromstring(book.read(f"xl/worksheets/sheet{member}.xml"))
    for row in root.iterfind(".//{*}row"):
        cells = {}
        for c in row.iterfind("{*}c"):
            v = c.findtext("{*}v")
            if v is not None:
                cells[re.sub(r"\d+$", "", c.attrib["r"])] = (
                    strings[int(v)] if c.get("t") == "s" else v
                )
        yield cells


def survey_projection(growth, releases, deadlines):
    """Project published sources onto finite historical origins; future releases can append rows."""
    annualized = {}
    for row in sheet(growth, 1):
        if row.get("A") == "YEAR":
            if row.get("D") != "DRGDP3":
                raise ValueError("Unexpected survey horizon column")
        elif row.get("A", "").replace(".", "").isdigit():
            origin = pd.Period(f"{int(float(row['A']))}Q{int(float(row['B']))}", freq="Q")
            if pd.Period("1996Q4", freq="Q") <= origin <= pd.Period("2025Q4", freq="Q"):
                annualized[origin] = float(row["D"])
    first = {}
    for row in sheet(releases, 2):
        if re.fullmatch(r"\d{4}:Q[1-4]", row.get("A", "")):
            first[pd.Period(row["A"].replace(":", ""), freq="Q")] = pd.to_numeric(
                row.get("B"), errors="coerce"
            )
    dates = {}
    for line in deadlines.splitlines():
        match = re.match(r"^\s*(?:(\d{4})\s+)?Q([1-4])\s+(\d+/\d+/\d+)\**\s+(\d+/\d+/\d+)", line)
        if match:
            if match[1]:
                year = int(match[1])
            dates[pd.Period(f"{year}Q{match[2]}", freq="Q")] = [
                datetime.strptime(match[i], "%m/%d/%y").date().isoformat() for i in (3, 4)
            ]
    rows = []
    for origin in pd.period_range("1996Q4", "2025Q4", freq="Q"):
        target = origin + 1
        # For both shutdown quarters use their separately dated BEA initial announcements.
        label = np.nan if str(target) in SUPPLEMENT else first[target]
        rows.append((str(origin), *dates[origin], annualized[origin], label))
    return pd.DataFrame(rows, columns=["origin", "deadline", "news_release", "spf", "first"])


def vintage(cache_dir, date, hashes):
    path = fetch(ALFRED + date, cache_dir, f"alfred-gdp-{date}.csv", hashes[date], user_agent=None)
    values = pd.read_csv(path, index_col="observation_date", parse_dates=True)
    column = values.columns.item()
    key = re.fullmatch(r"GDPC1_(\d{8})", column)
    if key is None or key[1] > date.replace("-", ""):
        raise ValueError(f"Unexpected or future vintage column {column}")
    values = values[column].dropna()
    values.index = values.index.to_period("Q")
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError("Expected one time-ordered GDP level per quarter")
    return values


def annual_growth(level):
    return 100 * ((level / level.shift(1)) ** 4 - 1)


def ar_forecast(level, last_public, target):
    """Fit fixed AR(1) on growth available through last_public; recurse to the target."""
    if target <= last_public:
        raise ValueError("Target must follow the latest public quarter")
    level = level.loc["1991Q4":last_public]
    if len(level) < 12 or level.index[-1] != last_public:
        raise ValueError("Incomplete public GDP history")
    if not level.index.equals(pd.period_range(level.index[0], last_public, freq="Q")):
        raise ValueError("A GDP observation quarter is missing inside a vintage")
    growth = annual_growth(level).iloc[1:].to_numpy()
    beta = np.linalg.lstsq(
        np.column_stack([np.ones(len(growth) - 1), growth[:-1]]), growth[1:], rcond=None
    )[0]
    value = growth[-1]
    for _ in range(target.ordinal - last_public.ordinal):
        value = beta[0] + beta[1] * value
    return float(value)


def metrics(actual, forecast):
    error = np.asarray(actual) - np.asarray(forecast)
    return {
        "n": len(error),
        "mean_actual_minus_forecast": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def run(cache_dir, verify_survey=False):
    survey = pd.read_csv(HERE / "gdp-survey.csv")
    if verify_survey:
        with (
            urlopen(SURVEY, timeout=90) as a,
            urlopen(RELEASES, timeout=90) as b,
            urlopen(DEADLINES, timeout=90) as c,
        ):
            current = survey_projection(a.read(), b.read(), c.read().decode())
        pd.testing.assert_frame_equal(current, survey)
    hashes = pd.read_csv(HERE / "gdp-vintages.csv").set_index("vintage").sha256
    revised = vintage(cache_dir, EVALUATION_VINTAGE, hashes)
    revised_growth = annual_growth(revised)
    initial = {}
    for quarter, day in SUPPLEMENT.items():
        first = vintage(cache_dir, day, hashes)
        assert first.index[-1] == pd.Period(quarter, freq="Q")
        initial[quarter] = float(annual_growth(first).iloc[-1])
    np.testing.assert_allclose(list(initial.values()), [4.3, 1.4], atol=0.05, rtol=0)

    official = survey[survey.origin.between("1996Q4", "2022Q4")]
    reference = metrics(official["first"], official.spf)
    assert reference["n"] == 105
    np.testing.assert_allclose(
        list(reference.values())[1:], [-0.37, 2.02, 4.54], atol=0.005, rtol=0
    )
    origins = pd.period_range("1992Q1", "2025Q4", freq="Q")
    split = TimeSeriesCrossValidator(n_splits=104, test_size=1, expanding=True)
    results = []
    survey = survey.set_index("origin")
    for train, test in split.split(np.arange(len(origins))):
        origin = origins[test.item()]
        target = origin + 1
        obs = survey.loc[str(origin)]
        history = vintage(cache_dir, obs.deadline, hashes)
        last = history.index[-1]
        if last > origins[train[-1]] or last > pd.Timestamp(obs.deadline).to_period("Q"):
            raise ValueError("A feature uses a future GDP observation")
        label = initial.get(str(target), obs["first"])
        if not np.isfinite(label):
            raise ValueError(f"Missing first-release label for {target}")
        results.append(
            {
                "origin": str(origin),
                "deadline": obs.deadline,
                "published": obs.news_release,
                "last_input": str(last),
                "last_input_growth_as_of": float(annual_growth(history).iloc[-1]),
                "last_input_growth_revised": float(revised_growth[last]),
                "target": str(target),
                "first": float(label),
                "revised": float(revised_growth[target]),
                "survey": float(obs.spf),
                "as_of_ar": ar_forecast(history, last, target),
                "revised_input_ar": ar_forecast(revised, last, target),
            }
        )
    rows = pd.DataFrame(results)
    if rows.iloc[-1].last_input != "2025Q2":
        raise AssertionError("Shutdown-quarter feature cutoff drifted")
    mae_change = {}
    for label in ("first", "revised"):
        losses = (rows[label] - rows.revised_input_ar).abs() - (rows[label] - rows.as_of_ar).abs()
        delta = float(losses.mean())
        se = float(newey_west_se(losses.to_numpy() - delta, bandwidth=4).se)
        mae_change[label] = {
            "revised_input_minus_as_of": delta,
            "hac_se": se,
            "interval": [delta - 1.95996398454 * se, delta + 1.95996398454 * se],
        }
    return {
        "mae_change": mae_change,
        "quarters": len(rows),
        "official_reference": reference,
        "metrics": {
            label: {
                model: metrics(rows[label], rows[model])
                for model in ("survey", "as_of_ar", "revised_input_ar")
            }
            for label in ("first", "revised")
        },
        "supplemental_first": initial,
        "shutdown": results[-1],
        "origins": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-survey",
        action="store_true",
        help="also compare archived survey cells to live Fed files",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir, args.verify_survey), indent=2, allow_nan=False))
