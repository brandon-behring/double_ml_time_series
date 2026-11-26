"""Ex-ante information sets, randomized-arm support, and published-data replays."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "docs/audits/2026-11-19_time_series_dml/external/forward"


def load_case(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gdp = load_case("gdp")
pa = load_case("pennsylvania")


@pytest.mark.tier1
def test_gdp_forecast_ignores_every_unreleased_quarter():
    dates = pd.period_range("1991Q4", "2003Q4", freq="Q")
    levels = pd.Series(100 * np.exp(0.025 * np.arange(len(dates))), index=dates)
    latest = pd.Period("1999Q2", freq="Q")
    # At an irregular deadline the forecast may need to recurse several quarters.
    target = latest + 3
    before = gdp.ar_forecast(levels, latest, target)
    levels.loc[latest + 1 :] = 9000000
    assert before == pytest.approx(100 * (np.exp(0.1) - 1))
    assert gdp.ar_forecast(levels, latest, target) == before
    with pytest.raises(ValueError, match="missing"):
        gdp.ar_forecast(levels.drop(latest - 2), latest, target)


@pytest.mark.tier1
def test_forward_bonus_rejects_unmatured_labels_and_future_people():
    rows = []
    for office, baseline in [(1, 45), (2, 100)]:
        for arm in (0, 4):
            for slot in range(4):
                for start, selection in [
                    ("1988-11-01", "1988-11-09"),
                    ("1989-04-01", "1989-04-10"),
                    ("1989-07-01", "1989-07-12"),
                ]:
                    rows.append(
                        {
                            "lo": office,
                            "tg": arm,
                            "allocation_period": 1,
                            "ab_dt": pd.Timestamp(start),
                            "selt_dt": pd.Timestamp(selection) + pd.Timedelta(days=slot),
                            "wages": baseline + slot + 200 * (arm == 4),
                        }
                    )
    data = pd.DataFrame(rows)
    early = pa.early_records(data, 1, "earnings")
    expected = pa.contrast(early, "wages")
    assert expected["estimate"] == pytest.approx(200)
    assert expected["n"] == 16 and expected["offices"] == 2
    data.loc[~data.index.isin(early.index), "wages"] = -123456
    assert pa.contrast(pa.early_records(data, 1, "earnings"), "wages") == expected
    with pytest.raises(ValueError, match="per arm"):
        pa.contrast(early.loc[~(early.tg.eq(4) & early.lo.eq(2))], "wages")


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="source archive opt-in")
@pytest.mark.parametrize("name", ["gdp", "pa"])
def test_forward_economic_archive(name, tmp_path):
    cache = Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path))
    if name == "gdp":
        result = gdp.run(cache)
        assert result["quarters"] == 104
        assert result["metrics"]["first"]["as_of_ar"]["rmse"] == pytest.approx(5.55944, abs=1e-5)
        assert result["shutdown"]["last_input"] == "2025Q2"
        recession = next(x for x in result["origins"] if x["origin"] == "2009Q1")
        np.testing.assert_allclose(
            [recession["last_input_growth_as_of"], recession["last_input_growth_revised"]],
            [-3.80367, -8.47277],
            atol=1e-5,
            rtol=0,
        )
    else:
        result = pa.run(cache)
        assert result["historical_table_VIII_3"]["pooled"] == pytest.approx(74.1407, abs=1e-4)
        assert result["earnings"]["1"]["prediction_minus_later"] == pytest.approx(
            -108.284, abs=1e-3
        )
        assert result["lag_two_common_office"]["later"]["n"] == 116
