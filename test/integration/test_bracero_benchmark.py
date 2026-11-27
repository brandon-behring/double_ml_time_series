"""Check the continuous-exposure wage replay, clustered inference and placebo timing."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

case = importlib.import_module("docs.audits.2026-11-19_time_series_dml.external.forward.bracero")


@pytest.mark.tier1
def test_exposure_uses_valid_1955_monthly_fractions_and_includes_recorded_zeros():
    frame = pd.DataFrame(
        [
            ("A", 1, 1955, 1, 1.0, 2),
            ("A", 1, 1955, 2, np.nan, 20),
            ("A", 1, 1955, 3, 0, 0),
            ("A", 1, 1956, 1, 100, 100),
            ("B", 2, 1955, 1, np.nan, np.nan),
        ],
        columns=[
            "State",
            "State_FIPS",
            "Year",
            "Month",
            "Mexican_final",
            "TotalHiredSeasonal_final",
        ],
    )
    exposure = case.exposure_1955(frame)
    assert exposure["A"] == 0.25
    assert np.isnan(exposure["B"])


def synthetic_panel():
    rng = np.random.default_rng(115)
    states = ["A", "B", "C", "D"]
    periods = pd.period_range("1956Q1", "1962Q4", freq="Q")
    panel = pd.MultiIndex.from_product([states, periods], names=["State", "quarter"]).to_frame(
        index=False
    )
    panel["exposure"] = panel.State.map(dict(zip(states, [0, 0.2, 0.5, 0.6], strict=True)))
    panel["wage"] = rng.normal(size=len(panel)) + 0.7 * panel.exposure * panel.quarter.dt.year.ge(
        1959
    )
    return panel


@pytest.mark.tier1
def test_unbalanced_fixed_effects_and_state_scores_match_dummy_regression():
    panel = synthetic_panel().drop([0, 4, 16, 29, 89])
    result = case.wage_contrast(panel, 1959)
    panel["time"] = panel.quarter.astype(str)
    panel["x"] = panel.exposure * panel.quarter.dt.year.ge(1959)
    fit = smf.ols("wage ~ x + C(State) + C(time)", panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel.State, "use_correction": False}
    )
    n, states, periods = result["state_quarters"], result["states"], result["quarters"]
    adjustment = np.sqrt(states / (states - 1) * (n - 1) / (n - periods - 1))
    assert result["estimate"] == pytest.approx(fit.params["x"])
    assert result["se"] == pytest.approx(fit.bse["x"] * adjustment)
    shuffled = panel.sample(frac=1, random_state=281)
    assert case.wage_contrast(shuffled, 1959) == result
    with pytest.raises(ValueError, match="Duplicate state-quarter"):
        case.wage_contrast(pd.concat([panel, panel.iloc[:1]]), 1959)
    with pytest.raises(ValueError, match="insufficient variation"):
        case.wage_contrast(panel, 1965)


@pytest.mark.tier1
def test_placebos_use_the_same_windows_and_cannot_see_1962_or_later():
    panel = synthetic_panel()
    grid = case.placebo_grid(panel)
    for cut, row in grid.items():
        assert row["first_quarter"] == f"{int(cut) - 2}Q1"
        assert row["last_quarter"] == f"{int(cut) + 1}Q4"
        assert row["quarters"] == 16
        assert row["p"] <= row["p_holm"] <= 1
    panel.loc[panel.quarter.dt.year.ge(1962), "wage"] += 1e6
    assert case.placebo_grid(panel) == grid
    with pytest.raises(ValueError, match="Incomplete calendar"):
        case.placebo_grid(panel[panel.quarter.astype(str).ne("1956Q4")])


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in archive replay")
def test_published_wage_and_three_pre_exclusion_checks(tmp_path):
    result = case.run(Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path)))
    published = result["table_1_hourly"]
    assert published["state_quarters"] == 4324
    assert published["estimate"] == pytest.approx(-0.03564384, abs=1e-7)
    assert published["se"] == pytest.approx(0.04259597, abs=1e-7)
    assert result["exposure_1955"]["high_states"] == ["AZ", "CA", "NE", "NM", "SD", "TX"]
    grid = result["placebos"]
    assert [row["state_quarters"] for row in grid.values()] == [736, 736, 736]
    assert [row["p_holm"] for row in grid.values()] == pytest.approx(
        [0.50565, 0.09311, 0.06149], abs=1e-5
    )
    assert result["exploratory_1959_six_years"]["estimate"] == pytest.approx(0.04311632, abs=1e-7)
