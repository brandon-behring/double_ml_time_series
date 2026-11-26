"""Validate the published encouragement contrasts and the limits of the public deposit."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

case = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.weatherization"
)


@pytest.mark.tier1
def test_authors_filter_requires_both_reading_counts_below_three():
    data = pd.DataFrame(
        {
            "normal": [1, 1, 1, 1, 0, 1],
            "IN_RED": [1, 1, 1, 1, 1, 0],
            "countPOST_GAS": [2, 2, 3, np.nan, 5, 5],
            "countPOST_ELEC": [2, 3, 2, 2, 5, 5],
        }
    )
    assert case.study_sample(data).index.tolist() == [1, 2, 3]


@pytest.mark.tier1
def test_household_contrasts_match_ols_and_count_missing_audits_as_zero():
    data = pd.DataFrame(
        {
            "fwhhid": range(7),
            "RED_ENC": [0, 0, 0, 0, 1, 1, 1],
            "IN_CAA": [0, 1, 0, 0, 1, 1, 1],
            "IN_NEAT": [0, 1, np.nan, 0, 0, 1, 1],
            "WAP": [0, 1, 0, 0, 0, 0, 1],
        }
    )
    estimates = case.takeup(data.sample(frac=1, random_state=31))["outcomes"]
    for name, column in case.OUTCOMES.items():
        fit = sm.OLS(data[column].fillna(0), sm.add_constant(data.RED_ENC)).fit()
        assert estimates[name]["effect_pp"] == pytest.approx(100 * fit.params["RED_ENC"])
        assert estimates[name]["published_ols_se_pp"] == pytest.approx(100 * fit.bse["RED_ENC"])
    assert estimates["audit"]["control_n"] == 4
    assert estimates["audit"]["control_events"] == 1
    with pytest.raises(ValueError, match="one record"):
        case.takeup(pd.concat([data, data.iloc[:1]]))


@pytest.mark.tier1
def test_engineering_summaries_keep_cost_and_savings_denominators_separate():
    measures = pd.DataFrame(
        {
            "neat_clientid": [10, 11, 12, np.nan, 14],
            "WAP": [1, 1, 1, 1, 0],
            "neat_npv_savings_cum": [100, 200, np.nan, 999, 999],
            "heatingmbtu": [4, 6, 5, 99, 99],
            "iwc_job_cost_act": [20, np.nan, 40, 999, 999],
        }
    )
    result = case.engineering(measures)
    assert result["audited_weatherized_records"] == 3
    assert result["projected_discounted_bill_savings_dollars"]["n"] == 2
    assert result["projected_discounted_bill_savings_dollars"]["mean"] == 150
    assert result["installation_cost_dollars"]["mean"] == 30
    assert result["projected_annual_heating_savings_mmbtu"]["n"] == 3


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(
    os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in source archive replay"
)
def test_michigan_author_archive(tmp_path):
    result = case.run(Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path)))
    assert result["archived_experiment_rows"] == 28890
    assert result["duplicate_household_rows_before_filter"] == 2
    assert result["table_iii"]["households"] == 28888
    outcomes = result["table_iii"]["outcomes"]
    assert [outcomes[name]["effect_pp"] for name in case.OUTCOMES] == pytest.approx(
        [12.57744155, 5.27205884, 4.93207348]
    )
    assert outcomes["weatherization"]["independent_arm_ci95_pp"] == pytest.approx(
        [4.39167228, 5.47247467]
    )
    engineering = result["engineering"]
    assert engineering["projected_discounted_bill_savings_dollars"]["n"] == 1619
    assert engineering["projected_discounted_bill_savings_dollars"]["mean"] == pytest.approx(
        10611.0188
    )
    assert engineering["projected_annual_heating_savings_mmbtu"]["mean"] == pytest.approx(
        50.4330311
    )
    assert result["billing_access"] == {
        "nonpublic_monthly_panels": ["RED_est.dta", "QUASI_cmb_est.dta"],
        "monthly_savings_refitted": False,
    }
