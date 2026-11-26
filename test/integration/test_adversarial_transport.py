"""Guard the cohorts, units and integrity of the retrospective transport comparison."""

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

audit = importlib.import_module("docs.audits.2026-11-19_time_series_dml.adversarial.transport")
expected = json.loads((audit.HERE / "transport.json").read_bytes())


@pytest.mark.tier1
def test_gss_forecasts_use_earlier_waves_and_the_same_later_target():
    data = expected["gss"]
    assert len(data["later_waves"]) == len(data["history"]) - 1 == 7
    for idx, row in enumerate(data["later_waves"], 1):
        prior = data["history"][:idx]
        assert max(w["year"] for w in prior) < row["year"] == data["history"][idx]["year"]
        assert row["target"] == data["history"][idx]["direct"]
        for method, field in [
            ("historical", "direct"),
            ("aipw", "aipw"),
            ("standardized_ols", "ols"),
        ]:
            assert row["forecasts"][method] == pytest.approx(np.mean([w[field] for w in prior]))
    assert audit.forecast_rmse(data["later_waves"]) == pytest.approx(data["rmse"])


@pytest.mark.tier2
@pytest.mark.skipif(
    os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="pinned survey and labor sources"
)
def test_pinned_survey_and_earnings_reproduce_matched_alternatives():
    cache = Path(os.environ.get("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache/dml-ts/external"))
    assert audit.survey(cache)["rmse"] == pytest.approx(expected["gss"]["rmse"])
    wages = audit.training_offers(cache)
    for cohort in ("history", "holdout"):
        assert wages[cohort] == pytest.approx(expected["jtpa"][cohort])
    assert sum(wages["later_observed_at_new_sites"].values()) == 428
