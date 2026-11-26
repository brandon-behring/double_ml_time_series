"""Numerical and release-gate oracles for prospective GSS form effects."""

import hashlib
import importlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gss = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.prospective_gss"
)


def miniature_wave():
    # The fourth PSU contains only an ineligible ballot D respondent.
    return pd.DataFrame(
        [
            (1, 1, 1, 1, 1.0, ".i"),
            (1, 1, 1, 2, ".i", ".n"),
            (1, 2, 1, 1, ".d", ".i"),
            (1, 2, 2, 2, ".i", 1.0),
            (2, 1, 3, 1, 3.0, ".i"),
            (2, 1, 3, 2, ".i", 1.0),
            (2, 2, 4, 1, ".i", ".i"),
        ],
        columns=["vstrat", "vpsu", "ballot", "form", "natfare", "natfarey"],
    ).assign(year=2028, id=range(7), wtssps=1.0)


@pytest.mark.tier1
def test_hajek_uses_assigned_form_denominators_and_zero_domain_psus():
    df = gss.decode(miniature_wave())
    result = gss.survey_difference(df, df.eligible)
    assert result["estimate"] == pytest.approx(100 / 3)
    assert result["welfare_n"] == result["assistance_n"] == 3
    assert result["welfare_nonresponse"] == result["assistance_nonresponse"] == 1
    assert result["design_df"] == 2
    assert result["psus"] == 4
    # Independent, by-hand PSU contributions (-4/9, 2/9) and (2/9, 0).
    assert result["se"] == pytest.approx(100 * np.sqrt(40) / 9)
    assert gss.survey_difference(df, df.valid)["estimate"] == pytest.approx(50)
    weighted = df.assign(wtssps=[2, 1, 1, 1, 1, 1, 3])
    expected = 100 * (2 / 3 - 2 / 4)
    assert gss.survey_difference(weighted, weighted.eligible)["estimate"] == pytest.approx(expected)
    permuted = df.sample(frac=1, random_state=17).assign(wtssps=7)
    assert gss.survey_difference(permuted, permuted.eligible)["se"] == pytest.approx(result["se"])


@pytest.mark.tier1
@pytest.mark.parametrize(
    ("col", "value"),
    [("form", 3), ("ballot", 0), ("natfare", ".i"), ("natfarey", 2), ("wtssps", 0)],
)
def test_unverified_layout_or_survey_design_cannot_supply_a_benchmark(col, value):
    frame = miniature_wave()
    frame.loc[0, col] = value
    with pytest.raises(ValueError):
        gss.decode(frame)


def design_review():
    return {
        "wave": 2028,
        "status": "compatible",
        "checks": dict.fromkeys(gss.CHECKS, True),
        "design_source": "synthetic review fixture",
        "data_source": "synthetic fixture",
        "questionnaire_source": "synthetic instrument",
        "english_form_x": "Welfare",
        "english_form_y": "Assistance for the poor",
        "context_and_languages": "Reviewed language, mode and prior item context on all ballots",
        "reason": "All fields verified in synthetic fixture",
        "reviewer": "test",
        "fieldwork_start": "2027-12-15",
        "fieldwork_end": "2028-03-01",
        "released": "2028-09-01",
    }


@pytest.mark.tier1
def test_review_preserves_schedule_without_loading_data(monkeypatch):
    monkeypatch.setattr(gss, "read_stata", lambda *args, **kwargs: pytest.fail("premature access"))
    pending = gss.assess(gss.RECORDS, {"wave": 2028, "status": "pending"})
    assert [r["method"] for r in pending["rows"]] == gss.METHODS
    review = design_review()
    review.update(status="incompatible", reason="Different intervention wording")
    review["checks"]["same_form_wordings"] = False
    invalid = gss.assess(gss.RECORDS, review)
    assert invalid["status"] == "incompatible"
    assert invalid["rows"][-1]["comparisons"]["half_history"]["status"] == "incompatible"
    assert all(r["estimate"] is None and r["reason"] == review["reason"] for r in invalid["rows"])
    with pytest.raises(ValueError, match="conflicts"):
        gss.assess(gss.RECORDS, {**review, "status": "compatible"})
    with pytest.raises(ValueError, match="dates"):
        gss.assess(gss.RECORDS, design_review(), today=date(2027, 1, 1))
    changed = {**design_review(), "english_form_y": "Temporary income transfers"}
    assert gss.assess(gss.RECORDS, changed)["status"] == "incompatible"


@pytest.mark.tier1
def test_scoring_requires_released_pinned_source_and_routing(tmp_path, monkeypatch):
    review = design_review()
    data = tmp_path / "example.dta"
    data.write_bytes(b"synthetic test bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        gss.assess(gss.RECORDS, review, data, today=date(2029, 1, 1))
    review["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    frame = miniature_wave()
    monkeypatch.setattr(gss, "read_stata", lambda *args, **kwargs: frame)
    observed = gss.assess(gss.RECORDS, review, data, today=date(2029, 1, 1))
    assert observed["status"] == "scored"
    assert observed["benchmark"]["estimate"] == pytest.approx(100 / 3)
    dml = observed["rows"][-1]
    assert dml["comparisons"]["half_history"]["status"] == "unresolved"
    assert all(not flag for flag in dml["equivalence_sensitivity"].values())
    historical_variant = {**review, "english_form_y": "Assistance to the poor"}
    frame.drop(columns="vstrat", inplace=True)
    point_only = gss.assess(gss.RECORDS, historical_variant, data, today=date(2029, 1, 1))
    assert point_only["status"] == "scored"
    assert point_only["benchmark"]["se"] is None
    assert point_only["benchmark"]["estimate"] == pytest.approx(100 / 3)
    assert point_only["rows"][-1]["comparisons"]["half_history"]["status"] == "point_only"
    frame.loc[0, "natfare"] = ".i"
    failed = gss.assess(gss.RECORDS, review, data, today=date(2029, 1, 1))
    assert failed["status"] == "incompatible"
    assert all(r["signed_error"] is None for r in failed["rows"])


@pytest.mark.tier1
def test_singleton_stratum_has_point_only_and_freeze_matches_receipt():
    frame = miniature_wave().iloc[:-1]
    df = gss.decode(frame)
    result = gss.survey_difference(df, df.eligible)
    assert result["se"] is result["ci95"] is None
    assert result["estimate"] == pytest.approx(100 / 3)
    receipt = json.loads((gss.RECORDS / "gss-issuance.json").read_text())
    manifest = (gss.RECORDS / "gss-frozen.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == receipt["sha256"]
    frozen = json.loads(manifest)
    assert frozen["protocol"]["issued_at_utc"] == receipt["issued_at_utc"]
    vals = [r["value"] for r in frozen["forecasts"]]
    assert vals == [0, 42.287214013, 21.1436070065, 42.441534778]
    assert vals[2] == vals[1] / 2


@pytest.mark.tier1
def test_replacing_record_and_nearby_receipt_cannot_shift_a_forecast(tmp_path):
    manifest = json.loads((gss.RECORDS / "gss-frozen.json").read_text())
    manifest["forecasts"][-1]["value"] += 1
    replaced = json.dumps(manifest).encode()
    receipt = {"issued_at_utc": gss.ISSUED_UTC, "sha256": hashlib.sha256(replaced).hexdigest()}
    (tmp_path / "gss-issuance.json").write_text(json.dumps(receipt))
    (tmp_path / "gss-frozen.json").write_bytes(replaced)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        gss.assess(tmp_path, {"wave": 2028, "status": "pending"})


@pytest.mark.tier2
@pytest.mark.skipif(os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="pinned GSS source")
def test_public_release_recreates_all_eight_wave_estimates_and_the_issuance():
    cache = Path(os.environ.get("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache/dml-ts/external"))
    rows = gss.history(cache)
    archived = pd.read_csv(gss.RECORDS / "gss-history.csv")
    assert archived.year.tolist() == [r["year"] for r in rows] == list(gss.WAVES)
    assert not set(gss.ERRATA).intersection(gss.WAVES)
    for col in ("estimate", "se", "welfare_n", "assistance_n", "aipw_pp", "valid_only_pp"):
        assert archived[col].to_numpy() == pytest.approx([row[col] for row in rows])
    np.testing.assert_allclose(
        archived.nonresponse_weight_pp.to_numpy(),
        [r["nonresponse_weight_pp"] or np.nan for r in rows],
        rtol=1e-8,
        equal_nan=True,
    )
    assert archived.iloc[-1].estimate == pytest.approx(36.81900224)
    # NORC Social Change Report 63, tables K1/K2: published one-decimal
    # substantive-answer rates, from an earlier vintage and weighting convention.
    valid = {r["year"]: r["valid_only_pp"] for r in rows}
    assert valid[2008] == pytest.approx(70.2 - 25.4, abs=0.15)
    assert valid[2014] == pytest.approx(63.6 - 19.8, abs=0.15)
    assert pd.isna(archived.prior_dml_pp.iloc[0])
    assert archived.prior_dml_pp.iloc[-1] == pytest.approx(archived.aipw_pp.iloc[:-1].mean())
    assert rows[2]["welfare_n"] + rows[2]["assistance_n"] == 2992
    record = json.loads((gss.RECORDS / "gss-frozen.json").read_text())
    assert gss.forecast_rows(rows, record["protocol"]["origins"][0]) == record["forecasts"]
