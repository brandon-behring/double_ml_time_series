import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from dml_ts.validation.effect_forecast import (
    adjustment_direction,
    forecast_comparison,
    freeze,
    ranking_precision,
    score,
)

pytestmark = pytest.mark.tier1


def case():
    protocol = {
        "target": "assignment ITT",
        "unit": "pp",
        "horizon": "7 days",
        "methods": ["zero", "historical", "dml"],
        "origins": ["2023-01-01", "2023-04-01"],
    }
    forecasts = [
        {"origin": origin, "method": method, "value": i}
        for origin in protocol["origins"]
        for i, method in enumerate(protocol["methods"])
    ]
    forecasts[-1].update(value=None, reason="no historical treatment variation")
    obs = {
        "origin": "2023-01-01",
        "estimate": 1.5,
        **{k: protocol[k] for k in ("target", "unit", "horizon")},
    }
    return protocol, forecasts, obs


def test_forecasts_and_outcomes_are_separate_and_every_origin_retained(tmp_path):
    protocol, forecasts, obs = case()
    path = tmp_path / "sealed.json"
    h = freeze(path, protocol, forecasts)
    rows = score(path, h, [obs])
    assert len(rows) == 6
    assert rows[2]["signed_error"] == 0.5 and rows[2]["gain_vs_zero"] == 2
    assert rows[2]["gain_vs_historical"] == 0
    assert rows[3]["reason"] == "Later estimate unavailable"
    assert rows[-1]["reason"] == "no historical treatment variation"
    assert score(path, h, [{**obs, "estimate": 2}])[2]["signed_error"] == 0
    assert h == freeze(path, protocol, forecasts[::-1])


def test_concurrent_publication_and_tamper_check(tmp_path):
    p, f, obs = case()
    path = tmp_path / "file.json"
    with ThreadPoolExecutor(max_workers=5) as pool:
        hashes = list(pool.map(lambda _: freeze(path, p, f), range(8)))
    assert len(set(hashes)) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        freeze(path, p, [dict(f[0], value=99), *f[1:]])
    changed = json.loads(path.read_text())
    changed["forecasts"][0]["value"] = 99
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SHA-256"):
        score(path, hashes[0], [obs])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "infinite", "unexplained"])
def test_invalid_forecasts(tmp_path, mutation):
    p, f, _ = case()
    if mutation == "missing":
        f.pop()
    if mutation == "duplicate":
        f.append(f[0])
    if mutation == "infinite":
        f[0]["value"] = float("inf")
    if mutation == "unexplained":
        f[0]["value"] = None
    with pytest.raises(ValueError):
        freeze(tmp_path / "m", p, f)


@pytest.mark.parametrize("key", ["horizon", "target", "unit", "intervention", "comparator"])
def test_changed_contrast_is_rejected(tmp_path, key):
    p, f, obs = case()
    p.update(intervention="activity message", comparator="no message")
    obs.update(intervention=p["intervention"], comparator=p["comparator"])
    h = freeze(tmp_path / "m", p, f)
    with pytest.raises(ValueError, match="Mismatched"):
        score(tmp_path / "m", h, [{**obs, key: "different"}])
    with pytest.raises(ValueError, match="Duplicate"):
        score(tmp_path / "m", h, [obs, obs])


@pytest.mark.parametrize(
    "f,b,ci,status",
    [
        (2, 1, (1.51, 1.60), "forecast_closer"),
        (1, 2, (1.51, 1.60), "comparator_closer"),
        (2, 1, (1.49, 1.60), "unresolved"),
        (2, 1, (1.50, 1.60), "unresolved"),
        (2, 2, (1.51, 1.60), "identical"),
    ],
)
def test_ranking_matches_difference_of_losses_and_respects_interval(f, b, ci, status):
    z = 1.55
    actual = forecast_comparison(f, b, z, ci)
    assert actual["midpoint"] == (f + b) / 2
    assert actual["gain"] == pytest.approx((z - b) ** 2 - (z - f) ** 2)
    assert actual["gain_ci95"] == pytest.approx(sorted((v - b) ** 2 - (v - f) ** 2 for v in ci))
    assert actual["status"] == status


def test_ranking_precision_is_for_two_sided_interval_and_identical_forecasts():
    from statistics import NormalDist

    se = ranking_precision(1, 2)["se_for_80pct"]
    normal = NormalDist()
    assert normal.cdf(0.5 / se - normal.inv_cdf(0.975)) == pytest.approx(0.8)
    assert ranking_precision(1, 1)["se_for_80pct"] is None
    assert forecast_comparison(1, 2, None)["status"] == "pending"
    assert forecast_comparison(1, 2, 1)["status"] == "point_only"


@pytest.mark.parametrize(
    "forecast,historical,direction",
    [
        (-0.5, -1, "toward_zero"),
        (-1.1, -1, "away_from_zero"),
        (0.01, -1, "sign_reversal"),
        (0, -1, "toward_zero"),
        (-1, 0, "away_from_zero"),
        (0, 0, "unchanged"),
    ],
)
def test_direction(forecast, historical, direction):
    assert adjustment_direction(forecast, historical) == direction


@pytest.mark.parametrize("ci", [[2, 1], [1], [True, 1], [0, float("nan")], ".5", 2])
def test_accept_only_valid_benchmark_interval(ci):
    with pytest.raises(ValueError, match="ci95"):
        forecast_comparison(1, 2, 0.5, ci)


def test_reject_invalid_benchmark_even_when_all_forecasts_are_missing(tmp_path):
    protocol, forecasts, obs = case()
    protocol["comparators"] = ["historical"]
    for row in forecasts:
        row.update(value=None, reason="No earlier data")
    path = tmp_path / "frozen.json"
    digest = freeze(path, protocol, forecasts)
    with pytest.raises(ValueError, match="ci95"):
        score(path, digest, [{**obs, "ci95": [2, 1]}])


def test_explicit_comparators_are_frozen_and_half_history_is_exact(tmp_path):
    p, f, obs = case()
    p["methods"].append("half_history")
    p["comparators"] = ["historical", "half_history"]
    for origin in p["origins"]:
        f.append({"origin": origin, "method": "half_history", "value": 0.5})
    path = tmp_path / "m"
    digest = freeze(path, p, f)
    rows = score(path, digest, [{**obs, "ci95": [1.4, 1.6]}])
    dml = next(r for r in rows if r["method"] == "dml" and r["origin"] == obs["origin"])
    assert dml["gain_vs_half_history"] == 0.75
    assert dml["comparisons"]["half_history"]["status"] == "forecast_closer"
    assert dml["comparisons"]["historical"]["status"] == "unresolved"
    assert dml["direction_vs_historical"] == "away_from_zero"
    assert rows[-1]["comparisons"]["historical"]["status"] == "pending"
    f[-1]["value"] = 0.6
    with pytest.raises(ValueError, match="half_history"):
        freeze(tmp_path / "bad", p, f)
    with pytest.raises(ValueError, match="comparator"):
        freeze(tmp_path / "bad", {**p, "comparators": ["missing"]}, f)
