"""Freeze effect forecasts and score them against separately supplied experimental estimates."""

import argparse
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
from statistics import NormalDist
from tempfile import NamedTemporaryFile
from typing import Any, TypeGuard


def _labels(protocol: dict[str, Any], key: str) -> list[str]:
    labels = protocol.get(key)
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label.strip() for label in labels)
    ):
        raise ValueError(f"Expected nonempty {key}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate {key}")
    return labels


def _finite_number(value: object) -> TypeGuard[float | int]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row[key]
    if value is None:
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"Unavailable {key} requires a reason")
        return None
    if not _finite_number(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def ranking_precision(forecast: float, comparator: float) -> dict[str, float | None]:
    """Benchmark SE for 80% power to favor the true forecast using a two-sided 95% CI.

    Assumes a normal benchmark centered on either fixed forecast. At their midpoint
    both forecasts have the same loss, regardless of the available sample size.
    """
    if not all(_finite_number(x) for x in (forecast, comparator)):
        raise ValueError("Ranking requires two finite forecasts")
    separation = abs(forecast - comparator)
    normal = NormalDist()
    return {
        "midpoint": forecast / 2 + comparator / 2,
        "separation": separation,
        "se_for_80pct": separation / (2 * (normal.inv_cdf(0.975) + normal.inv_cdf(0.8)))
        if separation
        else None,
    }


def _validate_interval(
    estimate: float | None,
    ci95: tuple[float, float] | list[float] | None,
) -> None:
    if ci95 is not None and (
        estimate is None
        or not isinstance(ci95, (tuple, list))
        or len(ci95) != 2
        or not all(_finite_number(v) for v in ci95)
        or ci95[0] > ci95[1]
    ):
        raise ValueError("Expected an ordered finite ci95 for an observed benchmark")


def forecast_comparison(
    forecast: float,
    comparator: float,
    estimate: float | None,
    ci95: tuple[float, float] | list[float] | None = None,
) -> dict[str, Any]:
    """Paired squared-loss gain, conditional on two frozen numbers and a common target."""
    midpoint = ranking_precision(forecast, comparator)["midpoint"]
    assert midpoint is not None
    if estimate is not None and not _finite_number(estimate):
        raise ValueError("Expected a finite benchmark")
    _validate_interval(estimate, ci95)
    slope = 2 * (forecast - comparator)
    gain = None if estimate is None else slope * (estimate - midpoint)
    interval = None if ci95 is None else sorted(slope * (bound - midpoint) for bound in ci95)
    if forecast == comparator:
        status = "identical"
    elif estimate is None:
        status = "pending"
    elif interval is None:
        status = "point_only"
    elif interval[0] > 0:
        status = "forecast_closer"
    elif interval[1] < 0:
        status = "comparator_closer"
    else:
        status = "unresolved"
    return {"midpoint": midpoint, "gain": gain, "gain_ci95": interval, "status": status}


def adjustment_direction(forecast: float, historical: float) -> str:
    """Describe the change from a historical estimate on a zero-null effect scale."""
    if forecast == historical:
        return "unchanged"
    if (forecast < 0 < historical) or (historical < 0 < forecast):
        return "sign_reversal"
    return "toward_zero" if abs(forecast) < abs(historical) else "away_from_zero"


def freeze(path: Path, protocol: dict[str, Any], forecasts: list[dict[str, Any]]) -> str:
    """Publish once, atomically. Identical retries return the same SHA-256."""
    for field in ("target", "unit", "horizon"):
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            raise ValueError(f"Missing protocol {field}")
    origins = _labels(protocol, "origins")
    methods = _labels(protocol, "methods")
    if "comparators" in protocol and not set(_labels(protocol, "comparators")).issubset(methods):
        raise ValueError("Every comparator must be a scheduled method")
    if [date.fromisoformat(x).isoformat() for x in origins] != sorted(origins):
        raise ValueError("Expected chronological ISO calendar dates")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for forecast in forecasts:
        key = (forecast["origin"], forecast["method"])
        if key in indexed:
            raise ValueError(f"Duplicate forecast: {key}")
        _number(forecast, "value")
        indexed[key] = forecast
    expected = [(origin, method) for origin in origins for method in methods]
    if indexed.keys() != set(expected):
        raise ValueError("Supply exactly one forecast or reason per scheduled origin and method")
    if "half_history" in methods:
        if "historical" not in methods:
            raise ValueError("half_history requires a historical forecast")
        for origin in origins:
            value = indexed[origin, "historical"]["value"]
            if indexed[origin, "half_history"]["value"] != (None if value is None else value / 2):
                raise ValueError(
                    f"half_history must equal half the historical forecast at {origin}"
                )

    record = {"version": 1, "protocol": protocol, "forecasts": [indexed[k] for k in expected]}
    content = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A hard link publishes the fully written file only if the destination is still free.
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        tmp = Path(temporary.name)
        try:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"Conflicting freeze at {path}") from None
        finally:
            tmp.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def score(
    path: Path, expected_digest: str, observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return every scheduled forecast with signed error and paired baseline loss gains."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError("Frozen forecast SHA-256 mismatch")
    record = json.loads(content)
    if record["version"] != 1:
        raise ValueError("Unsupported forecast format")
    protocol = record["protocol"]
    measured = {}
    for obs in observations:
        origin = obs["origin"]
        if origin in measured or origin not in protocol["origins"]:
            raise ValueError(f"Duplicate or unscheduled observation: {origin}")
        for key in ("target", "unit", "horizon", "intervention", "comparator"):
            if key in protocol and obs.get(key) != protocol[key]:
                raise ValueError(f"Mismatched {key} for {origin}")
        estimate = _number(obs, "estimate")
        _validate_interval(estimate, obs.get("ci95"))
        measured[origin] = obs

    rows = []
    losses = {}
    for forecast in record["forecasts"]:
        origin, method, value = forecast["origin"], forecast["method"], forecast["value"]
        obs = measured.get(origin, {"estimate": None, "reason": "Later estimate unavailable"})
        estimate = obs["estimate"]
        reason = forecast.get("reason") if value is None else obs.get("reason")
        error = None if value is None or estimate is None else value - estimate
        loss = None if error is None else error**2
        losses[(origin, method)] = loss
        rows.append(
            {
                "origin": origin,
                "method": method,
                "forecast": value,
                "estimate": estimate,
                "signed_error": error,
                "squared_error": loss,
                "reason": reason,
            }
        )
    predictions = {(row["origin"], row["method"]): row["forecast"] for row in rows}
    for row in rows:
        origin = row["origin"]
        if "comparators" in protocol:
            comparisons = {}
            for baseline in protocol["comparators"]:
                other = predictions[origin, baseline]
                pair = (
                    None
                    if other is None or row["forecast"] is None
                    else forecast_comparison(
                        row["forecast"],
                        other,
                        row["estimate"],
                        measured.get(origin, {}).get("ci95"),
                    )
                )
                comparisons[baseline] = pair
                row[f"gain_vs_{baseline}"] = pair["gain"] if pair else None
            row["comparisons"] = comparisons
            historical = predictions.get((origin, "historical"))
            row["direction_vs_historical"] = (
                adjustment_direction(row["forecast"], historical)
                if row["forecast"] is not None and historical is not None
                else None
            )
            continue
        for baseline in ("zero", "historical"):
            reference = losses.get((row["origin"], baseline))
            loss = row["squared_error"]
            row[f"gain_vs_{baseline}"] = (
                None if reference is None or loss is None else reference - loss
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("freeze")
    issue.add_argument("--protocol", required=True, type=Path)
    issue.add_argument("--forecasts", required=True, type=Path)
    issue.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--frozen", required=True, type=Path)
    evaluate.add_argument("--sha256", required=True)
    evaluate.add_argument("--observations", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(
            freeze(
                args.output,
                json.loads(args.protocol.read_text()),
                json.loads(args.forecasts.read_text()),
            )
        )
    else:
        print(
            json.dumps(
                score(args.frozen, args.sha256, json.loads(args.observations.read_text())),
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
