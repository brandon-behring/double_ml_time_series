"""Reconstruct the experimental and model-implied PROGRESA contrasts from author tables."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402
from forward import progresa_raw  # noqa: E402

PAPER = "https://economics.sas.upenn.edu/sites/default/files/filevault/working-papers/03-022.pdf"
DIGEST = "fe89ad44af3721d3537aa912c4b2ce82460e8c86571621ba9bd371ed2671e7fa"


def count_from_rate(percent, n):
    """Accept a count only when the printed one-decimal attendance rate identifies it."""
    possible = [i for i in range(n + 1) if abs(100 * i / n - percent) <= 0.05000001]
    if len(possible) != 1:
        raise ValueError("Rounded attendance rate does not determine a unique integer count")
    return possible[0]


def numbers(line):
    return [float(num) for num in re.findall(r"\d+(?:\.\d+)?", line)]


def compare(text):
    tables = re.split(r"\f\s*Table (11|12|13)\s*\n", text)
    table11 = tables[tables.index("11") + 1]
    table12 = tables[tables.index("12") + 1]
    rows = table11.split("Age 12-15\n", 1)[1].split("Behind in School", 1)[0]
    rates = numbers(re.search(r"^\s*Actual +(.+)$", rows, re.MULTILINE)[1])
    ns = numbers(re.search(r"^\s*No\. obs\. +(.+)$", rows, re.MULTILINE)[1])
    sections = table12.split("Boys, Age 12-15", 1)
    groups = {}
    for offset, (sex, section) in enumerate(zip(("girls", "boys"), sections, strict=True)):
        control, treated = offset * 4 + 1, offset * 4 + 3
        nc, nt = int(ns[control]), int(ns[treated])
        ac = count_from_rate(rates[control], nc)
        at = count_from_rate(rates[treated], nt)
        measured = 100 * (at / nt - ac / nc)
        predictions = {}
        for year in (97, 98):
            cols = numbers(re.search(rf"^\s*{year} Control +(.+)$", section, re.MULTILINE)[1])
            observed_control, predicted_subsidy, printed_delta = cols[:3]
            diff = predicted_subsidy - observed_control
            if abs(diff - printed_delta) > 0.101:
                raise ValueError("Reconstructed subsidy difference disagrees with Table 12")
            predictions[f"control_19{year}"] = {
                "predicted_pp": round(diff, 2),
                "forecast_minus_experiment_pp": diff - measured,
            }
        # Independent reference: Wolpin's Yale lecture, slide 129, structural-model row.
        published_ate, published_forecast, published_se = (
            (8.0, 6.4, 4.6) if sex == "girls" else (3.8, 7.7, 4.2)
        )
        if (
            round(measured, 1) != published_ate
            or predictions["control_1998"]["predicted_pp"] != published_forecast
        ):
            raise ValueError("Table reconstruction disagrees with published structural-model row")
        groups[sex] = {
            "attendees_control": ac,
            "eligible_control": nc,
            "attendees_treated": at,
            "eligible_treated": nt,
            "experimental_effect_pp": measured,
            "reported_experimental_se_pp": published_se,
            "predictions": predictions,
        }
    return groups


def run(cache_dir):
    path = fetch(PAPER, cache_dir, "todd-wolpin-pier-03-022.pdf", DIGEST)
    text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    tables = compare(text)
    raw = progresa_raw.run(cache_dir)
    for sex in ("girls", "boys"):
        if abs(tables[sex]["experimental_effect_pp"] - raw["groups"][sex]["effect_pp"]) > 1e-10:
            raise ValueError("Model-validation table and archived child records disagree")
    return {**tables, "raw": raw}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2))
