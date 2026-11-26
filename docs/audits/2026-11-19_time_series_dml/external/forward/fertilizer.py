"""Check whether the earlier fertilizer adoption outcome was observable before rerandomization."""

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

SOURCE = "https://www.atai-research.org/wp-content/uploads/2016/07/aer-Nudging-Fertilizer-2011.pdf"
DIGEST = "101d953eb1bf4c4a8b7ca88d1e7ad432445da7a19e55bf3f0f332d304b119893"


def assess(text):
    body = " ".join(text.split())
    offer = re.search(r"In season 2 \(the (\d{4}) short rains\)", body)
    measured = re.search(r"post-treatment adoption data in (\d{4}) \(two seasons after", body)
    if not offer or not measured:
        raise ValueError("Both the second season and the adoption survey year are required")
    # Only years are needed: even the most permissive within-year bounds do not overlap.
    last_offer = date(int(offer[1]), 12, 31)
    first_measurement = date(int(measured[1]), 1, 1)
    return {
        "latest_possible_second_offer": last_offer.isoformat(),
        "earliest_possible_adoption_measurement": first_measurement.isoformat(),
        "adoption_definitely_unavailable_at_offer": first_measurement > last_offer,
    }


def run(cache_dir):
    path = fetch(SOURCE, cache_dir, "kenya-fertilizer-aer.pdf", DIGEST)
    body = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    result = assess(body)
    if not result["adoption_definitely_unavailable_at_offer"]:
        raise ValueError("Recheck the documented survey timing against the second season")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2))
