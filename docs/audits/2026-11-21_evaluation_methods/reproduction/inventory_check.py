"""Join human-reviewed application rows to the executable audit registry."""

import csv
import re
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent / "2026-11-19_time_series_dml"


def crosswalk():
    registry = runpy.run_path(str(OLD / "external/run.py"))
    suites = ("REPLICATIONS", "QUESTIONS", "FORWARD", "ROLLING", "PROSPECTIVE", "METHOD_VALIDATION")
    records = {name: [] for name in registry["STUDIES"]}
    families = []
    other = False
    for line in (HERE.parent / "study-inventory.md").read_text().splitlines():
        if line.startswith("## Other scientific"):
            other = True
        if not line.startswith("| "):
            continue
        label = line.split("|")[1].strip()
        if " · " in label:
            name, entries = label.split(" · ", 1)
            families.append(name)
        elif other:
            name, entries = line.split("|")[2].strip().split(".")[0], label
        else:
            continue
        for key in re.findall(r"`([a-z0-9-]+)`", entries):
            records[key].append(name)
    if not all(records.values()) or len(set(families)) != len(families):
        raise ValueError(f"Unmapped commands or duplicate families: {records}")
    return [
        (suite.lower(), name, " / ".join(records[name]))
        for suite in suites
        for name in registry[suite]
    ], families


if __name__ == "__main__":
    rows, families = crosswalk()
    with (HERE / "registry-crosswalk.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "command", "applications_or_simulation"])
        writer.writerows(rows)
    print(f"{len(families)} application families, {len(rows)} runnable commands")
