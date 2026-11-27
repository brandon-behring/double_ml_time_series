"""Replay the Michigan randomized encouragement and its engineering projections."""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

HARVARD = "https://dataverse.harvard.edu/api/access/datafile"
SOURCES = {
    "data": (3107601, "zip", "9c9528bb4280dcdbf4bd73b7fb899569118a19843359577437c550e1de1b36e6"),
    "programs": (
        3107603,
        "zip",
        "3837ff6c679e80b74df239a51567d0e39cf2908561a32ff007ba19368496ffeb",
    ),
    "readme": (3107604, "docx", "2f1d5e52fbf40f64a0d646bf89c065d862f9a21d99353b940a9e8291ff3ca496"),
}
OUTCOMES = {"application": "IN_CAA", "audit": "IN_NEAT", "weatherization": "WAP"}
MEASURES = {
    "projected_discounted_bill_savings_dollars": "neat_npv_savings_cum",
    "projected_annual_heating_savings_mmbtu": "heatingmbtu",
    "installation_cost_dollars": "iwc_job_cost_act",
}
NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def study_sample(data):
    """Follow application_rate_compare.do, retaining Stata's missing-value comparison."""
    too_few_readings = data.countPOST_GAS.lt(3) & data.countPOST_ELEC.lt(3)
    return data.loc[data.IN_RED.eq(1) & data.normal.eq(1) & ~too_few_readings]


def takeup(data):
    if data.fwhhid.isna().any() or not data.fwhhid.is_unique:
        raise ValueError("Expected one record per identified household")
    if not data.RED_ENC.isin([0, 1]).all():
        raise ValueError("Expected binary encouragement assignment")
    out = {"households": len(data), "outcomes": {}}
    for name, column in OUTCOMES.items():
        # The authors explicitly code missing home audits as zero.
        outcome = data[column].fillna(0) if name == "audit" else data[column]
        if not outcome.dropna().isin([0, 1]).all():
            raise ValueError(f"Expected binary {name} status")
        control, encouraged = (
            outcome.loc[data.RED_ENC.eq(arm)].dropna().astype(float) for arm in (0, 1)
        )
        nc, nt = len(control), len(encouraged)
        if min(nc, nt) < 2:
            raise ValueError(f"Both arms need at least two observed {name} outcomes")
        effect = 100 * (encouraged.mean() - control.mean())
        vc, vt = control.var(), encouraged.var()
        pooled_variance = ((nc - 1) * vc + (nt - 1) * vt) / (nc + nt - 2)
        independent_se = 100 * sqrt(vc / nc + vt / nt)
        out["outcomes"][name] = {
            "control_n": nc,
            "encouraged_n": nt,
            "control_events": int(control.sum()),
            "encouraged_events": int(encouraged.sum()),
            "control_pct": 100 * control.mean(),
            "encouraged_pct": 100 * encouraged.mean(),
            "effect_pp": effect,
            "published_ols_se_pp": 100 * sqrt(pooled_variance * (1 / nc + 1 / nt)),
            "independent_arm_ci95_pp": [
                effect - 1.96 * independent_se,
                effect + 1.96 * independent_se,
            ],
        }
    return out


def engineering(data):
    audited = data.loc[data.WAP.eq(1) & data.neat_clientid.notna()]
    summaries = {}
    for name, column in MEASURES.items():
        observed = audited[column].dropna().astype(float)
        summaries[name] = {"n": len(observed), "mean": observed.mean(), "sd": observed.std()}
    return {"audited_weatherized_records": len(audited), **summaries}


def billing_access(data_archive, programs, readme):
    # Compare each script's required input to the authors' access table and ZIP manifest.
    document = ElementTree.fromstring(readme.read("word/document.xml"))
    permissions = {}
    for row in document.findall(".//w:tr", NAMESPACE):
        cells = ["".join(cell.itertext()).strip() for cell in row.findall("w:tc", NAMESPACE)]
        if len(cells) == 3:
            permissions[cells[0]] = cells[1]
    data_files = {Path(name).name for name in data_archive.namelist()}
    missing = []
    for script, panel in (
        ("RED_estimate.do", "RED_est.dta"),
        ("quasi_estimate.do", "QUASI_cmb_est.dta"),
    ):
        if (
            panel.encode() not in programs.read(f"program_files/{script}")
            or permissions.get(panel) != "N"
            or panel in data_files
        ):
            raise ValueError(f"Recheck billing access for {panel} against the authors' files")
        missing.append(panel)
    return {"nonpublic_monthly_panels": missing, "monthly_savings_refitted": False}


def run(cache):
    paths = {
        name: fetch(
            f"{HARVARD}/{file_id}?format=original", cache, f"weatherization-{file_id}.{ext}", digest
        )
        for name, (file_id, ext, digest) in SOURCES.items()
    }
    with (
        ZipFile(paths["data"]) as source,
        ZipFile(paths["programs"]) as programs,
        ZipFile(paths["readme"]) as readme,
    ):
        access = billing_access(source, programs, readme)
        with source.open("data_files/application_status_data.dta") as member:
            data = pd.read_stata(
                member,
                convert_categoricals=False,
                columns=[
                    "fwhhid",
                    "IN_RED",
                    "normal",
                    "RED_ENC",
                    "countPOST_GAS",
                    "countPOST_ELEC",
                    *OUTCOMES.values(),
                ],
            )
        with source.open("data_files/measures_summary.dta") as member:
            measures = pd.read_stata(
                member,
                columns=["WAP", "neat_clientid", *MEASURES.values()],
                convert_categoricals=False,
            )
    table = takeup(study_sample(data))
    outcomes = table["outcomes"]
    if table["households"] != 28888:
        raise ValueError("Estimation sample differs from published Table III")
    np.testing.assert_allclose(
        [v["effect_pp"] for v in outcomes.values()], [13, 5, 5], atol=0.5, rtol=0
    )
    np.testing.assert_allclose(
        [v["control_pct"] for v in outcomes.values()], [2, 1, 1], atol=0.5, rtol=0
    )
    forecasts = engineering(measures)
    np.testing.assert_allclose(
        forecasts["projected_discounted_bill_savings_dollars"]["mean"], 10611, atol=0.5, rtol=0
    )
    in_experiment = data.loc[data.IN_RED.eq(1)]
    return {
        "source_status_rows": len(data),
        "archived_experiment_rows": len(in_experiment),
        "duplicate_household_rows_before_filter": int(in_experiment.fwhhid.duplicated().sum()),
        "table_iii": table,
        "engineering": forecasts,
        "billing_access": access,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
