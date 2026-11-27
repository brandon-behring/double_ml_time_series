"""Village-randomized attendance benchmark from the original Todd–Wolpin AER archive."""

import argparse
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from benchmark_io import fetch  # noqa: E402

ARCHIVE = (
    "https://web.archive.org/web/20150921083256id_/"
    "https://www.aeaweb.org/aer/data/dec06/20031016_data.zip"
)
DIGEST = "e46291e04412d8916e707d5a29dfd75140d2864b7ed60b3511a6fbfebcd66e0f"
LOCATION = ["entidad1", "mpio", "local"]


def contrasts(frame):
    sample = frame.loc[
        frame.year.eq(98)
        & frame.pobre.eq(1)
        & frame.agefin.between(12, 15)
        & frame.inschoolr.isin([0, 1])
    ]
    if sample.empty or sample.groupby(LOCATION).control.nunique().max() != 1:
        raise ValueError("Expected one randomized assignment for each represented locality")
    result = {}
    for code, sex in [(0, "girls"), (1, "boys")]:
        s = sample.loc[sample.sexr.eq(code)]
        assigned = s.control.eq(1)
        if not set(s.control.unique()) == {1, 2}:
            raise ValueError("Expected both village assignment arms for each sex")
        n1, n0 = assigned.sum(), (~assigned).sum()
        m1, m0 = s.inschoolr[assigned].mean(), s.inschoolr[~assigned].mean()
        effect = m1 - m0
        scores = np.where(assigned, (s.inschoolr - m1) / n1, -(s.inschoolr - m0) / n0)
        clustered = s.assign(score=scores).groupby(LOCATION).score.sum()
        g, n = len(clustered), len(s)
        se = np.linalg.norm(clustered) * np.sqrt(g / (g - 1) * (n - 1) / (n - 2))
        delta = t.ppf(0.975, g - 1) * se
        result[sex] = {
            "attending_treated": int(s.inschoolr[assigned].sum()),
            "n_treated": int(n1),
            "attending_control": int(s.inschoolr[~assigned].sum()),
            "n_control": int(n0),
            "villages": g,
            "effect_pp": float(100 * effect),
            "village_se_pp": float(100 * se),
            "village_ci95_pp": [float(100 * (effect - delta)), float(100 * (effect + delta))],
        }
    return result


def fortran_check(programs, expected):
    """Compile the authors' ASCII reader and their documented original subsidy constraint."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        prefix = "programs/simulation/"
        (root / "fin2a7v13c.raw").write_bytes(programs.read(prefix + "fin2a7v13c.raw"))
        reader = (
            programs.read(prefix + "getdata2.f")
            .decode()
            .replace("/home/ptodd/simul2/fin2a7v13c.raw", "fin2a7v13c.raw")
        )
        (root / "reader.f90").write_text(reader)
        (root / "constraint.f90").write_bytes(programs.read("programs/estimation/constraint1.f"))
        subprocess.run(
            [
                "gfortran",
                "-std=legacy",
                "-ffree-line-length-none",
                "-o",
                "raw-check",
                "reader.f90",
                "constraint.f90",
                str(HERE.with_name("progresa_raw_check.f90")),
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
        lines = subprocess.check_output([str(root / "raw-check")], cwd=root, text=True).splitlines()
        counts = list(map(int, lines[0].split()))
        intended = [19500]
        for sex in ["girls", "boys"]:
            g = expected[sex]
            intended.extend(
                g[key]
                for key in ["attending_treated", "n_treated", "attending_control", "n_control"]
            )
        np.testing.assert_array_equal(counts, intended)
        grants = list(map(float, lines[1].split()))
        np.testing.assert_allclose(grants, [1890 / 1.083, 1800 / 1.083], rtol=2e-7)
    return {
        "ascii_rows": counts[0],
        "original_grade_seven_subsidy_model_pesos": dict(
            zip(["girls", "boys"], grants, strict=True)
        ),
    }


def run(cache_dir, check_fortran=False):
    path = fetch(ARCHIVE, cache_dir, "todd-wolpin-20031016_data.zip", DIGEST)
    with (
        ZipFile(path) as outer,
        ZipFile(BytesIO(outer.read("DATAFILES.zip"))) as data,
        ZipFile(BytesIO(outer.read("programs.zip"))) as programs,
    ):
        train = pd.read_stata(BytesIO(data.read("fin2a7v11c.dta")), convert_categoricals=False)
        full = pd.read_stata(BytesIO(data.read("fin2a7v13c.dta")), convert_categoricals=False)
        groups = contrasts(full)
        for sex, expected in [("girls", [166, 223, 117, 176]), ("boys", [200, 262, 132, 182])]:
            np.testing.assert_array_equal(
                [
                    groups[sex][key]
                    for key in ["attending_treated", "n_treated", "attending_control", "n_control"]
                ],
                expected,
            )
        common = [*LOCATION, "folio", "numero", "year"]
        original = full.loc[~(full.year.eq(98) & full.control.eq(1))]
        left = train.set_index(common).sort_index()
        right = original.set_index(common).sort_index()
        if not left.index.is_unique or not right.index.is_unique:
            raise ValueError("Expected a unique child per household and survey wave")
        pd.testing.assert_index_equal(left.index, right.index)
        differences = (left != right).sum()
        changed = {name: int(n) for name, n in differences.items() if n}
        if changed != {"agedad": 3872, "flag98": 425}:
            raise ValueError(f"Unexpected changes to common child-wave fields: {changed}")
        report = {
            "estimation_rows": len(train),
            "estimation_1998_controls": int((train.year.eq(98) & train.control.eq(2)).sum()),
            "added_1998_treatment_rows": len(full) - len(train),
            "differing_fields_in_common_rows": changed,
            "groups": groups,
        }
        if check_fortran:
            report["fortran"] = fortran_check(programs, groups)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--check-fortran",
        action="store_true",
        help="Compile the original data reader and original policy budget check using gfortran.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir, args.check_fortran), indent=2, allow_nan=False))
