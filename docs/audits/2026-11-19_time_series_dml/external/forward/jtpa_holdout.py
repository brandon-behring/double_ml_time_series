"""Freeze and score the 1989Q1 JTPA adult holdout using original quarterly UI earnings."""

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402

from dml_ts.validation.effect_forecast import freeze, score  # noqa: E402

ROLLING = HERE.parents[2] / "rolling"
PROTOCOL_SHA = "56aa92f1a7a7d9a5c053318bf22c9a7b38a4079a6d52cfeaf713b1ef925b9c4d"
SOURCE_SHA = "3607617e265ec3eac11436f3f19a25e43e3ecf53ba6de6b98a9dede53cc3a76b"
BASE = "Data/expbif/expbif_stata.zip"
WAGES = "Data/expuiw/uiw_stata.zip"
ORIGIN = "1989-01-01"


def protocol():
    raw = (ROLLING / "jtpa-protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA:
        raise ValueError("JTPA protocol differs from the pre-outcome freeze")
    return json.loads(raw)


def nested_file(path, group, member):
    with ZipFile(path) as source, ZipFile(io.BytesIO(source.read(group))) as inner:
        return inner.read(member)


def load_design(path):
    columns = ["recid", "ra_dt", "ra_stat", "control", "treatmt", "adultm", "adultf", "age", "site"]
    buf = nested_file(path, BASE, "expbif.dta")
    frame = pd.read_stata(io.BytesIO(buf), columns=columns, convert_categoricals=False)
    if frame.recid.duplicated().any():
        raise ValueError("Duplicate baseline participant identifier")
    if not (
        frame.ra_stat.isin(["1", "2"]).all()
        and frame.ra_stat.eq("1").equals(frame.treatmt.eq("1"))
        and frame.ra_stat.eq("2").equals(frame.control.eq("1"))
        and frame.control.isin(["0", "1"]).all()
        and frame.treatmt.isin(["0", "1"]).all()
    ):
        raise ValueError("Unresolved random assignment coding")
    adults = frame.adultm.eq("1") | frame.adultf.eq("1")
    if (frame.adultm.eq("1") & frame.adultf.eq("1")).any():
        raise ValueError("Overlapping target groups")
    frame = frame.loc[adults].copy()
    frame["assigned_at"] = pd.to_datetime(frame.ra_dt, format="%y%m%d", errors="raise")
    frame["quarter"] = frame.assigned_at.dt.to_period("Q")
    frame["assigned"] = frame.treatmt.astype(int)
    frame["age"] = pd.to_numeric(frame.age, errors="raise")
    return frame


class WageArchive:
    """Expose ascertainment metadata and decode only explicitly allowed persons' Q+1 cells."""

    def __init__(self, path):
        self.data = nested_file(path, WAGES, "uiw.dta")
        # Pandas supplies the layout for this version-108 fixed-width Stata table.
        # Keeping an opaque byte buffer lets the issuing process select ID offsets
        # before decoding wages. A whole-column DataFrame load would unblind holdout wages.
        with StataReader(io.BytesIO(self.data)) as reader:
            reader.variable_labels()
            if reader._format_version != 108:
                raise ValueError("Unexpected UIW file layout")
            dtype = reader._dtype
            self.fields = {}
            for name in ("uearn01", "uempl01", "anflg01"):
                field = dtype.names[reader._varlist.index(name)]
                cell_type, offset = dtype.fields[field]
                if cell_type.kind != "f" or cell_type.itemsize != 4:
                    raise ValueError("Unexpected UIW source type")
                self.fields[name] = (cell_type, reader._data_location + offset)
            self.stride = dtype.itemsize
            _, self.max_valid = reader.VALID_RANGE["f"]
            self.index = reader.read(
                columns=["recid", "ardsite", "rejflag"], convert_categoricals=False
            )
        if self.index.recid.duplicated().any():
            raise ValueError("Duplicate UIW participant identifier")
        self.index["row"] = np.arange(len(self.index))

    def read_field(self, selected, name):
        rows = selected.to_numpy(dtype=np.int64)
        if ((rows < 0) | (rows >= len(self.index))).any():
            raise ValueError("Invalid UIW record index")
        cell_type, start = self.fields[name]
        view = np.ndarray(
            (len(self.index),),
            dtype=cell_type,
            buffer=self.data,
            offset=start,
            strides=(self.stride,),
        )
        values = view[rows].astype(float)
        values[values > self.max_valid] = np.nan
        return values

    def read_wages(self, selected, *, reproduce_original_issuance=False):
        values = self.read_field(selected, "uearn01")
        sentinel = values == 99999
        if not reproduce_original_issuance and sentinel.any():
            employment = self.read_field(selected, "uempl01")
            flags = self.read_field(selected, "anflg01")
            if not ((employment[sentinel] == 99) & np.isnan(flags[sentinel])).all():
                raise ValueError("Ambiguous 99999 wage marker")
            values[sentinel] = np.nan
        if (values < 0).any():
            raise ValueError("Undocumented negative quarterly earnings")
        return values


def select_cohort(design, origin, *, reporting_lag=None):
    cutoff = pd.Period(origin, freq="Q")
    if reporting_lag is None:
        return design.loc[design.quarter.eq(cutoff)].copy()
    if reporting_lag not in (1, 2):
        raise ValueError("Expected the primary or sensitivity reporting lag")
    # At the start of quarter K, the latest completed quarter is K-1.
    return design.loc[design.quarter + 1 + reporting_lag < cutoff].copy()


def prepare(selected, archive, *, reproduce_original_issuance=False):
    covered = selected.site.isin(set(archive.index.ardsite))
    frame = selected.loc[covered].merge(
        archive.index, on="recid", how="left", validate="one_to_one"
    )
    linked = frame.row.notna() & frame.recid.str.strip().ne("")
    if (frame.loc[linked, "site"] != frame.loc[linked, "ardsite"]).any():
        raise ValueError("Conflicting site between baseline and UI file")
    if not frame.loc[linked, "rejflag"].isin(["0", "1"]).all():
        raise ValueError("Undocumented UI linkage status")
    valid = linked & frame.rejflag.eq("0")
    data = frame.loc[valid].copy()
    data["earnings"] = archive.read_wages(
        data.row, reproduce_original_issuance=reproduce_original_issuance
    )
    used = data.loc[data.earnings.notna()].sort_values("recid").reset_index(drop=True)

    def arms(x):
        return {k: int(x.assigned.eq(i).sum()) for i, k in [(0, "control"), (1, "offer")]}

    counts = {
        "assigned": arms(selected),
        "covered_sites": arms(selected.loc[covered]),
        "linked": arms(frame.loc[linked]),
        "valid_link": arms(data),
        "observed": arms(used),
        "observed_zeros": arms(used.loc[used.earnings.eq(0)]),
        "site_observed": {k: int(v) for k, v in used.site.value_counts().sort_index().items()},
    }
    return used, counts


def arm_difference(data):
    by_arm = data.groupby("assigned").earnings.agg(["mean", "var", "count"])
    if set(by_arm.index) != {0, 1} or min(by_arm["count"]) < 2:
        raise ValueError("Two observations required in each assignment arm")
    return float(by_arm.loc[1, "mean"] - by_arm.loc[0, "mean"]), math.sqrt(
        float((by_arm["var"] / by_arm["count"]).sum())
    )


def augmented_effect(data):
    # Known adult assignment probability is shared across the training/evaluation arms.
    p = 2 / 3
    x = data[["site", "adultf", "age"]]
    a, y = data.assigned.to_numpy(), data.earnings.to_numpy()
    scores = np.zeros(len(data))
    folds = KFold(n_splits=5, shuffle=True, random_state=19890101)
    for tr, te in folds.split(x):
        for arm, prob, sign in [(1, p, 1), (0, 1 - p, -1)]:
            rows = tr[a[tr] == arm]
            if not len(rows):
                raise ValueError("Nuisance training fold lacks an assignment arm")
            transform = ColumnTransformer(
                [
                    (
                        "group",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["site", "adultf"],
                    ),
                    ("age", StandardScaler(), ["age"]),
                ]
            )
            model = make_pipeline(transform, Ridge(alpha=1.0, solver="svd"))
            model.fit(x.iloc[rows], y[rows])
            pred = model.predict(x.iloc[te])
            scores[te] += sign * (pred + (a[te] == arm) * (y[te] - pred) / prob)
    return float(scores.mean())


def issue(design, archive, out, *, reproduce_original_issuance=False):
    spec = protocol()
    issuances = {}
    for lag in (1, 2):
        earlier = select_cohort(design, ORIGIN, reporting_lag=lag)
        sample, counts = prepare(
            earlier, archive, reproduce_original_issuance=reproduce_original_issuance
        )
        estimates = [0.0, arm_difference(sample)[0], augmented_effect(sample)]
        forecasts = [
            {"origin": ORIGIN, "method": name, "value": round(v, 8)}
            for name, v in zip(spec["methods"], estimates, strict=True)
        ]
        config = dict(
            spec, protocol_sha256=PROTOCOL_SHA, source_sha256=SOURCE_SHA, reporting_lag_quarters=lag
        )
        if not reproduce_original_issuance:
            config["source_correction"] = (
                "Post-unblinding. Decode the joint UEARN01=99999 / UEMPL01=99 / missing ANFLG01 source marker as unavailable."
            )
        digest = freeze(frozen_path(out, lag, reproduce_original_issuance), config, forecasts)
        issuances[str(lag)] = {
            "sha256": digest,
            "training": counts,
            "latest_assignment": earlier.assigned_at.max().date().isoformat(),
        }
    return issuances


def frozen_path(out, lag, original):
    prefix = "jtpa" if original else "jtpa-corrected"
    return out / f"{prefix}-lag{lag}-frozen.json"


def evaluate(design, archive, out, issuances, *, reproduce_original_issuance=False):
    spec = protocol()
    # Verify every frozen record before accessing any outcomes for the holdout cohort.
    paths = {str(lag): frozen_path(out, lag, reproduce_original_issuance) for lag in (1, 2)}
    for lag, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != issuances[lag]["sha256"]:
            raise ValueError("JTPA issuance digest mismatch")
    sample, counts = prepare(
        select_cohort(design, ORIGIN),
        archive,
        reproduce_original_issuance=reproduce_original_issuance,
    )
    estimate, se = arm_difference(sample)
    identity = {k: spec[k] for k in ["target", "unit", "horizon", "intervention", "comparator"]}
    observation = dict(identity, origin=ORIGIN, estimate=estimate)
    rows = []
    for lag, path in paths.items():
        for row in score(path, issuances[lag]["sha256"], [observation]):
            error = row["signed_error"]
            ci = [error + sign * NormalDist().inv_cdf(0.975) * se for sign in [-1, 1]]
            eq = [error + sign * NormalDist().inv_cdf(0.95) * se for sign in [-1, 1]]
            rows.append(
                dict(
                    row,
                    reporting_lag_quarters=int(lag),
                    ci95_fixed_forecast=ci,
                    ci90_fixed_forecast=eq,
                    equivalent={str(t): bool(eq[0] > -t and eq[1] < t) for t in (50, 100, 250)},
                )
            )
    return {
        "protocol_sha256": PROTOCOL_SHA,
        "source_sha256": SOURCE_SHA,
        "issuances": issuances,
        "holdout": counts,
        "estimate": estimate,
        "se": se,
        "scores": rows,
    }


def audit_marker(path, archive):
    """Reconcile the source coding after both holdout forecasts have been scored."""
    rows = pd.Series(np.arange(len(archive.index)))
    original = archive.read_field(rows, "uearn01")
    marker = original == 99999
    registry = archive.index.loc[marker, ["recid", "ardsite"]]
    revised = nested_file(
        path, "Data/UI and SSA/Upjohn Institute Created Files/UI_and_SSA_stata.zip", "pubrawui.dta"
    )
    raw = pd.read_stata(
        io.BytesIO(revised), columns=["recid", "uiqr_001"], convert_categoricals=False
    )
    linked = registry.merge(raw, how="left", on="recid", validate="one_to_one")
    return {
        "original_uiw_rows": len(original),
        "marker_rows": int(marker.sum()),
        "number_of_employers_is_99": int((archive.read_field(rows[marker], "uempl01") == 99).sum()),
        "quarter_flag_missing": int(np.isnan(archive.read_field(rows[marker], "anflg01")).sum()),
        "sites": registry.ardsite.value_counts().sort_index().to_dict(),
        "revised_raw_quarter_missing": int(linked.uiqr_001.isna().sum()),
        "revised_raw_quarter_equals_99999": int(linked.uiqr_001.eq(99999).sum()),
    }


def run(cache, out, stage):
    spec = protocol()
    data = fetch(spec["source"], cache, "jtpa-national-evaluation.zip", SOURCE_SHA)
    baseline, wages = load_design(data), WageArchive(data)
    out.mkdir(parents=True, exist_ok=True)
    saved = ROLLING / "jtpa-issuance.json"
    if stage == "issue":
        return issue(baseline, wages, out, reproduce_original_issuance=True)
    if stage == "corrected":
        corrected = issue(baseline, wages, out)
        result = evaluate(baseline, wages, out, corrected)
        result["status"] = "Post-unblinding source-coding correction"
        return result
    expected = json.loads(saved.read_text())
    if (
        stage == "replay"
        and issue(baseline, wages, out, reproduce_original_issuance=True) != expected
    ):
        raise ValueError("Historical forecasts differ from the archived pre-score issuance")
    original = evaluate(baseline, wages, out, expected, reproduce_original_issuance=True)
    original["status"] = (
        "Sealed forecasts. The two fitted methods included a quarterly missing-value marker as earnings; zero and the later contrast are unaffected."
    )
    if stage == "score":
        return original
    corrected = issue(baseline, wages, out)
    repaired = evaluate(baseline, wages, out, corrected)
    repaired["status"] = "Post-unblinding source-coding correction"
    return {
        "sealed": original,
        "post_unblinding_correction": repaired,
        "source_audit": audit_marker(data, wages),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["issue", "score", "replay", "corrected"],
        default="replay",
        help="issue and score reproduce the original sealed analysis; corrected decodes the UIW marker; replay verifies both",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Directory holding the two frozen forecasts"
    )
    args = parser.parse_args()
    if args.stage in ("issue", "score") and args.output_dir is None:
        parser.error("issue and score require --output-dir for the frozen files")
    with TemporaryDirectory(prefix="jtpa-replay-") as tmp:
        result = run(args.cache_dir, args.output_dir or Path(tmp), args.stage)
        print(json.dumps(result, indent=2, allow_nan=False))
