"""Issue and conditionally score a prospective GSS questionnaire-form effect forecast."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, date, datetime
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402

from dml_ts.validation.effect_forecast import freeze, ranking_precision, score  # noqa: E402

RECORDS = HERE.parents[2] / "prospective"
URL = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"
SHA = "d4fd4e5057d53d6023e55d5114685430321c3ab00ef2bc11099c2358efc65dfe"
ISSUED_SHA256 = "163640289ef6a76d3537e9240fd8389435df846fb500e11fbba30994ad5b831c"
ISSUED_UTC = "2026-11-21T02:18:33+00:00"
MEMBER = "GSS_stata/gss7224_r3a.dta"
POLICY = "assistance-to-the-poor form minus welfare form, recorded too-little endorsement"
UNIT = "percentage points"
HORIZON = "same 2028 GSS cross-sectional interview"
METHODS = ["zero", "historical", "half_history", "dml"]
WAVES = (2000, 2004, 2006, 2008, 2014, 2021, 2022, 2024)
ERRATA = (2002, 2010, 2012, 2016, 2018)
DESIGN = ["year", "id", "ballot", "form", "wtssps", "vstrat", "vpsu"]
ITEMS = ["natfare", "natfarey"]
COVARS = ["age", "sex", "region"]
CHECKS = (
    "random_half_xy",
    "same_form_wordings",
    "same_ballots_and_population",
    "same_response_codes",
    "compatible_survey_weights_and_variance_design",
)
PENDING = "Awaiting the 2028 questionnaire, verified assignment design, and released responses."
# The public-use labels span two attested English instruments. See gss.md.
ASSISTANCE_WORDINGS = {"assistance to the poor", "assistance for the poor"}


def read_stata(path, member=None, *, covariates=True):
    """Retain Stata's tagged item missing codes to distinguish routing from nonresponse."""
    cols = DESIGN + ITEMS + (COVARS + ["wtssnrps"] if covariates else [])

    def read(file):
        with pd.read_stata(file, iterator=True) as reader:
            available = reader.variable_labels()
            selected = [
                col
                for col in cols
                if covariates or col not in ("vpsu", "vstrat") or col in available
            ]
            return reader.read(columns=selected, convert_categoricals=False, convert_missing=True)

    if member is None:
        return read(path)
    with ZipFile(path) as archive, archive.open(member) as source, NamedTemporaryFile() as f:
        shutil.copyfileobj(source, f)
        f.flush()
        return read(f.name)


def decode(frame):
    """One response per assigned A/B/C respondent, with the full design kept for domain SEs."""
    wave = frame.copy().reset_index(drop=True)
    for optional in ("vstrat", "vpsu"):
        if optional not in wave:
            wave[optional] = np.nan
    for col in DESIGN + [col for col in COVARS if col in wave]:
        wave[col] = pd.to_numeric(wave[col], errors="coerce")
    if (
        wave.year.nunique() != 1
        or wave.year.isna().any()
        or wave.id.isna().any()
        or wave.id.duplicated().any()
    ):
        raise ValueError("Expected unique respondents in exactly one wave")
    if not wave.ballot.isin([1, 2, 3, 4]).all():
        raise ValueError("Unverified ballot codes or missing ballot")
    domain = wave.ballot.isin([1, 2, 3])
    if not wave.loc[domain, "form"].isin([1, 2]).all() or not domain.any():
        raise ValueError("Unverified form codes or empty target population")
    if not np.isfinite(wave.wtssps).all() or (wave.wtssps <= 0).any():
        raise ValueError("Unusable survey weights")
    # The original and alternate items are mutually exclusive by design.
    first = wave.form.eq(1)

    def item_code(value):
        if isinstance(value, Real) and value in (1, 2, 3):
            return str(int(value))
        return str(value)

    assigned = wave.natfare.where(first, wave.natfarey).map(item_code)
    other = wave.natfarey.where(first, wave.natfare).astype(str)
    valid = assigned.isin(["1", "2", "3"])
    if (
        not (valid | assigned.isin([".d", ".n", ".s"]))[domain].all()
        or not other[domain].eq(".i").all()
    ):
        raise ValueError(
            "Incompatible assignment routing, response code, or structural missingness"
        )
    wave["eligible"] = domain
    wave["arm"] = wave.form.eq(2).astype(int)
    wave["valid"] = valid & domain
    wave["endorsed"] = assigned.eq("1").astype(float)
    return wave


def survey_difference(wave, domain):
    """Hájek arm contrast and with-replacement Taylor variance across PSUs within strata."""
    influence = pd.Series(0.0, index=wave.index)
    point = 0.0
    sizes, nonresponses = [], []
    for arm, sign in [(0, -1), (1, 1)]:
        used = domain & wave.arm.eq(arm)
        if not used.any():
            raise ValueError("One questionnaire form has no eligible respondents")
        w = wave.loc[used, "wtssps"]
        rate = np.average(wave.loc[used, "endorsed"], weights=w)
        point += sign * rate
        influence.loc[used] = sign * w / w.sum() * (wave.loc[used, "endorsed"] - rate)
        sizes.append(int(used.sum()))
        nonresponses.append(int((used & ~wave.valid).sum()))
    units = influence.groupby([wave.vstrat, wave.vpsu]).sum()
    strata = units.groupby(level=0)
    n = strata.size()
    # PSUs with zero eligible respondents appear via their zero influence above.
    df = int((n - 1).sum())
    psus, n_strata = len(units), len(n)
    reason = None
    if not np.isfinite(wave[["vstrat", "vpsu"]]).all().all():
        se = ci = df = psus = n_strata = None
        reason = "Missing public variance stratum or PSU identifiers"
    elif (n < 2).any() or df <= 0:
        se = ci = None
        reason = "Insufficient independent PSUs within a variance stratum"
    else:
        se = 100 * float(np.sqrt((n * strata.var(ddof=1)).sum()))
        radius = t.ppf(0.975, df) * se
        ci = [float(100 * point - radius), float(100 * point + radius)]
    return {
        "estimate": float(100 * point),
        "se": se,
        "ci95": ci,
        "design_df": df,
        "psus": psus,
        "strata": n_strata,
        "interval_reason": reason,
        "welfare_n": sizes[0],
        "assistance_n": sizes[1],
        "welfare_nonresponse": nonresponses[0],
        "assistance_nonresponse": nonresponses[1],
    }


def augmented(wave):
    """Hold out complete variance PSUs when estimating arm-specific outcome means."""
    data = wave.loc[wave.eligible].sort_values("id")
    age = ((data.age.fillna(50) - 50) / 20).to_numpy()
    x = np.column_stack(
        [
            age,
            age**2,
            data.age.isna(),
            data.sex.eq(2),
            data.sex.isna(),
            *[data.region.eq(k) for k in range(1, 10)],
            *[data.ballot.eq(k) for k in (2, 3)],
        ]
    )
    y = data.endorsed.to_numpy()
    a = data.arm.to_numpy()
    w = data.wtssps.to_numpy()
    groups = data.groupby(["vstrat", "vpsu"], sort=True).ngroup().to_numpy()
    preds = np.empty((len(y), 2))
    for train, test in GroupKFold(n_splits=5).split(x, groups=groups):
        for k in (0, 1):
            select = train[a[train] == k]
            if not len(select):
                raise ValueError("Nuisance training fold lacks an arm")
            model = Ridge(alpha=1, solver="svd").fit(
                x[select], y[select], sample_weight=w[select] / w[select].mean()
            )
            preds[test, k] = model.predict(x[test])
    mu0, mu1 = preds.T
    pseudo = mu1 - mu0 + 2 * a * (y - mu1) - 2 * (1 - a) * (y - mu0)
    return float(100 * np.average(pseudo, weights=w))


def history(cache):
    path = fetch(URL, cache, "gss-1972-2024-3a.zip", SHA)
    all_rows = read_stata(path, MEMBER)
    if (int(all_rows.year.min()), int(all_rows.year.max())) != (1972, 2024):
        raise ValueError("Incorrect source vintage")
    rows = []
    for year in WAVES:
        wave = decode(all_rows.loc[all_rows.year.eq(year)])
        direct = survey_difference(wave, wave.eligible)
        valid = survey_difference(wave, wave.eligible & wave.valid)
        nrps = pd.to_numeric(wave.wtssnrps, errors="coerce")
        adjusted = None
        if nrps.notna().all() and (nrps > 0).all():
            adjusted = survey_difference(wave.assign(wtssps=nrps), wave.eligible)["estimate"]
        rows.append(
            {
                "year": year,
                **direct,
                "valid_only_pp": valid["estimate"],
                "nonresponse_weight_pp": adjusted,
                "aipw_pp": augmented(wave),
            }
        )
    return rows


def make_protocol(issued_at):
    return {
        "study": "GSS 2028 cross-sectional wording experiment",
        "issued_at_utc": issued_at,
        "origins": [issued_at[:10]],
        "methods": METHODS,
        "comparators": ["zero", "historical", "half_history"],
        "primary": "dml vs half_history",
        "target": POLICY,
        "unit": UNIT,
        "horizon": HORIZON,
        "population": "Cross-sectional respondents assigned forms X/Y on standard ballots A/B/C",
        "intervention": "Form Y with assistance to the poor",
        "comparator": "Form X with welfare",
        "endpoint": "Recorded too-little response among all eligible assigned respondents; nonresponse is non-endorsement",
        "sources": {
            "history_url": URL,
            "history_sha256": SHA,
            "release": "3a, July 2026",
            "member": MEMBER,
        },
        "history_waves": list(WAVES),
        "known_assignment_errors_excluded": list(ERRATA),
        "fit": "Equal prior-wave mean of WTSSPS-weighted arm contrasts; DML averages five PSU-fold known-p=0.5 ridge-AIPW contrasts, alpha=1. Age, age squared, sex, region, ballot and missing indicators.",
        "scoring": "Later WTSSPS Hájek arm difference, Taylor variance using VSTRAT/VPSU and design-t 95% interval. Missing design support yields a point only.",
        "compatibility_checks": list(CHECKS),
        "exploratory_accuracy_margins_pp": [2, 5, 10],
        "decision_threshold": None,
    }


def forecast_rows(rows, origin):
    h = round(float(np.mean([r["estimate"] for r in rows])), 9)
    dml = round(float(np.mean([r["aipw_pp"] for r in rows])), 9)
    return [
        {"origin": origin, "method": name, "value": val}
        for name, val in zip(METHODS, [0.0, h, h * 0.5, dml], strict=True)
    ]


def issue_or_replay(cache, records, *, issue=False):
    frozen = records / "gss-frozen.json"
    issued_at = ISSUED_UTC
    if not issue:
        score(frozen, ISSUED_SHA256, [])
    protocol = make_protocol(issued_at)
    rows = history(cache)
    forecasts = forecast_rows(rows, issued_at[:10])
    digest = freeze(frozen, protocol, forecasts)
    issuance_path = records / "gss-issuance.json"
    if digest != ISSUED_SHA256:
        raise ValueError("Original GSS issuance differs")
    if issue and not issuance_path.exists():
        with issuance_path.open("x") as f:
            json.dump({"issued_at_utc": issued_at, "sha256": digest}, f, indent=2)
            f.write("\n")
    if json.loads(issuance_path.read_text()) != {"sha256": digest, "issued_at_utc": issued_at}:
        raise ValueError("Original GSS issuance differs")
    result = {
        "issuance_sha256": digest,
        "status": "pending",
        "reason": PENDING,
        "rows": score(frozen, digest, []),
        "dml_vs_half": ranking_precision(forecasts[3]["value"], forecasts[2]["value"]),
        "historical_wave_ses_pp": [r["se"] for r in rows],
    }
    if issue:
        table = pd.DataFrame(rows)
        table[["ci95_low", "ci95_high"]] = pd.DataFrame(table.pop("ci95").tolist())
        table["prior_history_pp"] = table.estimate.expanding().mean().shift(1)
        table["prior_half_pp"] = table.prior_history_pp / 2
        table["prior_dml_pp"] = table.aipw_pp.expanding().mean().shift(1)
        table.to_csv(records / "gss-history.csv", index=False, float_format="%.10g")
        (records / "gss-pending.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    return result


def assess(records, review, data=None, *, today=None):
    """Require documentary design review before reading the future wave's outcomes."""
    today = today or datetime.now(UTC).date()
    manifest = records / "gss-frozen.json"
    receipt = json.loads((records / "gss-issuance.json").read_text())
    sha = ISSUED_SHA256
    if receipt != {"sha256": sha, "issued_at_utc": ISSUED_UTC}:
        raise ValueError("GSS issuance differs from pinned SHA-256")
    score(manifest, sha, [])  # Verify the issuance before accepting the review.
    p = json.loads(manifest.read_text())["protocol"]
    identity = {k: p[k] for k in ("target", "unit", "horizon", "intervention", "comparator")}
    identity["origin"] = p["origins"][0]

    def unavailable(status, reason):
        rows = score(manifest, sha, [{**identity, "estimate": None, "reason": reason}])
        for row in rows:
            row["status"] = status
            if status == "incompatible":
                for comparison in row["comparisons"].values():
                    if comparison["status"] == "pending":
                        comparison["status"] = "incompatible"
        return {"status": status, "reason": reason, "rows": rows, "review": review, "sha256": sha}

    if review.get("wave") != 2028:
        raise ValueError("Expected the frozen 2028 wave")
    if review.get("status") == "pending":
        return unavailable("pending", review.get("reason") or PENDING)
    if not review.get("design_source") or not review.get("reviewer") or not review.get("reason"):
        raise ValueError("Supply the design document, review attribution and reasoning")
    checks = review.get("checks", {})
    if set(checks) != set(CHECKS) or any(
        v is not None and not isinstance(v, bool) for v in checks.values()
    ):
        raise ValueError("Record every protocol compatibility check")
    if review.get("status") == "incompatible" and False in checks.values():
        return unavailable("incompatible", review["reason"])
    if review.get("status") != "compatible" or not all(checks.values()):
        raise ValueError("Design review status conflicts with compatibility checks")
    if not review.get("data_source"):
        raise ValueError("Supply the public data release citation")
    for field in (
        "questionnaire_source",
        "english_form_x",
        "english_form_y",
        "context_and_languages",
    ):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError("Document fielded wordings, earlier form context and translations")
    if (
        review["english_form_x"].strip().casefold() != "welfare"
        or review["english_form_y"].strip().casefold() not in ASSISTANCE_WORDINGS
    ):
        return unavailable(
            "incompatible", "Fielded English item differs from both historical versions"
        )
    start, end, release = [
        date.fromisoformat(review[k]) for k in ("fieldwork_start", "fieldwork_end", "released")
    ]
    if not date.fromisoformat(p["origins"][0]) < start <= end <= release <= today:
        raise ValueError("Future, invalid or pre-issuance fieldwork/release dates")
    if data is None:
        raise ValueError("A released public survey file is required")
    with Path(data).open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != review.get("sha256"):
            raise ValueError("Future release SHA-256 mismatch")
    try:
        df = read_stata(data, review.get("stata_member"), covariates=False)
        wave = decode(df.loc[df.year.eq(2028)])
        result = survey_difference(wave, wave.eligible)
    except (KeyError, ValueError) as exc:
        return unavailable("incompatible", str(exc))
    observation = {
        **identity,
        "estimate": result["estimate"],
        "ci95": result["ci95"],
    }
    scores = score(manifest, sha, [observation])
    for row in scores:
        row["equivalence_sensitivity"] = None
        if result["se"] is not None:
            width = t.ppf(0.95, result["design_df"]) * result["se"]
            error = row["signed_error"]
            row["equivalence_sensitivity"] = {
                str(m): bool(abs(error) + width < m) for m in p["exploratory_accuracy_margins_pp"]
            }
    valid_domain = wave.eligible & wave.valid
    valid_result = (
        survey_difference(wave, valid_domain)
        if wave.loc[valid_domain, "arm"].nunique() == 2
        else {"estimate": None, "reason": "One form has no substantive responses"}
    )
    return {
        "status": "scored",
        "sha256": sha,
        "review": review,
        "benchmark": result,
        "valid_responses_only": valid_result,
        "rows": scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--records", type=Path, default=RECORDS)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--issue", action="store_true")
    stages.add_argument("--review", type=Path, help="2028 design review JSON")
    parser.add_argument(
        "--future-data", type=Path, help="Public 2028 Stata file or ZIP identified by the review"
    )
    args = parser.parse_args()
    if args.future_data and not args.review:
        parser.error("--future-data requires --review")
    result = (
        assess(args.records, json.loads(args.review.read_text()), args.future_data)
        if args.review
        else issue_or_replay(args.cache_dir, args.records, issue=args.issue)
    )
    print(json.dumps(result, indent=2, allow_nan=False))
