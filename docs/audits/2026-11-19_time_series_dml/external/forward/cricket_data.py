"""Cricsheet match records for the prospective randomized innings-choice benchmark."""

import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

URL = "https://cricsheet.org/downloads/t20s_json.zip"
SOURCE_SHA = "956f29df9d2b95dd6fe12de5bc70d4522f23693e5e7e2ac4c1a004e1b541ad8e"
MATCH_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) - (\S+) - (\S+) - (\S+) - (\d+) - .+$")
COLUMNS = [
    "match_id",
    "date",
    "gender",
    "focal",
    "opponent",
    "venue",
    "series",
    "status",
    "focal_toss",
    "focal_credit",
]
METHODS = ["zero", "historical", "half_history", "dml"]


def digest(path):
    with Path(path).open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def match_record(match_id, date, info):
    if (
        info["dates"][0] != date
        or info["team_type"] != "international"
        or info["match_type"] != "T20"
    ):
        raise ValueError(f"Inconsistent match metadata {match_id}")
    teams = sorted(info["teams"])
    if len(teams) != 2 or teams[0] == teams[1] or info["gender"] not in ("male", "female"):
        raise ValueError(f"Invalid teams or format in {match_id}")
    focal, opponent = teams
    event = info.get("event", {}).get("name") or " / ".join(teams)
    series = " | ".join([info["gender"], str(info["season"]), event])
    row = dict(
        zip(
            COLUMNS[:7],
            [
                str(match_id),
                date,
                info["gender"],
                focal,
                opponent,
                info.get("venue") or "unknown",
                series,
            ],
            strict=True,
        )
    )
    toss = info.get("toss") or {}
    # Exclude non-senior sides by competition/team name. Withheld Afghanistan coverage is frozen.
    if any(re.search(r"under[ -]?\d+|\bu-?19\b", name, re.I) for name in [event, *teams]):
        status = "youth"
    elif "Afghanistan" in teams:
        status = "coverage_exclusion"
    elif not toss.get("winner"):
        status = "no_toss"
    elif toss.get("uncontested"):
        status = "uncontested"
    else:
        if toss["winner"] not in teams:
            raise ValueError(f"Unknown toss winner in {match_id}")
        outcome = info["outcome"]
        win = outcome.get("winner") or outcome.get("eliminator") or outcome.get("bowl_out")
        if win is not None and win not in teams:
            raise ValueError(f"Unknown match winner in {match_id}")
        if win is None and outcome.get("result") not in ("tie", "draw", "no result"):
            raise ValueError(f"Unrecognized outcome in {match_id}")
        return dict(
            row,
            status="eligible",
            focal_toss=int(toss["winner"] == focal),
            focal_credit=float(win == focal) if win else 0.5,
        )
    return dict(row, status=status, focal_toss=None, focal_credit=None)


def read_archive(path, start, end):
    """Read outcomes only for indexed match dates in the half-open local-date interval."""
    rows = []
    with ZipFile(path) as archive:
        indexed = set()
        for line in archive.read("README.txt").decode().splitlines():
            parsed = MATCH_LINE.fullmatch(line)
            if not parsed:
                continue
            date, team_type, match_type, sex, match_id = parsed.groups()
            if match_id in indexed:
                raise ValueError(f"Duplicate source match {match_id}")
            indexed.add(match_id)
            if not start <= date < end or team_type != "international" or match_type != "T20":
                continue
            info = json.loads(archive.read(f"{match_id}.json"))["info"]
            if sex != info["gender"]:
                raise ValueError(f"Conflicting gender in {match_id}")
            rows.append(match_record(match_id, date, info))
        members = [n[:-5] for n in archive.namelist() if n.endswith(".json")]
        if not indexed or indexed != set(members) or len(members) != len(indexed):
            raise ValueError("Cricsheet README does not uniquely index every JSON match")
    return (
        pd.DataFrame(rows, columns=COLUMNS).sort_values(["date", "match_id"]).reset_index(drop=True)
    )


def archive_extract(frame):
    """Stable distributable extract of the inputs needed to reconstruct the forecasts."""
    buf = io.StringIO(newline="")
    frame.to_csv(buf, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    out = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as g:
        g.write(buf.getvalue().encode())
    return out.getvalue()


def score_contributions(frame):
    """Known-fair-toss ITT, one contribution per match in percentage points of win credit."""
    return 100 * (2 * frame.focal_toss.to_numpy() - 1) * (2 * frame.focal_credit.to_numpy() - 1)


def features(frame):
    return [
        {
            "focal_team": f"{r.gender}:{r.focal}",
            "opponent_team": f"{r.gender}:{r.opponent}",
            "venue": r.venue,
            "gender": r.gender,
            "month": r.date[5:7],
            "year_since_2015": (float(r.date[:4]) - 2015) / 10,
        }
        for r in frame.itertuples()
    ]


def forecasts(frame):
    data = frame.loc[frame.status == "eligible"].sort_values(["date", "match_id"])
    if data.series.nunique() < 5:
        raise ValueError("Five independent group folds require at least five series")
    x = features(data)
    a = data.focal_toss.to_numpy()
    y = data.focal_credit.to_numpy()
    influence = np.zeros(len(data))
    for tr, va in GroupKFold(n_splits=5).split(data, groups=data.series):
        v = DictVectorizer()
        xt = v.fit_transform([x[i] for i in tr])
        xv = v.transform([x[i] for i in va])
        for arm, sign in ((1, 1), (0, -1)):
            take = a[tr] == arm
            if not take.any():
                raise ValueError("A training fold lacks randomized treatment variation")
            model = Ridge(alpha=1.0, solver="lsqr", tol=1e-10).fit(xt[take], y[tr][take])
            mu = model.predict(xv).clip(0, 1)
            influence[va] += sign * (mu + 2 * (a[va] == arm) * (y[va] - mu))
    historical = round(float(score_contributions(data).mean()), 10)
    return {
        "zero": 0.0,
        "historical": historical,
        "half_history": historical / 2,
        "dml": round(float(100 * influence.mean()), 10),
    }


def cluster_interval(values, groups):
    n = len(values)
    mean = values.mean()
    contributions = pd.Series(values - mean).groupby(np.asarray(groups)).sum()
    clusters = len(contributions)
    if clusters < 2:
        return {"clusters": clusters, "se": None, "ci95": None}
    se = float(np.linalg.norm(contributions) / n * np.sqrt(clusters / (clusters - 1)))
    width = float(t.ppf(0.975, clusters - 1)) * se
    return {"clusters": clusters, "se": se, "ci95": [float(mean - width), float(mean + width)]}


def benchmark(frame):
    data = frame.loc[frame.status == "eligible"]
    if data.empty:
        return None
    scores = score_contributions(data)
    blocks = pd.to_datetime(data.date).sub(pd.Timestamp("2000-01-01")).dt.days // 28
    return {
        "matches": len(data),
        "estimate": float(scores.mean()),
        "series": cluster_interval(scores, data.series),
        "calendar_28d": cluster_interval(scores, blocks),
    }
