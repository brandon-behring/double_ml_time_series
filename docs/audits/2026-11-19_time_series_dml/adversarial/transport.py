"""Retrospective comparisons of within-trial adjustment and forecasting later effects."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "external"))
from benchmark_io import fetch  # noqa: E402
from forward import cricket_data, jtpa_holdout, prospective_cricket, prospective_gss  # noqa: E402


def gss_standardized(wave):
    """Fit weighted arm-specific linear regressions on the ridge model's predictors."""
    d = wave.loc[wave.eligible]
    age = (d.age.fillna(50) - 50) / 20
    x = np.column_stack(
        [age, age**2, d.age.isna(), d.sex.eq(2), d.sex.isna()]
        + [d.region.eq(k) for k in range(1, 10)]
        + [d.ballot.eq(k) for k in (2, 3)]
    )
    predictions = []
    for arm in (0, 1):
        select = d.arm.eq(arm)
        model = LinearRegression().fit(
            x[select], d.endorsed[select], sample_weight=d.wtssps[select]
        )
        predictions.append(model.predict(x))
    return 100 * float(np.average(predictions[1] - predictions[0], weights=d.wtssps))


def forecast_rmse(rows):
    return {
        name: float(
            np.sqrt(np.mean([(row["forecasts"][name] - row["target"]) ** 2 for row in rows]))
        )
        for name in rows[0]["forecasts"]
    }


def survey(cache):
    g = prospective_gss
    raw = fetch(g.URL, cache, "gss-1972-2024-3a.zip", g.SHA)
    frame = g.read_stata(raw, g.MEMBER)
    history, rows = [], []
    for year in g.WAVES:
        wave = g.decode(frame.loc[frame.year.eq(year)])
        direct = g.survey_difference(wave, wave.eligible)
        if history:
            values = np.array([h["direct"] for h in history])
            forecasts = {
                "zero": 0.0,
                "historical": float(values.mean()),
                "half_history": float(values.mean() / 2),
                "aipw": float(np.mean([h["aipw"] for h in history])),
                "standardized_ols": float(np.mean([h["ols"] for h in history])),
                "latest_wave": float(values[-1]),
                "last_three": float(values[-3:].mean()),
                "inverse_variance": float(
                    np.average(values, weights=[h["se"] ** -2 for h in history])
                ),
            }
            rows.append({"year": year, "target": direct["estimate"], "forecasts": forecasts})
        history.append(
            {
                "year": year,
                "direct": direct["estimate"],
                "se": direct["se"],
                "ols": gss_standardized(wave),
                "aipw": g.augmented(wave),
            }
        )
    return {
        "source_sha256": g.SHA,
        "history": history,
        "later_waves": rows,
        "rmse": forecast_rmse(rows),
    }


def cricket():
    data = prospective_cricket.history().query("status == 'eligible'").copy()
    data["year"] = data.date.str[:4].astype(int)
    rows = []
    for year in range(2015, 2026):
        past = data[data.year < year]
        later = data[data.year == year]
        forecasts = cricket_data.forecasts(past)
        forecasts["latest_year"] = cricket_data.benchmark(past[past.year == year - 1])["estimate"]
        observed = cricket_data.benchmark(later)
        rows.append(
            {
                "year": year,
                "matches": len(later),
                "target": observed["estimate"],
                "forecasts": forecasts,
            }
        )
    past = data[data.year < 2025]
    later = data[data.year == 2025]
    v = DictVectorizer()
    train, test = (
        v.fit_transform(cricket_data.features(past)),
        v.transform(cricket_data.features(later)),
    )

    def predict(mask):
        fit = Ridge(alpha=1, solver="lsqr", tol=1e-10).fit(
            train[mask], past.focal_credit.to_numpy()[mask]
        )
        return fit.predict(test).clip(0, 1)

    pooled = predict(np.ones(len(past), dtype=bool))
    arms = (
        predict(past.focal_toss.eq(0).to_numpy()) + predict(past.focal_toss.eq(1).to_numpy())
    ) / 2
    effects = {}
    for method, baseline in [("unadjusted", 0.5), ("pooled_ridge", pooled), ("arm_ridge", arms)]:
        residual = later.focal_credit.to_numpy() - baseline
        scores = 200 * (2 * later.focal_toss.to_numpy() - 1) * residual
        effects[method] = {
            "estimate": float(scores.mean()),
            **cricket_data.cluster_interval(scores, later.series),
            "sharp_null_randomization_sd": float(200 * np.linalg.norm(residual) / len(later)),
        }
    return {
        "source_sha256": prospective_cricket.HISTORY_SHA,
        "later_years": rows,
        "rmse": forecast_rmse(rows),
        "estimation_2025": {
            "training_matches": len(past),
            "test_matches": len(later),
            "effects": effects,
        },
    }


def training_offers(cache):
    j = jtpa_holdout
    source = fetch(j.protocol()["source"], cache, "jtpa-national-evaluation.zip", j.SOURCE_SHA)
    base, wages = j.load_design(source), j.WageArchive(source)
    training, _ = j.prepare(j.select_cohort(base, j.ORIGIN, reporting_lag=1), wages)
    later, _ = j.prepare(j.select_cohort(base, j.ORIGIN), wages)
    out = {}
    for label, data in [("history", training), ("holdout", later)]:
        strata = [(len(site), *j.arm_difference(site)) for _, site in data.groupby("site")]
        weights = np.array([s[0] for s in strata]) / len(data)
        pooled, pooled_se = j.arm_difference(data)
        out[label] = {
            "n": len(data),
            "pooled": pooled,
            "se": pooled_se,
            "site_standardized": float(weights @ [s[1] for s in strata]),
            "site_standardized_se_fixed_mix": float(
                np.linalg.norm(weights * [s[2] for s in strata])
            ),
        }
    new = later.loc[~later.site.isin(training.site)]
    return {
        "source_sha256": j.SOURCE_SHA,
        **out,
        "historical_aipw": j.augmented_effect(training),
        "later_observed_at_new_sites": new.site.value_counts().to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "gss": survey(args.cache_dir),
                "cricket": cricket(),
                "jtpa": training_offers(args.cache_dir),
            },
            indent=2,
            allow_nan=False,
        )
    )
