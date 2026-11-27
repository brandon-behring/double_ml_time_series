"""Small deterministic controls for support, attrition, calendar time and method ranking."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from temporalcv import TimeSeriesCrossValidator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from dml_ts.dml._panel import panel_standard_error  # noqa: E402
from dml_ts.dml._utils import compute_r2  # noqa: E402
from scripts.comprehensive_method_comparison import (  # noqa: E402
    best_baseline_comparison,
    rank_methods,
)


def evaluate():
    # Equal numbers in both randomization arms at every baseline risk level.
    latent, arm = np.tile(np.array([[-1, 0], [-1, 1], [1, 0], [1, 1]]), (100, 1)).T
    outcome = latent.copy()  # Y(0) and Y(1) are identical for each person.
    retained = (arm == 1) == (latent == 1)
    complete = [np.mean(outcome[(arm == a) & retained]) for a in (0, 1)]
    rates = [np.mean(retained[arm == a]) for a in (0, 1)]
    rng = np.random.default_rng(7303)
    scores = rng.normal(size=80)
    for i in range(1, len(scores)):
        scores[i] += 0.65 * scores[i - 1]
    days = np.arange(80)
    irregular = days + 300 * (days >= 40)

    def hac(d):
        se, b = panel_standard_error(scores, np.arange(len(scores)), d, "date_hac")
        return {"se": se, "bandwidth_in_observed_dates": b}

    clock = {"regular": hac(days), "gap_of_300_days": hac(irregular)}
    tr, te = next(iter(TimeSeriesCrossValidator(n_splits=4, test_size=10, gap=0).split(days)))
    tr4, te4 = next(iter(TimeSeriesCrossValidator(n_splits=4, test_size=10, gap=4).split(days)))
    h = 4
    clock["training_labels_unavailable_at_origin"] = {
        "horizon": h,
        "without_gap": int((tr + h >= te[0]).sum()),
        "with_horizon_gap": int((tr4 + h >= te4[0]).sum()),
    }
    y = np.array([8.0, 10.0, 12.0])
    pred = y - 2
    table = pd.read_csv(ROOT / "results/method_comparison/method_comparison_detailed.csv")
    previous = json.loads((ROOT / "results/method_comparison/findings.json").read_text())
    stats = previous["method_statistics"]
    old = max(
        (m for m in stats if m not in {"DML", "RandomForest", "XGBoost"}),
        key=lambda m: stats[m]["mean_abs_bias"],
    )
    best, reduction = best_baseline_comparison(stats)
    return {
        "attrition": {
            "true_effect": 0,
            "eligible": len(arm),
            "retained_by_arm": rates,
            "complete_case_effect": complete[1] - complete[0],
        },
        "clock": clock,
        "prediction": {
            "y": y.tolist(),
            "predictions": pred.tolist(),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "pooled_test_r2": compute_r2(y, pred),
            "mse_skill_over_frozen_zero": 1 - np.sum((y - pred) ** 2) / np.sum(y**2),
        },
        "archived_method_ranking": {
            "old_baseline": old,
            "old_bias_reduction_percent": 100
            * (1 - stats["DML"]["mean_abs_bias"] / stats[old]["mean_abs_bias"]),
            "corrected_baseline": best,
            "corrected_bias_reduction_percent": reduction,
            "corrected_aggregate_ranking": rank_methods(table),
        },
    }


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2, allow_nan=False))
