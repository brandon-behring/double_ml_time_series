"""Recursive dynamic g-estimation (Lewis-Syrgkanis) -- runnable example.

Demonstrates ``DynamicGEstimationDML`` recovering known period-specific blips
under time-varying confounding, where a naive pooled regression is biased; shows
the single-series adaptation; and, if EconML is installed, cross-checks the
native estimator against the reference ``DynamicDML``.

Run:
    venv/bin/python examples/example_dynamic_g_estimation.py
"""

import numpy as np

from dml_ts.dml import DynamicGEstimationDML, econml_available, fit_econml_reference
from dml_ts.validation import DynamicTreatmentDGP


def main() -> None:
    rng_seed = 7
    print("=" * 64)
    print("Recursive Dynamic G-Estimation (Lewis-Syrgkanis, 2021)")
    print("=" * 64)

    # --- Panel with treatment-dependent state (genuine time-varying confounding) ---
    dgp = DynamicTreatmentDGP(
        n_periods=3,
        theta_t=[1.0, 0.5, 1.5],
        n_units=1500,
        p=3,
        state_feedback=True,
        state_transition=0.4,
        treatment_state_coef=0.8,
        treatment_policy_coef=1.0,
        confounding_coef=1.0,
        treatment_noise=1.0,
        outcome_noise=1.0,
        random_state=rng_seed,
    )
    data = dgp.generate()
    print(f"\nPanel: {data.n_units} units x {data.n_periods} periods, p={data.p}")
    print(f"True per-period blips (total effect): {np.round(data.theta_t, 3)}")
    print(f"True cumulative effect:               {data.cumulative_effect:.3f}")

    # Nuisance choice matters: this DGP has linear confounding, so ridge nuisances are
    # correctly specified. Flexible learners ("random_forest", "gradient_boosting") are for
    # nonlinear confounding -- but on a linear DGP they add approximation error that the
    # recursive peeling amplifies, so match the learner to the confounding.
    est = DynamicGEstimationDML(model_y="ridge", model_t="ridge", n_folds=5, random_state=0)
    result = est.fit(data.Y, data.T, data.X, groups=data.groups)
    print("\n" + result.summary())
    print(
        f"\nRecovery error |theta_hat - theta|: {np.round(np.abs(result.theta_t - data.theta_t), 3)}"
    )

    # --- A naive pooled regression ignores the sequential structure and is biased ---
    y_p, t_p, _ = data.panels()
    naive, *_ = np.linalg.lstsq(
        np.column_stack([t_p, np.ones(data.n_units)]), y_p[:, -1], rcond=None
    )
    print("\nNaive pooled regression (no sequential controls):")
    print(f"  blips: {np.round(naive[:3], 3)}   vs truth {np.round(data.theta_t, 3)}  -> biased")

    # --- Single-series stationary adaptation ---
    series = DynamicTreatmentDGP(
        n_periods=2,
        theta_t=[1.2, 0.6],
        p=2,
        mode="series",
        series_length=3000,
        noise_level=0.5,
        treatment_noise=1.0,
        random_state=rng_seed,
    ).generate()
    ser_est = DynamicGEstimationDML(n_periods=2, model_y="ridge", model_t="ridge", n_folds=5)
    ser_res = ser_est.fit(series.Y, series.T, series.X, mode="series")
    print("\nSingle-series adaptation (distributed-lag g-estimation, HAC inference):")
    print(
        f"  theta_t = {np.round(ser_res.theta_t, 3)}  (truth {series.theta_t}), "
        f"se = {np.round(ser_res.se_t, 3)}"
    )

    # --- Optional EconML cross-check ---
    print("\n" + "-" * 64)
    if econml_available():
        ref = fit_econml_reference(data.Y, data.T, data.X, groups=data.groups, cv=5, random_state=0)
        print("EconML DynamicDML reference (cross-check):")
        print(f"  custom theta_t : {np.round(result.theta_t, 3)}")
        print(f"  econml theta_t : {np.round(ref.theta_t, 3)}")
        print(f"  max |custom - econml| = {np.abs(result.theta_t - ref.theta_t).max():.3f}")
    else:
        print("EconML not installed (optional). Install the reference cross-check with:")
        print('    uv pip install -e ".[full]"')
    print("=" * 64)


if __name__ == "__main__":
    main()
