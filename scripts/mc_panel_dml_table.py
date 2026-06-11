"""Reproducible Monte Carlo for the chapter-6 Panel DML comparison table.

Generates the bias/SE/coverage numbers for Table `tab:panel_monte_carlo`
(naive pooled DML without fixed effects vs PanelDML with individual and
two-way FE) on a panel DGP whose individual effects are correlated with
treatment — the omitted-variable setting the table illustrates.

Run: venv/bin/python scripts/mc_panel_dml_table.py
"""

import warnings

import numpy as np

from dml_ts import PanelDML, double_ml

TRUE_TAU = 2.0
N_SIMS = 100
N_UNITS = 40
N_PERIODS = 15


def _panel(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = N_UNITS * N_PERIODS
    individual_id = np.repeat(np.arange(N_UNITS), N_PERIODS)
    time_id = np.tile(np.arange(N_PERIODS), N_UNITS)
    alpha = rng.normal(size=N_UNITS)[individual_id]  # individual effects
    X = rng.normal(size=(n, 2))
    # Treatment correlated with the individual effect -> omitting FE biases tau
    T = 0.5 * X[:, 0] + 0.8 * alpha + rng.normal(size=n)
    Y = TRUE_TAU * T + X[:, 0] + alpha + rng.normal(size=n)
    return Y, T, X, individual_id, time_id


def main() -> None:
    rows: dict[str, dict[str, list[float]]] = {
        "naive": {"theta": [], "se": [], "cover": []},
        "fe": {"theta": [], "se": [], "cover": []},
        "twoway": {"theta": [], "se": [], "cover": []},
    }
    for sim in range(N_SIMS):
        Y, T, X, ind, tid = _panel(1000 + sim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = double_ml(Y, T, X, n_folds=3, model="ridge", random_state=0)
            rows["naive"]["theta"].append(r.theta)
            rows["naive"]["se"].append(r.se)
            rows["naive"]["cover"].append(float(r.ci_lower <= TRUE_TAU <= r.ci_upper))

            for key, fe in (("fe", "individual"), ("twoway", "twoway")):
                m = PanelDML(
                    fixed_effects=fe,  # type: ignore[arg-type]
                    cluster_se=True,
                    model_y="ridge",
                    model_t="ridge",
                    random_state=0,
                )
                res = m.fit(Y, T, X, ind, tid)
                rows[key]["theta"].append(res.theta)
                rows[key]["se"].append(res.se)
                rows[key]["cover"].append(float(res.ci_lower <= TRUE_TAU <= res.ci_upper))

    print(f"true tau = {TRUE_TAU}, {N_SIMS} sims, {N_UNITS} units x {N_PERIODS} periods")
    for key, label in (
        ("naive", "Naive DML (no FE)"),
        ("fe", "Panel DML (individual FE)"),
        ("twoway", "Panel DML (two-way FE)"),
    ):
        bias = float(np.mean(rows[key]["theta"])) - TRUE_TAU
        se = float(np.mean(rows[key]["se"]))
        cover = float(np.mean(rows[key]["cover"]))
        print(f"{label:28s} bias={bias:+.3f}  avg SE={se:.3f}  coverage={cover:.0%}")


if __name__ == "__main__":
    main()
