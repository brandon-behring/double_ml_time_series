"""
Insurance/Annuity DGP Generator for DML Validation.

Generates synthetic data mimicking insurance/annuity competitor pricing dynamics
with configurable realism levels. Designed for validating time series DML methods
before applying to real competitor pricing data.

Key Features:
1. Parameterized realism levels (simple/moderate/full)
2. Realistic confounding from macroeconomic conditions
3. Competitor lag effects and strategic responses
4. Product lifecycle dynamics
5. Seasonal patterns and regulatory regime changes

DGP Levels:
- simple: Pedagogical baseline, Y = tau*T + X'beta + eps
- moderate: + macro confounding, competitor lags, AR errors
- full: + surrender rates, crediting strategies, seasonal, regulatory

References:
- Insurance pricing elasticity literature
- Dynamic competitive pricing models
- Macroeconomic controls for financial services

Usage:
    >>> from src.validation import create_insurance_dgp
    >>> dgp = create_insurance_dgp(
    ...     realism="moderate",
    ...     n_periods=120,
    ...     n_products=10,
    ...     seed=42
    ... )
    >>> Y, T, X, time_idx, product_idx, true_params = dgp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Type aliases
RealismLevel = Literal["simple", "moderate", "full"]


@dataclass
class InsuranceDGPParams:
    """True parameters of the insurance DGP for validation.

    Attributes:
        tau: True treatment effect (competitor price -> our sales)
        tau_by_product: Product-specific effects (if heterogeneous)
        beta_macro: Coefficients on macroeconomic controls
        beta_product: Product-level fixed effects
        ar_coef: AR coefficient for error autocorrelation
        seasonal_amplitude: Amplitude of seasonal component
        regime_shift: Treatment effect change after regime shift
    """

    tau: float
    tau_by_product: Optional[NDArray[np.float64]]
    beta_macro: Dict[str, float]
    beta_product: Optional[NDArray[np.float64]]
    ar_coef: float
    seasonal_amplitude: float
    regime_shift: float
    regime_shift_period: int


@dataclass
class InsuranceDGPResult:
    """Result container for generated insurance DGP data.

    Attributes:
        Y: Outcome (our sales volume or market share)
        T: Treatment (competitor rate change in bps)
        X: Control variables (firm-level + macro)
        X_macro: Macroeconomic controls subset
        time_index: Time period indices
        product_index: Product/entity identifiers
        true_params: True DGP parameters for validation
        description: Text description of DGP configuration
    """

    Y: NDArray[np.float64]
    T: NDArray[np.float64]
    X: NDArray[np.float64]
    X_macro: NDArray[np.float64]
    time_index: NDArray[np.int64]
    product_index: NDArray[np.int64]
    true_params: InsuranceDGPParams
    description: str


def _generate_macro_environment(
    n_periods: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
    """Generate synthetic macroeconomic environment.

    Simulates correlated macro variables:
    - Interest rates (Fed funds, 10-year Treasury)
    - Inflation (CPI-based)
    - GDP growth
    - Unemployment

    Args:
        n_periods: Number of time periods
        rng: Random number generator

    Returns:
        Tuple of (stacked macro array, dict of individual series)
    """
    # Fed funds rate: persistent AR(1) with occasional level shifts
    fed_funds = np.zeros(n_periods)
    fed_funds[0] = 2.0 + rng.normal(0, 0.2)
    for t in range(1, n_periods):
        # Mean reversion + random walk + occasional jumps
        mean_revert = 0.02 * (2.5 - fed_funds[t - 1])
        innovation = rng.normal(0, 0.1)
        jump = rng.choice([0, 0.25, -0.25], p=[0.95, 0.025, 0.025])
        fed_funds[t] = fed_funds[t - 1] + mean_revert + innovation + jump

    # 10-year Treasury: Fed funds + spread
    spread = 1.5 + 0.5 * rng.standard_normal(n_periods)
    spread = np.maximum(0.5, spread)  # Minimum spread
    treasury_10y = fed_funds + spread

    # Inflation: persistent with relation to rates
    inflation = np.zeros(n_periods)
    inflation[0] = 2.0 + rng.normal(0, 0.3)
    for t in range(1, n_periods):
        # Inflation responds inversely to rates (simplified)
        target = 2.0 - 0.1 * (fed_funds[t - 1] - 2.5)
        inflation[t] = 0.7 * inflation[t - 1] + 0.3 * target + rng.normal(0, 0.2)

    # GDP growth: business cycle with rate sensitivity
    gdp_growth = np.zeros(n_periods)
    gdp_growth[0] = 2.5 + rng.normal(0, 0.5)
    for t in range(1, n_periods):
        # GDP affected by rates (inverted)
        rate_effect = -0.3 * (fed_funds[t - 1] - 2.5)
        gdp_growth[t] = 0.6 * gdp_growth[t - 1] + 0.4 * (2.5 + rate_effect) + rng.normal(0, 0.4)

    # Unemployment: inverse of GDP (Okun's law style)
    unemployment = 5.0 - 0.5 * (gdp_growth - 2.5) + rng.normal(0, 0.3, n_periods)
    unemployment = np.clip(unemployment, 3.0, 10.0)

    macro_dict = {
        "fed_funds": fed_funds,
        "treasury_10y": treasury_10y,
        "inflation": inflation,
        "gdp_growth": gdp_growth,
        "unemployment": unemployment,
    }

    # Stack into array (n_periods, n_macro)
    X_macro = np.column_stack(list(macro_dict.values()))

    return X_macro, macro_dict


def _generate_competitor_treatment(
    n_periods: int,
    n_products: int,
    X_macro: NDArray[np.float64],
    macro_dict: Dict[str, NDArray[np.float64]],
    realism: RealismLevel,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate competitor rate changes (treatment variable).

    Competitor pricing depends on:
    - Interest rate environment (main driver)
    - Own past pricing (autocorrelation)
    - Strategic response to market conditions

    Args:
        n_periods: Number of time periods
        n_products: Number of products
        X_macro: Macroeconomic environment
        macro_dict: Dict of individual macro series
        realism: DGP complexity level
        rng: Random number generator

    Returns:
        Treatment array (n_periods * n_products,)
    """
    n_total = n_periods * n_products
    fed_funds = macro_dict["fed_funds"]

    if realism == "simple":
        # Simple: just random variation + rate dependence
        T_base = np.zeros((n_periods, n_products))
        for p in range(n_products):
            # Competitor rates follow fed funds + noise
            T_base[:, p] = 0.5 * (fed_funds - fed_funds.mean()) + rng.normal(
                0, 3.0, n_periods
            )  # bps scale
        return T_base.flatten()

    elif realism == "moderate":
        # Moderate: AR structure + rate sensitivity + product heterogeneity
        T_base = np.zeros((n_periods, n_products))
        for p in range(n_products):
            # Product-specific rate sensitivity
            rate_sens = 0.3 + 0.4 * rng.random()
            ar_coef = 0.2 + 0.2 * rng.random()

            T_base[0, p] = rng.normal(0, 2.0)
            for t in range(1, n_periods):
                rate_change = fed_funds[t] - fed_funds[t - 1]
                T_base[t, p] = (
                    ar_coef * T_base[t - 1, p]
                    + rate_sens * rate_change * 100  # Convert to bps
                    + rng.normal(0, 2.5)
                )
        return T_base.flatten()

    else:  # full
        # Full: strategic behavior, regime changes, competitor interactions
        T_base = np.zeros((n_periods, n_products))
        gdp = macro_dict["gdp_growth"]
        unemployment = macro_dict["unemployment"]

        for p in range(n_products):
            rate_sens = 0.3 + 0.4 * rng.random()
            ar_coef = 0.3 + 0.2 * rng.random()
            gdp_sens = 0.5 + 0.3 * rng.random()

            T_base[0, p] = rng.normal(0, 2.0)
            for t in range(1, n_periods):
                rate_change = fed_funds[t] - fed_funds[t - 1]
                gdp_effect = gdp_sens * (gdp[t] - 2.5)

                # Strategic lag: competitors react to each other (average lag)
                if t > 1 and p > 0:
                    competitor_lag = 0.1 * T_base[t - 1, :p].mean()
                else:
                    competitor_lag = 0.0

                # Recession behavior: less aggressive pricing in downturns
                recession_adj = -0.5 if unemployment[t] > 7.0 else 0.0

                T_base[t, p] = (
                    ar_coef * T_base[t - 1, p]
                    + rate_sens * rate_change * 100
                    + gdp_effect
                    + competitor_lag
                    + recession_adj
                    + rng.normal(0, 2.0)
                )

        return T_base.flatten()


def _generate_outcome(
    T: NDArray[np.float64],
    X_macro: NDArray[np.float64],
    n_periods: int,
    n_products: int,
    true_tau: float,
    realism: RealismLevel,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], InsuranceDGPParams]:
    """Generate outcome variable (our sales/market share).

    Sales depend on:
    - Competitor pricing (treatment effect)
    - Macroeconomic conditions (confounding)
    - Product fixed effects
    - Seasonal patterns
    - Regime changes (policy shifts)

    Args:
        T: Treatment variable
        X_macro: Macro controls (n_periods, n_macro)
        n_periods: Number of periods
        n_products: Number of products
        true_tau: True treatment effect
        realism: DGP complexity level
        rng: Random number generator

    Returns:
        Tuple of (outcome array, true parameters dataclass)
    """
    n_total = n_periods * n_products
    T_reshaped = T.reshape(n_periods, n_products)

    # Macro coefficients (outcome relationship)
    beta_macro = {
        "fed_funds": 0.8,  # Higher rates -> more annuity demand
        "treasury_10y": 0.3,
        "inflation": -0.5,  # Inflation hurts sales
        "gdp_growth": 0.4,  # Good economy -> more sales
        "unemployment": -0.3,  # High unemployment -> fewer sales
    }

    if realism == "simple":
        # Simple: Y = tau*T + X'beta + eps
        Y = np.zeros(n_total)

        # Treatment effect
        Y += true_tau * T

        # Macro effects (stacked for all observations)
        X_macro_expanded = np.tile(X_macro, (n_products, 1))
        for i, (key, beta) in enumerate(beta_macro.items()):
            Y += beta * X_macro_expanded[:, i]

        # IID noise
        Y += rng.normal(0, 5.0, n_total)

        params = InsuranceDGPParams(
            tau=true_tau,
            tau_by_product=None,
            beta_macro=beta_macro,
            beta_product=None,
            ar_coef=0.0,
            seasonal_amplitude=0.0,
            regime_shift=0.0,
            regime_shift_period=0,
        )

        return Y, params

    elif realism == "moderate":
        # Moderate: + product FE, AR errors, lagged treatment
        Y_base = np.zeros((n_periods, n_products))

        # Product fixed effects
        beta_product = 5.0 * rng.standard_normal(n_products)

        # AR error component
        ar_coef = 0.3
        eps = np.zeros((n_periods, n_products))
        eps[0, :] = rng.normal(0, 3.0, n_products)
        for t in range(1, n_periods):
            eps[t, :] = ar_coef * eps[t - 1, :] + rng.normal(0, 3.0, n_products)

        for p in range(n_products):
            # Contemporaneous + lagged treatment
            Y_base[:, p] = true_tau * T_reshaped[:, p]
            Y_base[1:, p] += 0.3 * true_tau * T_reshaped[:-1, p]  # Lag effect

            # Macro effects
            for i, (key, beta) in enumerate(beta_macro.items()):
                Y_base[:, p] += beta * X_macro[:, i]

            # Product fixed effect
            Y_base[:, p] += beta_product[p]

            # AR errors
            Y_base[:, p] += eps[:, p]

        params = InsuranceDGPParams(
            tau=true_tau,
            tau_by_product=None,
            beta_macro=beta_macro,
            beta_product=beta_product,
            ar_coef=ar_coef,
            seasonal_amplitude=0.0,
            regime_shift=0.0,
            regime_shift_period=0,
        )

        return Y_base.flatten(), params

    else:  # full
        # Full: everything including seasonality, regime change, heterogeneous effects
        Y_base = np.zeros((n_periods, n_products))

        # Heterogeneous treatment effects by product
        tau_by_product = true_tau + 0.3 * rng.standard_normal(n_products)

        # Product fixed effects (larger)
        beta_product = 8.0 * rng.standard_normal(n_products)

        # Seasonal pattern (quarterly)
        seasonal_amplitude = 3.0
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_periods) / 4)

        # Regime shift: treatment effect changes after period 60
        regime_shift_period = min(60, n_periods - 1)
        regime_shift = -0.2  # Effect weakens 20% after shift

        # AR error with GARCH-like heteroskedasticity
        ar_coef = 0.4
        eps = np.zeros((n_periods, n_products))
        vol = np.ones(n_periods)
        for t in range(n_periods):
            if t > 0:
                vol[t] = 0.7 * vol[t - 1] + 0.3 * np.abs(eps[t - 1, :]).mean() + 0.5
            for p in range(n_products):
                if t == 0:
                    eps[t, p] = rng.normal(0, 3.0)
                else:
                    eps[t, p] = ar_coef * eps[t - 1, p] + rng.normal(0, vol[t])

        for p in range(n_products):
            # Time-varying treatment effect (pre/post regime)
            tau_t = np.where(
                np.arange(n_periods) < regime_shift_period,
                tau_by_product[p],
                tau_by_product[p] * (1 + regime_shift),
            )

            # Contemporaneous effect
            Y_base[:, p] = tau_t * T_reshaped[:, p]

            # Lagged treatment effects (decay)
            for lag in range(1, 4):
                if lag < n_periods:
                    Y_base[lag:, p] += (0.5**lag) * tau_t[lag:] * T_reshaped[:-lag, p]

            # Macro effects
            for i, (key, beta) in enumerate(beta_macro.items()):
                Y_base[:, p] += beta * X_macro[:, i]

            # Product fixed effect + seasonal
            Y_base[:, p] += beta_product[p] + seasonal

            # Errors
            Y_base[:, p] += eps[:, p]

        params = InsuranceDGPParams(
            tau=true_tau,
            tau_by_product=tau_by_product,
            beta_macro=beta_macro,
            beta_product=beta_product,
            ar_coef=ar_coef,
            seasonal_amplitude=seasonal_amplitude,
            regime_shift=regime_shift,
            regime_shift_period=regime_shift_period,
        )

        return Y_base.flatten(), params


def _generate_firm_controls(
    n_periods: int,
    n_products: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate firm-level control variables.

    Includes:
    - Product age/maturity
    - Market share proxy
    - Distribution channel mix
    - Risk classification

    Args:
        n_periods: Number of periods
        n_products: Number of products
        rng: Random number generator

    Returns:
        Firm controls array (n_total, n_firm_vars)
    """
    n_total = n_periods * n_products

    # Product age (persistent)
    product_age = np.zeros((n_periods, n_products))
    initial_age = rng.integers(1, 20, n_products)
    for p in range(n_products):
        product_age[:, p] = initial_age[p] + np.arange(n_periods) / 12

    # Market share (slowly varying)
    market_share = np.zeros((n_periods, n_products))
    init_share = rng.dirichlet(np.ones(n_products) * 2) * 100
    for p in range(n_products):
        market_share[0, p] = init_share[p]
        for t in range(1, n_periods):
            market_share[t, p] = (
                0.95 * market_share[t - 1, p] + 0.05 * init_share[p] + rng.normal(0, 0.5)
            )

    # Distribution channel mix (binary indicators)
    has_direct = (rng.random(n_products) > 0.3).astype(float)
    has_broker = (rng.random(n_products) > 0.2).astype(float)

    # Stack
    X_firm = np.column_stack(
        [
            product_age.flatten(),
            market_share.flatten(),
            np.tile(has_direct, n_periods),
            np.tile(has_broker, n_periods),
        ]
    )

    return X_firm


def create_insurance_dgp(
    realism: RealismLevel = "moderate",
    n_periods: int = 120,
    n_products: int = 10,
    true_tau: float = -0.8,
    seed: Optional[int] = None,
) -> InsuranceDGPResult:
    """Create synthetic insurance/annuity competitor pricing DGP.

    Generates a complete dataset for validating DML methods on insurance
    pricing applications. The treatment is competitor rate changes (in bps),
    and the outcome is our sales volume/market share.

    Args:
        realism: Complexity level of the DGP
            - "simple": Y = tau*T + X'beta + eps (pedagogical)
            - "moderate": + macro confounding, AR errors, product FE
            - "full": + seasonal, regime change, heterogeneous effects
        n_periods: Number of time periods (months recommended)
        n_products: Number of product lines
        true_tau: True treatment effect (competitor price -> our sales)
            Default -0.8: 1 bps competitor rate increase -> 0.8 unit sales increase for us
        seed: Random seed for reproducibility

    Returns:
        InsuranceDGPResult containing all generated data and true parameters

    Example:
        >>> dgp = create_insurance_dgp(realism="moderate", n_periods=120, seed=42)
        >>> print(f"True effect: {dgp.true_params.tau}")
        >>> print(f"Observations: {len(dgp.Y)}")
    """
    if realism not in ("simple", "moderate", "full"):
        raise ValueError(f"realism must be 'simple', 'moderate', or 'full', got '{realism}'")
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    if n_products < 1:
        raise ValueError(f"n_products must be >= 1, got {n_products}")

    rng = np.random.default_rng(seed)
    n_total = n_periods * n_products

    # Generate macro environment
    X_macro, macro_dict = _generate_macro_environment(n_periods, rng)

    # Generate treatment (competitor rate changes)
    T = _generate_competitor_treatment(n_periods, n_products, X_macro, macro_dict, realism, rng)

    # Generate outcome (our sales)
    Y, true_params = _generate_outcome(T, X_macro, n_periods, n_products, true_tau, realism, rng)

    # Generate firm-level controls
    X_firm = _generate_firm_controls(n_periods, n_products, rng)

    # Expand macro to match panel structure
    X_macro_expanded = np.tile(X_macro, (n_products, 1))

    # Combine all controls
    X = np.column_stack([X_firm, X_macro_expanded])

    # Create indices
    time_index = np.tile(np.arange(n_periods), n_products)
    product_index = np.repeat(np.arange(n_products), n_periods)

    # Description
    desc_parts = [
        f"Insurance DGP: realism={realism}",
        f"n_periods={n_periods}, n_products={n_products}",
        f"true_tau={true_tau}",
    ]
    if realism == "moderate":
        desc_parts.append(f"AR({true_params.ar_coef:.1f}) errors")
    elif realism == "full":
        desc_parts.extend(
            [
                f"AR({true_params.ar_coef:.1f}) errors",
                f"seasonal_amplitude={true_params.seasonal_amplitude}",
                f"regime_shift={true_params.regime_shift} at t={true_params.regime_shift_period}",
            ]
        )

    return InsuranceDGPResult(
        Y=Y,
        T=T,
        X=X,
        X_macro=X_macro_expanded,
        time_index=time_index,
        product_index=product_index,
        true_params=true_params,
        description=" | ".join(desc_parts),
    )


def validate_dgp_recovery(
    realism: RealismLevel = "moderate",
    n_periods: int = 120,
    n_products: int = 10,
    true_tau: float = -0.8,
    n_sims: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Validate DGP by checking if DML can recover true treatment effect.

    Runs Monte Carlo simulation to verify:
    1. Bias is small (< 10% of true effect)
    2. Coverage is close to nominal (93-97% for 95% CI)
    3. Standard errors are correctly calibrated

    Args:
        realism: DGP complexity level
        n_periods: Periods per simulation
        n_products: Products per simulation
        true_tau: True effect to recover
        n_sims: Number of Monte Carlo simulations
        seed: Base random seed

    Returns:
        Dict with bias, rmse, coverage, avg_se, empirical_se

    Note:
        Requires DynamicDML and PanelDML to be available.
    """
    # Import here to avoid circular dependency
    try:
        from src.dml import DynamicDML, PanelDML
    except ImportError:
        raise ImportError("DynamicDML/PanelDML required for validation")

    estimates = []
    ses = []
    covers = []

    for sim in range(n_sims):
        dgp = create_insurance_dgp(
            realism=realism,
            n_periods=n_periods,
            n_products=n_products,
            true_tau=true_tau,
            seed=seed + sim,
        )

        # Use PanelDML with fixed effects
        model = PanelDML(
            fixed_effects="individual",
            cluster_se=True,
            model_y="ridge",
            model_t="ridge",
        )

        try:
            result = model.fit(dgp.Y, dgp.T, dgp.X, dgp.product_index, dgp.time_index)
            estimates.append(result.theta)
            ses.append(result.se)
            covers.append(result.ci_lower <= true_tau <= result.ci_upper)
        except Exception:
            # Skip failed simulations
            continue

    if len(estimates) < n_sims * 0.5:
        raise RuntimeError(f"Too many simulation failures: {n_sims - len(estimates)}/{n_sims}")

    estimates_arr: np.ndarray = np.array(estimates)
    ses_arr: np.ndarray = np.array(ses)

    return {
        "bias": float(np.mean(estimates_arr) - true_tau),
        "rmse": float(np.sqrt(np.mean((estimates_arr - true_tau) ** 2))),
        "coverage": float(np.mean(covers)),
        "avg_se": float(np.mean(ses_arr)),
        "empirical_se": float(np.std(estimates_arr)),
        "n_successful": len(estimates_arr),
    }
