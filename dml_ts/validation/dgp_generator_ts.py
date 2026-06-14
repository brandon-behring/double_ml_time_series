"""
Time Series Data Generating Process (DGP) Generator.

Generates synthetic time series data with known treatment effects and controllable
temporal dependence for validating time series Double ML estimators.

Key features:
- AR(p) treatment dynamics
- VAR confounders with cross-sectional dependencies
- Panel DGP with fixed effects
- Autocorrelated errors
- Known dynamic and cumulative treatment effects
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class TimeSeriesDGPResult:
    """Result from time series DGP generation.

    Attributes:
        Y: Outcome variable (n_periods,) or (n_units, n_periods)
        T: Treatment variable (n_periods,) or (n_units, n_periods)
        X: Confounders (n_periods, p) or (n_units, n_periods, p)
        true_effect: Contemporaneous treatment effect
        true_cumulative_effect: Total effect including lags
        n_periods: Number of time periods
        n_units: Number of units (1 for single time series)
        p: Number of confounders
        time_index: Time index array
        unit_ids: Unit identifier array (for panel)
        ar_coefs: Treatment AR coefficients used
        var_coefs: Confounder VAR coefficient matrix used
        error_ar_coef: Error autocorrelation coefficient
    """

    Y: NDArray[np.float64]
    T: NDArray[np.float64]
    X: NDArray[np.float64]
    true_effect: float
    true_cumulative_effect: float
    n_periods: int
    n_units: int
    p: int
    time_index: NDArray[np.int64]
    unit_ids: NDArray[np.int64] | None = None
    ar_coefs: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    var_coefs: NDArray[np.float64] | None = None
    error_ar_coef: float = 0.0


class TimeSeriesDGPGenerator:
    """Generate synthetic time series data with known temporal properties.

    This generator creates data where:
    - Treatment follows AR(p) dynamics with dependence on confounders
    - Confounders follow VAR(1) process with controllable persistence
    - Outcome has contemporaneous and lagged treatment effects
    - Errors can be autocorrelated (AR(1))
    - True effects are known by construction

    Examples:
        >>> # Simple AR(1) treatment DGP
        >>> dgp = TimeSeriesDGPGenerator(
        ...     n_periods=200, p=3, true_effect=2.0,
        ...     treatment_ar_order=1, random_state=42
        ... )
        >>> result = dgp.generate()
        >>> result.Y.shape
        (200,)

        >>> # Panel DGP with fixed effects
        >>> dgp = TimeSeriesDGPGenerator(
        ...     n_periods=100, n_units=50, p=5, true_effect=1.5,
        ...     panel_fixed_effects=True, random_state=42
        ... )
        >>> result = dgp.generate()
        >>> result.Y.shape
        (50, 100)
    """

    def __init__(
        self,
        n_periods: int,
        p: int,
        true_effect: float,
        n_units: int = 1,
        treatment_ar_order: int = 1,
        treatment_ar_coefs: NDArray[np.float64] | None = None,
        confounder_var_coef: float = 0.5,
        confounding_strength: float = 1.0,
        lagged_effect_coefs: NDArray[np.float64] | None = None,
        error_ar_coef: float = 0.0,
        noise_level: float = 1.0,
        panel_fixed_effects: bool = False,
        treatment_type: Literal["continuous", "binary"] = "continuous",
        random_state: int | None = None,
    ):
        """Initialize time series DGP generator.

        Args:
            n_periods: Number of time periods (must be > treatment_ar_order + 10)
            p: Number of confounders (must be positive)
            true_effect: Contemporaneous treatment effect
            n_units: Number of units for panel (1 for single time series)
            treatment_ar_order: AR order for treatment dynamics
            treatment_ar_coefs: Custom AR coefficients (if None, generates random)
            confounder_var_coef: Diagonal VAR(1) coefficient for confounders (0-1)
            confounding_strength: Strength of confounding (0 = none, higher = more)
            lagged_effect_coefs: Coefficients for lagged treatment effects
            error_ar_coef: AR(1) coefficient for error autocorrelation (-1, 1)
            noise_level: Standard deviation of noise terms
            panel_fixed_effects: Whether to include unit fixed effects
            treatment_type: 'continuous' or 'binary'
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if n_periods <= treatment_ar_order + 10:
            raise ValueError(
                f"n_periods ({n_periods}) must be > treatment_ar_order + 10 "
                f"({treatment_ar_order + 10})"
            )
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if n_units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")
        if not -1 < confounder_var_coef < 1:
            raise ValueError(f"confounder_var_coef must be in (-1, 1), got {confounder_var_coef}")
        if not -1 < error_ar_coef < 1:
            raise ValueError(f"error_ar_coef must be in (-1, 1), got {error_ar_coef}")
        if treatment_type not in ["continuous", "binary"]:
            raise ValueError(
                f"treatment_type must be 'continuous' or 'binary', got {treatment_type}"
            )

        self.n_periods = n_periods
        self.p = p
        self.true_effect = true_effect
        self.n_units = n_units
        self.treatment_ar_order = treatment_ar_order
        self.confounder_var_coef = confounder_var_coef
        self.confounding_strength = confounding_strength
        self.error_ar_coef = error_ar_coef
        self.noise_level = noise_level
        self.panel_fixed_effects = panel_fixed_effects
        self.treatment_type = treatment_type
        self.random_state = random_state

        # Initialize random number generator
        self._rng = np.random.default_rng(random_state)

        # Set up AR coefficients for treatment
        if treatment_ar_coefs is not None:
            if len(treatment_ar_coefs) != treatment_ar_order:
                raise ValueError(
                    f"treatment_ar_coefs length ({len(treatment_ar_coefs)}) "
                    f"must match treatment_ar_order ({treatment_ar_order})"
                )
            self.treatment_ar_coefs = np.asarray(treatment_ar_coefs)
        else:
            # Generate stable AR coefficients (sum < 1 for stationarity)
            self.treatment_ar_coefs = self._generate_stable_ar_coefs(treatment_ar_order)

        # Set up lagged effect coefficients
        if lagged_effect_coefs is not None:
            self.lagged_effect_coefs = np.asarray(lagged_effect_coefs)
        else:
            # No lagged effects by default
            self.lagged_effect_coefs = np.array([])

    def _generate_stable_ar_coefs(self, order: int) -> NDArray[np.float64]:
        """Generate stable AR coefficients that ensure stationarity.

        Args:
            order: AR order

        Returns:
            Array of AR coefficients with |sum| < 0.9
        """
        if order == 0:
            return np.array([])

        # Generate random coefficients
        coefs = self._rng.uniform(-0.5, 0.7, size=order)

        # Scale to ensure stationarity
        total = np.abs(coefs).sum()
        if total > 0.85:
            coefs = coefs * (0.85 / total)

        return coefs

    def generate(self) -> TimeSeriesDGPResult:
        """Generate a single dataset from the time series DGP.

        Returns:
            TimeSeriesDGPResult with Y, T, X, and metadata
        """
        if self.n_units == 1:
            return self._generate_single_series()
        else:
            return self._generate_panel()

    def _generate_single_series(self) -> TimeSeriesDGPResult:
        """Generate single time series.

        Returns:
            TimeSeriesDGPResult for single unit
        """
        n = self.n_periods

        # Generate confounders (VAR(1) process)
        X = self._generate_var_confounders(n)

        # Generate treatment (AR(p) + confounders)
        T = self._generate_treatment(X)

        # Generate outcome
        Y = self._generate_outcome(T, X)

        # Compute cumulative effect
        cumulative_effect = self.true_effect + np.sum(self.lagged_effect_coefs)

        return TimeSeriesDGPResult(
            Y=Y,
            T=T,
            X=X,
            true_effect=self.true_effect,
            true_cumulative_effect=cumulative_effect,
            n_periods=n,
            n_units=1,
            p=self.p,
            time_index=np.arange(n),
            unit_ids=None,
            ar_coefs=self.treatment_ar_coefs,
            var_coefs=None,
            error_ar_coef=self.error_ar_coef,
        )

    def _generate_panel(self) -> TimeSeriesDGPResult:
        """Generate panel data with optional fixed effects.

        Returns:
            TimeSeriesDGPResult with panel structure
        """
        n_t = self.n_periods
        n_i = self.n_units

        # Generate unit fixed effects
        if self.panel_fixed_effects:
            unit_fe = self._rng.standard_normal(n_i) * 0.5
            time_fe = self._rng.standard_normal(n_t) * 0.3
        else:
            unit_fe = np.zeros(n_i)
            time_fe = np.zeros(n_t)

        # Initialize arrays
        Y = np.zeros((n_i, n_t))
        T = np.zeros((n_i, n_t))
        X = np.zeros((n_i, n_t, self.p))

        for i in range(n_i):
            # Generate unit-specific confounders
            X[i] = self._generate_var_confounders(n_t)

            # Generate treatment with unit heterogeneity
            T[i] = self._generate_treatment(X[i], unit_effect=unit_fe[i] * 0.2)

            # Generate outcome with fixed effects
            Y[i] = self._generate_outcome(T[i], X[i], unit_effect=unit_fe[i], time_effects=time_fe)

        # Create unit IDs
        unit_ids = np.repeat(np.arange(n_i), n_t)

        # Compute cumulative effect
        cumulative_effect = self.true_effect + np.sum(self.lagged_effect_coefs)

        return TimeSeriesDGPResult(
            Y=Y,
            T=T,
            X=X,
            true_effect=self.true_effect,
            true_cumulative_effect=cumulative_effect,
            n_periods=n_t,
            n_units=n_i,
            p=self.p,
            time_index=np.arange(n_t),
            unit_ids=unit_ids,
            ar_coefs=self.treatment_ar_coefs,
            var_coefs=None,
            error_ar_coef=self.error_ar_coef,
        )

    def _generate_var_confounders(self, n: int) -> NDArray[np.float64]:
        """Generate VAR(1) confounders.

        X_t = ρ * X_{t-1} + ε_t

        where ρ is the diagonal VAR coefficient and ε_t ~ N(0, I).

        Args:
            n: Number of time periods

        Returns:
            Confounder array (n, p)
        """
        X = np.zeros((n, self.p))

        # Initialize
        X[0] = self._rng.standard_normal(self.p)

        # Generate VAR(1) process (diagonal coefficient matrix)
        innovation_var = 1 - self.confounder_var_coef**2
        innovation_std = np.sqrt(max(innovation_var, 0.01))

        for t in range(1, n):
            innovation = self._rng.standard_normal(self.p) * innovation_std
            X[t] = self.confounder_var_coef * X[t - 1] + innovation

        return X

    def _generate_treatment(
        self,
        X: NDArray[np.float64],
        unit_effect: float = 0.0,
    ) -> NDArray[np.float64]:
        """Generate AR(p) treatment with confounder dependence.

        T_t = Σ_{j=1}^{p} φ_j T_{t-j} + β'X_t + unit_effect + ε_t

        Args:
            X: Confounders (n, p)
            unit_effect: Unit fixed effect on treatment (for panel)

        Returns:
            Treatment array (n,)
        """
        n = X.shape[0]
        ar_order = self.treatment_ar_order

        # Confounding effect
        beta = self._rng.standard_normal(self.p) * self.confounding_strength * 0.3

        # Generate AR process for treatment
        T = np.zeros(n)

        # Burn-in period
        burn_in = max(ar_order + 10, 50)
        T_extended = np.zeros(n + burn_in)

        # Innovation variance (adjusted for stationarity)
        innovation_var = 0.5
        innovation_std = np.sqrt(innovation_var)

        for t in range(burn_in, n + burn_in):
            # AR component
            ar_component = 0.0
            for j in range(ar_order):
                if t - j - 1 >= 0:
                    ar_component += self.treatment_ar_coefs[j] * T_extended[t - j - 1]

            # Confounder component (use X at corresponding time)
            t_actual = t - burn_in
            if 0 <= t_actual < n:
                conf_component = X[t_actual] @ beta
            else:
                conf_component = 0.0

            # Innovation
            innovation = self._rng.standard_normal() * innovation_std

            # Combine
            T_extended[t] = ar_component + conf_component + unit_effect + innovation

        # Extract non-burn-in portion
        T = T_extended[burn_in:]

        # Convert to binary if needed
        if self.treatment_type == "binary":
            propensity = 1 / (1 + np.exp(-T))
            T = (self._rng.random(n) < propensity).astype(float)

        return T

    def _generate_outcome(
        self,
        T: NDArray[np.float64],
        X: NDArray[np.float64],
        unit_effect: float = 0.0,
        time_effects: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Generate outcome with treatment effects and autocorrelated errors.

        Y_t = τ_0 T_t + Σ_{j=1}^{L} τ_j T_{t-j} + γ'X_t + α_i + λ_t + ε_t

        where ε_t = ρ ε_{t-1} + u_t is AR(1) error.

        Args:
            T: Treatment (n,)
            X: Confounders (n, p)
            unit_effect: Unit fixed effect
            time_effects: Time fixed effects (n,)

        Returns:
            Outcome array (n,)
        """
        n = len(T)

        # Contemporaneous treatment effect
        Y = self.true_effect * T.copy()

        # Lagged treatment effects
        for lag, coef in enumerate(self.lagged_effect_coefs, start=1):
            if lag < n:
                Y[lag:] += coef * T[:-lag]

        # Confounder effect
        gamma = self._rng.standard_normal(self.p) * self.confounding_strength
        Y += X @ gamma

        # Fixed effects
        Y += unit_effect
        if time_effects is not None:
            Y += time_effects

        # AR(1) error
        error = self._generate_ar1_errors(n)
        Y += error

        return Y

    def _generate_ar1_errors(self, n: int) -> NDArray[np.float64]:
        """Generate AR(1) errors.

        ε_t = ρ ε_{t-1} + u_t, u_t ~ N(0, σ²)

        Args:
            n: Number of time periods

        Returns:
            Error array (n,)
        """
        errors = np.zeros(n)

        # Innovation variance (unconditional variance = σ² / (1 - ρ²))
        innovation_var = self.noise_level**2 * (1 - self.error_ar_coef**2)
        innovation_std = np.sqrt(max(innovation_var, 0.01))

        innovations = self._rng.standard_normal(n) * innovation_std

        errors[0] = innovations[0] / np.sqrt(1 - self.error_ar_coef**2)
        for t in range(1, n):
            errors[t] = self.error_ar_coef * errors[t - 1] + innovations[t]

        return errors


class BreakDGPGenerator:
    """Generate time series with structural breaks in treatment effects.

    Useful for testing rolling window estimators that should detect
    time-varying effects.

    Examples:
        >>> dgp = BreakDGPGenerator(
        ...     n_periods=300, p=3,
        ...     effects=[1.0, 3.0, 1.5],
        ...     break_points=[100, 200],
        ...     random_state=42
        ... )
        >>> result = dgp.generate()
        >>> result.effect_periods
        [(0, 100), (100, 200), (200, 300)]
    """

    def __init__(
        self,
        n_periods: int,
        p: int,
        effects: list[float],
        break_points: list[int],
        confounding_strength: float = 1.0,
        noise_level: float = 1.0,
        random_state: int | None = None,
    ):
        """Initialize break DGP generator.

        Args:
            n_periods: Total number of time periods
            p: Number of confounders
            effects: Treatment effect for each regime
            break_points: Time indices where breaks occur
            confounding_strength: Strength of confounding
            noise_level: Standard deviation of noise
            random_state: Random seed

        Raises:
            ValueError: If effects and break_points are inconsistent
        """
        if len(effects) != len(break_points) + 1:
            raise ValueError(
                f"Number of effects ({len(effects)}) must equal "
                f"number of break_points + 1 ({len(break_points) + 1})"
            )

        sorted_breaks = sorted(break_points)
        if sorted_breaks != break_points:
            raise ValueError("break_points must be in ascending order")

        if break_points and (break_points[0] <= 0 or break_points[-1] >= n_periods):
            raise ValueError("break_points must be within (0, n_periods)")

        self.n_periods = n_periods
        self.p = p
        self.effects = effects
        self.break_points = break_points
        self.confounding_strength = confounding_strength
        self.noise_level = noise_level
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def generate(self) -> "BreakDGPResult":
        """Generate data with structural breaks.

        Returns:
            BreakDGPResult with Y, T, X, and regime information
        """
        n = self.n_periods

        # Generate confounders (simple AR(1))
        X = np.zeros((n, self.p))
        X[0] = self._rng.standard_normal(self.p)
        for t in range(1, n):
            X[t] = 0.5 * X[t - 1] + self._rng.standard_normal(self.p) * 0.87

        # Generate treatment
        beta = self._rng.standard_normal(self.p) * self.confounding_strength * 0.3
        T = X @ beta + self._rng.standard_normal(n) * 0.5

        # Generate outcome with time-varying effects
        Y = np.zeros(n)
        gamma = self._rng.standard_normal(self.p) * self.confounding_strength

        # Determine regime for each time point
        regime_indices = self._get_regime_indices()

        for t in range(n):
            regime = regime_indices[t]
            effect = self.effects[regime]
            Y[t] = effect * T[t] + X[t] @ gamma + self._rng.standard_normal() * self.noise_level

        # Effect periods
        all_breaks = [0] + self.break_points + [n]
        effect_periods = [(all_breaks[i], all_breaks[i + 1]) for i in range(len(self.effects))]

        return BreakDGPResult(
            Y=Y,
            T=T,
            X=X,
            effects=self.effects,
            break_points=self.break_points,
            effect_periods=effect_periods,
            regime_indices=regime_indices,
            n_periods=n,
            p=self.p,
        )

    def _get_regime_indices(self) -> NDArray[np.int64]:
        """Get regime index for each time point.

        Returns:
            Array of regime indices (n,)
        """
        n = self.n_periods
        regime_indices = np.zeros(n, dtype=np.int64)

        all_breaks = [0] + self.break_points + [n]
        for regime, (start, end) in enumerate(zip(all_breaks[:-1], all_breaks[1:], strict=True)):
            regime_indices[start:end] = regime

        return regime_indices


@dataclass
class BreakDGPResult:
    """Result from break DGP generation.

    Attributes:
        Y: Outcome variable (n,)
        T: Treatment variable (n,)
        X: Confounders (n, p)
        effects: True effect for each regime
        break_points: Time indices of breaks
        effect_periods: (start, end) tuples for each regime
        regime_indices: Regime index for each time point
        n_periods: Number of time periods
        p: Number of confounders
    """

    Y: NDArray[np.float64]
    T: NDArray[np.float64]
    X: NDArray[np.float64]
    effects: list[float]
    break_points: list[int]
    effect_periods: list[tuple[int, int]]
    regime_indices: NDArray[np.int64]
    n_periods: int
    p: int


def create_ar_dgp(
    n: int = 200,
    p: int = 5,
    true_effect: float = 2.0,
    ar_coef: float = 0.5,
    confounding_strength: float = 1.0,
    random_state: int | None = None,
) -> TimeSeriesDGPResult:
    """Convenience function for simple AR(1) treatment DGP.

    Args:
        n: Number of time periods
        p: Number of confounders
        true_effect: True treatment effect
        ar_coef: AR(1) coefficient for treatment
        confounding_strength: Strength of confounding
        random_state: Random seed

    Returns:
        TimeSeriesDGPResult
    """
    dgp = TimeSeriesDGPGenerator(
        n_periods=n,
        p=p,
        true_effect=true_effect,
        treatment_ar_order=1,
        treatment_ar_coefs=np.array([ar_coef]),
        confounding_strength=confounding_strength,
        random_state=random_state,
    )
    return dgp.generate()


def create_panel_dgp(
    n_periods: int = 100,
    n_units: int = 50,
    p: int = 5,
    true_effect: float = 2.0,
    fixed_effects: bool = True,
    random_state: int | None = None,
) -> TimeSeriesDGPResult:
    """Convenience function for panel DGP with fixed effects.

    Args:
        n_periods: Number of time periods per unit
        n_units: Number of cross-sectional units
        p: Number of confounders
        true_effect: True treatment effect
        fixed_effects: Whether to include unit/time fixed effects
        random_state: Random seed

    Returns:
        TimeSeriesDGPResult with panel structure
    """
    dgp = TimeSeriesDGPGenerator(
        n_periods=n_periods,
        n_units=n_units,
        p=p,
        true_effect=true_effect,
        panel_fixed_effects=fixed_effects,
        random_state=random_state,
    )
    return dgp.generate()


def create_break_dgp(
    n: int = 300,
    p: int = 5,
    effects: list[float] | None = None,
    break_points: list[int] | None = None,
    random_state: int | None = None,
) -> BreakDGPResult:
    """Convenience function for structural break DGP.

    Args:
        n: Number of time periods
        p: Number of confounders
        effects: Treatment effects by regime (default: [1.0, 3.0, 1.0])
        break_points: Break locations (default: [n//3, 2*n//3])
        random_state: Random seed

    Returns:
        BreakDGPResult
    """
    if effects is None:
        effects = [1.0, 3.0, 1.0]
    if break_points is None:
        break_points = [n // 3, 2 * n // 3]

    dgp = BreakDGPGenerator(
        n_periods=n,
        p=p,
        effects=effects,
        break_points=break_points,
        random_state=random_state,
    )
    return dgp.generate()
