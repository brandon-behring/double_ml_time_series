"""
Enhanced Data Generating Process for Double ML robustness testing.

Extends basic DGP with:
- Heterogeneous treatment effects (HTE)
- Misspecification scenarios
- Propensity score extremeness
- Measurement error
- Omitted variable bias

Used to test DML performance under assumption violations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EnhancedDGPResult:
    """Result from enhanced DGP generation.

    Attributes:
        Y: Outcome variable (n,)
        T: Treatment variable (n,) - binary {0, 1}
        X: Observed confounders (n, p)
        X_true: True confounders including omitted variables (n, p + p_omitted)
        true_effect: Individual treatment effects (n,) - varies by individual for HTE
        true_ate: Average treatment effect (scalar)
        propensity_score: True propensity scores P(T=1|X) (n,)
        n: Sample size
        p: Number of observed confounders
        p_omitted: Number of omitted confounders
        measurement_error_std: Measurement error standard deviation
    """

    Y: np.ndarray
    T: np.ndarray
    X: np.ndarray
    X_true: np.ndarray
    true_effect: np.ndarray  # Individual effects (HTE)
    true_ate: float  # Average effect
    propensity_score: np.ndarray
    n: int
    p: int
    p_omitted: int
    measurement_error_std: float


class EnhancedDGPGenerator:
    """Generate synthetic data with realistic assumption violations.

    Features:

    1. **Heterogeneous Treatment Effects (HTE)**:

       - Effect varies by X: τ(X) = base_effect + β^T X
       - Tests DML robustness to effect heterogeneity

    2. **Misspecification**:

       - Omitted variables: confounders not included in X
       - Non-linear relationships with polynomial terms
       - Interaction effects between confounders

    3. **Propensity Score Extremeness**:

       - Very high/low treatment probabilities
       - Tests overlap assumption violations

    4. **Measurement Error**:

       - Classical errors: X_obs = X_true + ε
       - Tests robustness to noisy covariates

    Examples:
        >>> # HTE with linear heterogeneity
        >>> dgp = EnhancedDGPGenerator(
        ...     n=1000, p=5, base_effect=2.0,
        ...     hte_strength=1.0, random_state=42
        ... )
        >>> result = dgp.generate()
        >>> result.true_effect.mean()  # Average effect ≈ 2.0
        2.05

        >>> # Omitted variable bias
        >>> dgp = EnhancedDGPGenerator(
        ...     n=1000, p=5, base_effect=2.0,
        ...     p_omitted=2, omitted_var_strength=2.0,
        ...     random_state=42
        ... )
        >>> result = dgp.generate()
        >>> result.p_omitted
        2
    """

    def __init__(
        self,
        n: int,
        p: int,
        base_effect: float,
        # HTE parameters
        hte_strength: float = 0.0,
        hte_nonlinear: bool = False,
        # Misspecification parameters
        p_omitted: int = 0,
        omitted_var_strength: float = 0.0,
        polynomial_degree: int = 1,
        # Propensity score parameters
        propensity_extremeness: float = 1.0,  # 1.0 = normal, >1.0 = more extreme
        # Measurement error
        measurement_error_std: float = 0.0,
        # Standard parameters
        confounding_strength: float = 1.0,
        noise_level: float = 1.0,
        random_state: int | None = None,
    ):
        """Initialize enhanced DGP generator.

        Args:
            n: Sample size
            p: Number of observed confounders
            base_effect: Base treatment effect (constant part)
            hte_strength: Strength of heterogeneity (0 = constant effect)
            hte_nonlinear: Use non-linear HTE function
            p_omitted: Number of omitted confounders
            omitted_var_strength: Strength of omitted variable confounding
            polynomial_degree: Degree of polynomial terms (1 = linear, 2 = quadratic, etc.)
            propensity_extremeness: Multiplier for propensity logit (>1 = more extreme)
            measurement_error_std: Std dev of measurement error in X
            confounding_strength: Overall confounding strength
            noise_level: Noise in outcome equation
            random_state: Random seed
        """
        self.n = n
        self.p = p
        self.base_effect = base_effect
        self.hte_strength = hte_strength
        self.hte_nonlinear = hte_nonlinear
        self.p_omitted = p_omitted
        self.omitted_var_strength = omitted_var_strength
        self.polynomial_degree = polynomial_degree
        self.propensity_extremeness = propensity_extremeness
        self.measurement_error_std = measurement_error_std
        self.confounding_strength = confounding_strength
        self.noise_level = noise_level
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def generate(self) -> EnhancedDGPResult:
        """Generate data from enhanced DGP.

        Returns:
            EnhancedDGPResult with data and metadata
        """
        # Step 1: Generate true confounders (observed + omitted)
        p_total = self.p + self.p_omitted
        X_true = self._rng.randn(self.n, p_total)

        # Step 2: Add measurement error to observed confounders
        X_observed = X_true[:, : self.p].copy()
        if self.measurement_error_std > 0:
            measurement_error = self._rng.randn(self.n, self.p) * self.measurement_error_std
            X_observed += measurement_error

        # Step 3: Generate treatment with confounding from ALL confounders
        propensity_logit = self._calculate_propensity_logit(X_true)
        propensity_score = 1 / (1 + np.exp(-propensity_logit))
        T = (self._rng.rand(self.n) < propensity_score).astype(float)

        # Step 4: Calculate individual treatment effects (HTE)
        true_effect = self._calculate_treatment_effects(X_true)
        true_ate = float(np.mean(true_effect))

        # Step 5: Generate outcome with confounding
        Y = self._generate_outcome(X_true, T, true_effect)

        return EnhancedDGPResult(
            Y=Y,
            T=T,
            X=X_observed,  # Only observed confounders
            X_true=X_true,  # All confounders (observed + omitted)
            true_effect=true_effect,
            true_ate=true_ate,
            propensity_score=propensity_score,
            n=self.n,
            p=self.p,
            p_omitted=self.p_omitted,
            measurement_error_std=self.measurement_error_std,
        )

    def _calculate_propensity_logit(self, X: np.ndarray) -> np.ndarray:
        """Calculate propensity score logit from all confounders.

        Args:
            X: All confounders (n, p_total)

        Returns:
            Propensity logit values (n,)
        """
        # Linear term: β^T X
        beta = self._rng.randn(X.shape[1]) * self.confounding_strength
        logit = X @ beta

        # Add polynomial terms if requested
        if self.polynomial_degree >= 2:
            # Add X^2 terms for first few variables
            n_poly = min(3, X.shape[1])
            for i in range(n_poly):
                logit += 0.5 * self.confounding_strength * (X[:, i] ** 2)

        # Apply extremeness multiplier
        logit *= self.propensity_extremeness

        return logit

    def _calculate_treatment_effects(self, X: np.ndarray) -> np.ndarray:
        """Calculate individual treatment effects (HTE).

        Args:
            X: All confounders (n, p_total)

        Returns:
            Individual effects τ(X) (n,)
        """
        n = X.shape[0]

        # Start with base effect
        effects = np.full(n, self.base_effect)

        if self.hte_strength > 0:
            if self.hte_nonlinear:
                # Non-linear HTE: τ(X) = base + β*sin(X[0]) + γ*X[1]^2
                effects += self.hte_strength * np.sin(X[:, 0])
                if X.shape[1] > 1:
                    effects += 0.5 * self.hte_strength * (X[:, 1] ** 2)
            else:
                # Linear HTE: τ(X) = base + β^T X
                # Use only first min(3, p_total) variables for HTE
                n_hte = min(3, X.shape[1])
                hte_coef = self._rng.randn(n_hte) * self.hte_strength
                effects += X[:, :n_hte] @ hte_coef

        return effects

    def _generate_outcome(
        self, X: np.ndarray, T: np.ndarray, true_effect: np.ndarray
    ) -> np.ndarray:
        """Generate outcome with confounding and treatment effect.

        Args:
            X: All confounders (n, p_total)
            T: Treatment (n,)
            true_effect: Individual treatment effects (n,)

        Returns:
            Outcome Y (n,)
        """
        n = X.shape[0]

        # Confounding: Y depends on X
        gamma = self._rng.randn(X.shape[1]) * self.confounding_strength
        Y = X @ gamma

        # Add polynomial confounding if requested
        if self.polynomial_degree >= 2:
            n_poly = min(3, X.shape[1])
            for i in range(n_poly):
                Y += 0.5 * self.confounding_strength * (X[:, i] ** 2)

        # Add treatment effect: Y += τ(X) * T
        Y += true_effect * T

        # Add noise
        Y += self._rng.randn(n) * self.noise_level

        return Y

    def get_scenario_description(self) -> str:
        """Get human-readable description of DGP configuration.

        Returns:
            Formatted string describing the scenario
        """
        desc = [
            f"Enhanced DGP (n={self.n}, p={self.p})",
            f"Base effect: {self.base_effect:.2f}",
        ]

        if self.hte_strength > 0:
            hte_type = "non-linear" if self.hte_nonlinear else "linear"
            desc.append(f"HTE ({hte_type}): strength={self.hte_strength:.2f}")

        if self.p_omitted > 0:
            desc.append(
                f"Omitted variables: {self.p_omitted} (strength={self.omitted_var_strength:.2f})"
            )

        if self.polynomial_degree > 1:
            desc.append(f"Polynomial degree: {self.polynomial_degree}")

        if self.propensity_extremeness != 1.0:
            desc.append(f"Propensity extremeness: {self.propensity_extremeness:.2f}x")

        if self.measurement_error_std > 0:
            desc.append(f"Measurement error: σ={self.measurement_error_std:.2f}")

        return " | ".join(desc)
