"""
Data Generating Process (DGP) Generator for Double ML Validation.

Generates synthetic data with known treatment effects and controllable confounding
for validating Double ML estimators.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class DGPResult:
    """Result from DGP generation.

    Attributes:
        Y: Outcome variable (n,)
        T: Treatment variable (n,) - binary {0, 1}
        X: Confounders (n, p)
        true_effect: Known true treatment effect
        n: Sample size
        p: Number of confounders
    """

    Y: np.ndarray
    T: np.ndarray
    X: np.ndarray
    true_effect: float
    n: int
    p: int


class DGPGenerator:
    """Generate synthetic data with known properties for validation.

    This generator creates data where:
    - Treatment assignment can be linear or nonlinear in confounders
    - Outcome can be linear or nonlinear in confounders
    - True treatment effect is known by construction
    - Confounding strength is controllable

    Examples:
        >>> # Simple linear DGP with no confounding
        >>> dgp = DGPGenerator(n=1000, p=5, true_effect=2.0,
        ...                     confounding_strength=0.0, random_state=42)
        >>> result = dgp.generate()
        >>> result.Y.shape
        (1000,)

        >>> # Nonlinear DGP with strong confounding
        >>> dgp = DGPGenerator(n=1000, p=10, true_effect=3.0,
        ...                     confounding_strength=2.0,
        ...                     treatment_model='nonlinear',
        ...                     outcome_model='nonlinear',
        ...                     random_state=42)
        >>> result = dgp.generate()
    """

    def __init__(
        self,
        n: int,
        p: int,
        true_effect: float,
        confounding_strength: float = 1.0,
        treatment_model: Literal["linear", "nonlinear"] = "linear",
        outcome_model: Literal["linear", "nonlinear"] = "linear",
        noise_level: float = 1.0,
        random_state: int | None = None,
    ):
        """Initialize DGP generator.

        Args:
            n: Sample size (must be positive)
            p: Number of confounders (must be positive)
            true_effect: True treatment effect (can be any real number)
            confounding_strength: Strength of confounding (0 = none, higher = more)
            treatment_model: Treatment assignment model ('linear' or 'nonlinear')
            outcome_model: Outcome model ('linear' or 'nonlinear')
            noise_level: Standard deviation of noise terms
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If n or p are not positive, or models are invalid
        """
        # Validate parameters
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if treatment_model not in ["linear", "nonlinear"]:
            raise ValueError(
                f"treatment_model must be 'linear' or 'nonlinear', got {treatment_model}"
            )
        if outcome_model not in ["linear", "nonlinear"]:
            raise ValueError(f"outcome_model must be 'linear' or 'nonlinear', got {outcome_model}")

        self.n = n
        self.p = p
        self.true_effect = true_effect
        self.confounding_strength = confounding_strength
        self.treatment_model = treatment_model
        self.outcome_model = outcome_model
        self.noise_level = noise_level
        self.random_state = random_state

        # Initialize random number generator
        self._rng = np.random.RandomState(random_state)

    def generate(self) -> DGPResult:
        """Generate a single dataset from the DGP.

        Each call generates a new random realization with the same parameters.

        Returns:
            DGPResult with Y, T, X, and metadata
        """
        # Generate confounders (standardized normal)
        X = self._rng.randn(self.n, self.p)

        # Generate treatment
        T = self._generate_treatment(X)

        # Generate outcome
        Y = self._generate_outcome(T, X)

        return DGPResult(Y=Y, T=T, X=X, true_effect=self.true_effect, n=self.n, p=self.p)

    def _generate_treatment(self, X: np.ndarray) -> np.ndarray:
        """Generate treatment assignment.

        Args:
            X: Confounders (n, p)

        Returns:
            Binary treatment (n,) with values {0, 1}
        """
        if self.treatment_model == "linear":
            # Linear propensity score: e(X) = expit(α₀ + X β + γ₁X₁ + γ₂X₂ + ...)
            # Use confounding_strength to scale influence of X
            beta = self._rng.randn(self.p) * self.confounding_strength

            # Propensity score (probability of treatment)
            logit = X @ beta
            propensity = 1 / (1 + np.exp(-logit))

        else:  # nonlinear
            # Nonlinear propensity score using interactions and squares
            # e(X) = expit(X₁² + X₂² + X₁X₂ + ...)

            # Use first few features for nonlinearity
            n_interact = min(3, self.p)
            interactions = []

            # Add squares
            for i in range(n_interact):
                interactions.append(X[:, i] ** 2)

            # Add products if multiple features
            if n_interact >= 2:
                for i in range(n_interact - 1):
                    interactions.append(X[:, i] * X[:, i + 1])

            # Combine
            if interactions:
                interact_features = np.column_stack(interactions)
                beta_interact = (
                    self._rng.randn(interact_features.shape[1]) * self.confounding_strength
                )
                logit = interact_features @ beta_interact
            else:
                logit = np.zeros(self.n)

            propensity = 1 / (1 + np.exp(-logit))

        # Draw binary treatment
        T = (self._rng.rand(self.n) < propensity).astype(int)

        return T

    def _generate_outcome(self, T: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Generate outcome variable.

        Y = τT + f(X) + ε

        where τ is the true treatment effect, f(X) is the outcome function,
        and ε is noise.

        Args:
            T: Treatment (n,)
            X: Confounders (n, p)

        Returns:
            Continuous outcome (n,)
        """
        # Treatment effect component
        treatment_component = self.true_effect * T

        # Outcome function of confounders
        if self.outcome_model == "linear":
            # Linear: f(X) = X γ
            gamma = self._rng.randn(self.p) * self.confounding_strength
            outcome_function = X @ gamma

        else:  # nonlinear
            # Nonlinear: f(X) = sin(X₁) + exp(X₂/2) + X₃² + ...
            n_features = min(5, self.p)
            components = []

            if n_features >= 1:
                components.append(np.sin(X[:, 0]))
            if n_features >= 2:
                components.append(np.exp(X[:, 1] / 2))
            if n_features >= 3:
                components.append(X[:, 2] ** 2)
            if n_features >= 4:
                components.append(np.log(np.abs(X[:, 3]) + 1))
            if n_features >= 5:
                components.append(X[:, 4] * X[:, 0])

            # Scale by confounding strength
            outcome_function = np.sum(components, axis=0) * self.confounding_strength

        # Noise
        noise = self._rng.randn(self.n) * self.noise_level

        # Combine
        Y = treatment_component + outcome_function + noise

        return Y
