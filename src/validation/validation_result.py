"""Validation result data structure for Double ML validation methods."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Literal, Optional
import json
import numpy as np


@dataclass
class ValidationResult:
    """Result from a validation method.

    Standardized format for all 7 validation methods ensures consistency
    in reporting, storage, and analysis.

    Attributes:
        method: Name of validation method (e.g., "bias", "mse", "coverage")
        status: Validation status ("PASS", "FAIL", or "WARNING")
        bias: Estimated bias (difference from true effect)
        mse: Mean squared error
        coverage: Coverage rate for confidence intervals
        ci_lower: Lower bound of confidence interval for bias
        ci_upper: Upper bound of confidence interval for bias
        n_simulations: Number of Monte Carlo simulations
        timestamp: When validation was run
        metadata: Additional method-specific information

    Examples:
        >>> result = ValidationResult(
        ...     method="bias",
        ...     status="PASS",
        ...     bias=0.02,
        ...     mse=0.15,
        ...     coverage=0.95,
        ...     ci_lower=-0.05,
        ...     ci_upper=0.09,
        ...     n_simulations=1000,
        ...     timestamp=datetime.now(),
        ...     metadata={"dgp": "linear", "n": 1000}
        ... )
        >>> result.status
        'PASS'
    """

    method: str
    status: Literal["PASS", "FAIL", "WARNING"]
    bias: float
    mse: float
    coverage: float
    ci_lower: float
    ci_upper: float
    n_simulations: int
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with JSON-serializable types
        """
        d = asdict(self)
        # Convert datetime to ISO format string
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON-formatted string
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidationResult":
        """Create ValidationResult from dictionary.

        Args:
            d: Dictionary with ValidationResult fields

        Returns:
            ValidationResult instance
        """
        # Convert timestamp string back to datetime
        d_copy = d.copy()
        if isinstance(d_copy["timestamp"], str):
            d_copy["timestamp"] = datetime.fromisoformat(d_copy["timestamp"])

        return cls(**d_copy)

    @classmethod
    def from_json(cls, json_str: str) -> "ValidationResult":
        """Create ValidationResult from JSON string.

        Args:
            json_str: JSON-formatted string

        Returns:
            ValidationResult instance
        """
        d = json.loads(json_str)
        return cls.from_dict(d)

    def passes(self) -> bool:
        """Check if validation passed.

        Returns:
            True if status is "PASS", False otherwise
        """
        return self.status == "PASS"

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Formatted summary string
        """
        return f"""
Validation Method: {self.method}
Status: {self.status}
Bias: {self.bias:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])
MSE: {self.mse:.4f}
Coverage: {self.coverage:.2%}
Simulations: {self.n_simulations:,}
Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
