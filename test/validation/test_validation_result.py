"""
Tests for ValidationResult dataclass.

Test-first development: Ensure ValidationResult serialization and methods work correctly.

All tests in this module are unit tests (no estimation, pure serialization).
"""

import json
from datetime import datetime
import pytest

from src.validation.validation_result import ValidationResult

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.tier1


class TestValidationResultCreation:
    """Test ValidationResult creation and basic attributes."""

    def test_validation_result_creation(self):
        """Test ValidationResult can be created with all fields."""
        result = ValidationResult(
            method="bias",
            status="PASS",
            bias=0.02,
            mse=0.15,
            coverage=0.95,
            ci_lower=-0.05,
            ci_upper=0.09,
            n_simulations=1000,
            timestamp=datetime(2025, 11, 13, 12, 0, 0),
            metadata={"dgp": "linear", "n": 1000},
        )

        assert result.method == "bias"
        assert result.status == "PASS"
        assert result.bias == 0.02
        assert result.mse == 0.15
        assert result.coverage == 0.95
        assert result.n_simulations == 1000

    def test_validation_result_with_fail_status(self):
        """Test ValidationResult with FAIL status."""
        result = ValidationResult(
            method="mse",
            status="FAIL",
            bias=0.5,
            mse=1.2,
            coverage=0.85,
            ci_lower=0.3,
            ci_upper=0.7,
            n_simulations=500,
            timestamp=datetime.now(),
            metadata={},
        )

        assert result.status == "FAIL"
        assert not result.passes()

    def test_validation_result_with_warning_status(self):
        """Test ValidationResult with WARNING status."""
        result = ValidationResult(
            method="coverage",
            status="WARNING",
            bias=0.1,
            mse=0.3,
            coverage=0.92,
            ci_lower=0.05,
            ci_upper=0.15,
            n_simulations=500,
            timestamp=datetime.now(),
            metadata={"warning": "low power"},
        )

        assert result.status == "WARNING"
        assert not result.passes()


class TestValidationResultSerialization:
    """Test ValidationResult serialization to/from dict and JSON."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2025, 11, 13, 12, 0, 0)
        result = ValidationResult(
            method="bias",
            status="PASS",
            bias=0.02,
            mse=0.15,
            coverage=0.95,
            ci_lower=-0.05,
            ci_upper=0.09,
            n_simulations=1000,
            timestamp=timestamp,
            metadata={"test": "value"},
        )

        d = result.to_dict()

        assert d["method"] == "bias"
        assert d["status"] == "PASS"
        assert d["bias"] == 0.02
        assert d["timestamp"] == timestamp.isoformat()
        assert isinstance(d["timestamp"], str)

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "method": "mse",
            "status": "PASS",
            "bias": 0.01,
            "mse": 0.12,
            "coverage": 0.96,
            "ci_lower": -0.03,
            "ci_upper": 0.05,
            "n_simulations": 2000,
            "timestamp": "2025-11-13T12:00:00",
            "metadata": {"key": "value"},
        }

        result = ValidationResult.from_dict(d)

        assert result.method == "mse"
        assert result.bias == 0.01
        assert result.timestamp == datetime(2025, 11, 13, 12, 0, 0)

    def test_to_json(self):
        """Test conversion to JSON string."""
        result = ValidationResult(
            method="coverage",
            status="PASS",
            bias=0.0,
            mse=0.1,
            coverage=0.95,
            ci_lower=-0.02,
            ci_upper=0.02,
            n_simulations=1000,
            timestamp=datetime(2025, 11, 13, 12, 0, 0),
            metadata={},
        )

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "coverage" in json_str
        assert "PASS" in json_str
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["method"] == "coverage"

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = """{
            "method": "bias",
            "status": "FAIL",
            "bias": 0.3,
            "mse": 0.5,
            "coverage": 0.90,
            "ci_lower": 0.2,
            "ci_upper": 0.4,
            "n_simulations": 500,
            "timestamp": "2025-11-13T15:30:00",
            "metadata": {"note": "test"}
        }"""

        result = ValidationResult.from_json(json_str)

        assert result.method == "bias"
        assert result.status == "FAIL"
        assert result.bias == 0.3
        assert result.timestamp == datetime(2025, 11, 13, 15, 30, 0)

    def test_round_trip_serialization(self):
        """Test round-trip: result → dict → result."""
        original = ValidationResult(
            method="test",
            status="PASS",
            bias=0.05,
            mse=0.2,
            coverage=0.94,
            ci_lower=0.0,
            ci_upper=0.1,
            n_simulations=1500,
            timestamp=datetime(2025, 11, 13, 10, 30, 0),
            metadata={"test": "round_trip"},
        )

        # Round trip via dict
        d = original.to_dict()
        restored = ValidationResult.from_dict(d)

        assert restored.method == original.method
        assert restored.bias == original.bias
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata

    def test_round_trip_json_serialization(self):
        """Test round-trip: result → JSON → result."""
        original = ValidationResult(
            method="json_test",
            status="WARNING",
            bias=0.08,
            mse=0.25,
            coverage=0.91,
            ci_lower=0.03,
            ci_upper=0.13,
            n_simulations=800,
            timestamp=datetime(2025, 11, 13, 14, 0, 0),
            metadata={"format": "json"},
        )

        # Round trip via JSON
        json_str = original.to_json()
        restored = ValidationResult.from_json(json_str)

        assert restored.method == original.method
        assert restored.status == original.status
        assert restored.timestamp == original.timestamp


class TestValidationResultMethods:
    """Test ValidationResult helper methods."""

    def test_passes_method_with_pass_status(self):
        """Test passes() returns True for PASS status."""
        result = ValidationResult(
            method="test",
            status="PASS",
            bias=0.0,
            mse=0.1,
            coverage=0.95,
            ci_lower=-0.05,
            ci_upper=0.05,
            n_simulations=1000,
            timestamp=datetime.now(),
            metadata={},
        )

        assert result.passes() is True

    def test_passes_method_with_fail_status(self):
        """Test passes() returns False for FAIL status."""
        result = ValidationResult(
            method="test",
            status="FAIL",
            bias=0.5,
            mse=1.0,
            coverage=0.80,
            ci_lower=0.3,
            ci_upper=0.7,
            n_simulations=1000,
            timestamp=datetime.now(),
            metadata={},
        )

        assert result.passes() is False

    def test_passes_method_with_warning_status(self):
        """Test passes() returns False for WARNING status."""
        result = ValidationResult(
            method="test",
            status="WARNING",
            bias=0.1,
            mse=0.3,
            coverage=0.92,
            ci_lower=0.05,
            ci_upper=0.15,
            n_simulations=1000,
            timestamp=datetime.now(),
            metadata={},
        )

        assert result.passes() is False

    def test_summary_method(self):
        """Test summary() generates readable output."""
        result = ValidationResult(
            method="bias",
            status="PASS",
            bias=0.025,
            mse=0.154,
            coverage=0.952,
            ci_lower=-0.048,
            ci_upper=0.098,
            n_simulations=1000,
            timestamp=datetime(2025, 11, 13, 12, 30, 0),
            metadata={},
        )

        summary = result.summary()

        # Check key information is present
        assert "bias" in summary
        assert "PASS" in summary
        assert "0.0250" in summary  # Formatted bias
        assert "0.1540" in summary  # Formatted MSE
        assert "95.20%" in summary  # Formatted coverage
        assert "1,000" in summary  # Formatted n_simulations

    def test_summary_includes_confidence_interval(self):
        """Test summary includes confidence interval bounds."""
        result = ValidationResult(
            method="mse",
            status="PASS",
            bias=0.01,
            mse=0.12,
            coverage=0.95,
            ci_lower=-0.02,
            ci_upper=0.04,
            n_simulations=2000,
            timestamp=datetime.now(),
            metadata={},
        )

        summary = result.summary()

        assert "-0.0200" in summary  # CI lower
        assert "0.0400" in summary  # CI upper


class TestValidationResultMetadata:
    """Test ValidationResult metadata handling."""

    def test_metadata_preserved_in_serialization(self):
        """Test metadata dict is preserved through serialization."""
        metadata = {
            "dgp_type": "nonlinear",
            "n": 5000,
            "p": 20,
            "confounding": 1.5,
        }

        result = ValidationResult(
            method="test",
            status="PASS",
            bias=0.01,
            mse=0.1,
            coverage=0.95,
            ci_lower=-0.02,
            ci_upper=0.04,
            n_simulations=1000,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        # Serialize and deserialize
        json_str = result.to_json()
        restored = ValidationResult.from_json(json_str)

        assert restored.metadata == metadata
        assert restored.metadata["dgp_type"] == "nonlinear"
        assert restored.metadata["n"] == 5000

    def test_empty_metadata(self):
        """Test ValidationResult works with empty metadata."""
        result = ValidationResult(
            method="test",
            status="PASS",
            bias=0.0,
            mse=0.1,
            coverage=0.95,
            ci_lower=-0.05,
            ci_upper=0.05,
            n_simulations=1000,
            timestamp=datetime.now(),
            metadata={},
        )

        assert result.metadata == {}
        assert isinstance(result.metadata, dict)
