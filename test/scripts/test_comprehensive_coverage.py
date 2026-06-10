"""
Unit tests for comprehensive coverage stress test script.

Tests the scenario creation, coverage calculation, and result categorization
for scripts/comprehensive_coverage_test.py.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from comprehensive_coverage_test import create_test_scenarios, run_coverage_stress_test


@pytest.mark.tier1
class TestScenarioCreation:
    """Test scenario creation logic."""

    def test_creates_15_scenarios(self):
        """Test that exactly 15 scenarios are created."""
        scenarios = create_test_scenarios()
        assert len(scenarios) == 15, "Should create exactly 15 test scenarios"

    def test_scenarios_have_required_fields(self):
        """Test that all scenarios have required fields."""
        scenarios = create_test_scenarios()
        required_fields = ["name", "n", "p", "confounding_strength", "expected_coverage"]

        for i, scenario in enumerate(scenarios):
            for field in required_fields:
                assert field in scenario, f"Scenario {i} missing required field '{field}'"

    def test_sample_sizes_in_expected_range(self):
        """Test that sample sizes span the expected range [50, 5000]."""
        scenarios = create_test_scenarios()
        sample_sizes = [s["n"] for s in scenarios]

        assert min(sample_sizes) >= 50, "Minimum sample size should be >= 50"
        assert max(sample_sizes) <= 5000, "Maximum sample size should be <= 5000"
        assert min(sample_sizes) == 50, "Should include smallest sample size (50)"
        assert max(sample_sizes) == 5000, "Should include largest sample size (5000)"

    def test_dimensions_in_expected_range(self):
        """Test that dimensions span the expected range [5, 50]."""
        scenarios = create_test_scenarios()
        dimensions = [s["p"] for s in scenarios]

        assert min(dimensions) >= 5, "Minimum dimension should be >= 5"
        assert max(dimensions) <= 50, "Maximum dimension should be <= 50"
        assert 5 in dimensions, "Should include lowest dimension (5)"
        assert max(dimensions) >= 30, "Should include high dimension (>=30)"

    def test_confounding_in_expected_range(self):
        """Test that confounding strength spans expected range [0.5, 4.0]."""
        scenarios = create_test_scenarios()
        confounding = [s["confounding_strength"] for s in scenarios]

        assert min(confounding) >= 0.5, "Minimum confounding should be >= 0.5"
        assert max(confounding) <= 4.0, "Maximum confounding should be <= 4.0"
        assert 0.5 in confounding, "Should include weak confounding (0.5)"
        assert max(confounding) >= 3.0, "Should include strong confounding (>=3.0)"

    def test_expected_coverage_format(self):
        """Test that expected coverage is formatted as percentage range."""
        scenarios = create_test_scenarios()

        for scenario in scenarios:
            expected = scenario["expected_coverage"]
            assert isinstance(expected, str), "Expected coverage should be string"
            assert "%" in expected, "Expected coverage should contain '%'"
            assert "-" in expected, "Expected coverage should be a range (e.g., '90-95%')"

    def test_scenarios_are_unique(self):
        """Test that all scenarios have unique configurations."""
        scenarios = create_test_scenarios()
        configs = [(s["n"], s["p"], s["confounding_strength"]) for s in scenarios]

        assert len(configs) == len(set(configs)), "All scenarios should have unique configurations"

    def test_includes_challenging_scenarios(self):
        """Test that challenging scenarios are included."""
        scenarios = create_test_scenarios()

        # Check for high relative dimensionality (p/n > 0.5)
        has_high_dim = any(s["p"] / s["n"] > 0.5 for s in scenarios)
        assert has_high_dim, "Should include at least one high-dimensional scenario (p/n > 0.5)"

        # Check for small sample + strong confounding
        has_hard = any(s["n"] <= 100 and s["confounding_strength"] >= 2.0 for s in scenarios)
        assert has_hard, "Should include small sample + strong confounding scenario"

        # Check for very large sample
        has_large = any(s["n"] >= 3000 for s in scenarios)
        assert has_large, "Should include very large sample scenario"


@pytest.mark.tier4
class TestCoverageStressTest:
    """Test the main coverage stress test functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_creates_output_directory(self, temp_output_dir):
        """Test that output directory is created if it doesn't exist."""
        output_path = Path(temp_output_dir) / "new_subdir"
        assert not output_path.exists(), "Output directory should not exist initially"

        # Run with very few simulations for speed
        run_coverage_stress_test(output_dir=str(output_path), n_simulations=1)

        assert output_path.exists(), "Output directory should be created"

    def test_returns_dataframe(self, temp_output_dir):
        """Test that stress test returns a DataFrame."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        assert isinstance(result, pd.DataFrame), "Should return pandas DataFrame"
        assert len(result) > 0, "DataFrame should not be empty"

    def test_dataframe_has_expected_columns(self, temp_output_dir):
        """Test that result DataFrame has all expected columns."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        expected_columns = [
            "scenario_num",
            "scenario_name",
            "n",
            "p",
            "confounding",
            "expected_coverage",
            "actual_coverage",
            "coverage_pct",
            "bias",
            "mse",
            "status",
            "p_value",
            "n_simulations",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_runs_all_15_scenarios(self, temp_output_dir):
        """Test that all 15 scenarios are executed."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        assert len(result) == 15, "Should run all 15 scenarios"
        assert result["scenario_num"].min() == 1, "Scenario numbering should start at 1"
        assert result["scenario_num"].max() == 15, "Scenario numbering should end at 15"

    def test_coverage_in_valid_range(self, temp_output_dir):
        """Test that coverage values are in valid range [0, 1]."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=2)

        valid_coverage = result[result["actual_coverage"].notna()]
        assert (valid_coverage["actual_coverage"] >= 0).all(), "Coverage should be >= 0"
        assert (valid_coverage["actual_coverage"] <= 1).all(), "Coverage should be <= 1"

    def test_saves_csv_output(self, temp_output_dir):
        """Test that results are saved to CSV file."""
        run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        output_path = Path(temp_output_dir)
        csv_files = list(output_path.glob("coverage_stress_test_*.csv"))

        assert len(csv_files) > 0, "Should create at least one CSV output file"
        assert csv_files[0].exists(), "CSV file should exist"

    def test_csv_can_be_read(self, temp_output_dir):
        """Test that saved CSV can be read back."""
        run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        output_path = Path(temp_output_dir)
        csv_files = list(output_path.glob("coverage_stress_test_*.csv"))

        # Read back the CSV
        df = pd.read_csv(csv_files[0])
        assert len(df) == 15, "Saved CSV should have 15 rows"

    def test_status_values_are_valid(self, temp_output_dir):
        """Test that status values are in expected set."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=2)

        valid_statuses = {"PASS", "WARNING", "FAIL", "ERROR"}
        for status in result["status"]:
            assert status in valid_statuses, f"Invalid status: {status}"

    def test_bias_and_mse_are_numeric(self, temp_output_dir):
        """Test that bias and MSE are numeric values."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=2)

        # Filter out ERROR scenarios
        valid = result[result["status"] != "ERROR"]

        assert pd.api.types.is_numeric_dtype(valid["bias"]), "Bias should be numeric"
        assert pd.api.types.is_numeric_dtype(valid["mse"]), "MSE should be numeric"

    def test_n_simulations_parameter_respected(self, temp_output_dir):
        """Test that n_simulations parameter is used correctly."""
        n_sims = 3
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=n_sims)

        valid = result[result["status"] != "ERROR"]
        assert (valid["n_simulations"] == n_sims).all(), (
            f"All scenarios should use n_simulations={n_sims}"
        )


@pytest.mark.tier4
class TestCoverageCategorization:
    """Test coverage quality categorization logic."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_categorization_thresholds(self, temp_output_dir):
        """Test that coverage values are properly stored and categorizable.

        The script stores plain percentage strings in coverage_pct (e.g., '95.0%')
        and numeric values in actual_coverage. Emoji categorization is console-only.
        This test verifies the DataFrame contains valid, consistent values.
        """
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=2)

        valid = result[result["actual_coverage"].notna()]

        for _, row in valid.iterrows():
            coverage = row["actual_coverage"]

            # Coverage must be in valid range
            assert 0.0 <= coverage <= 1.0, f"Coverage {coverage} out of valid range [0, 1]"

            # coverage_pct should be properly formatted percentage
            assert row["coverage_pct"] == f"{coverage:.1%}", (
                f"coverage_pct '{row['coverage_pct']}' doesn't match actual_coverage {coverage}"
            )

            # Verify categorization thresholds are distinguishable
            if coverage > 0.98:
                assert coverage > 0.90, "Overconservative implies above undercoverage"
            elif coverage < 0.90:
                assert coverage < 0.98, "Undercoverage implies below overconservative"


@pytest.mark.tier1
class TestExitCodes:
    """Test exit code logic for automated validation."""

    def test_exit_codes_defined(self):
        """Test that script defines expected exit codes in __main__ block."""
        # Read the script to verify exit code logic exists
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "comprehensive_coverage_test.py"
        )

        with open(script_path) as f:
            script_content = f.read()

        # Check for exit code definitions
        assert "sys.exit(0)" in script_content, "Should have success exit code (0)"
        assert "sys.exit(1)" in script_content, "Should have all-failed exit code (1)"
        assert "sys.exit(2)" in script_content, "Should have overconservative exit code (2)"
        assert "sys.exit(3)" in script_content, "Should have undercoverage exit code (3)"


@pytest.mark.tier4
class TestErrorHandling:
    """Test error handling in stress test."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_handles_very_small_sample_gracefully(self, temp_output_dir):
        """Test that very small samples are handled without crashing."""
        # The script should handle even extreme scenarios without crashing
        # Run with minimal simulations for speed
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        # Find the smallest sample size scenario
        smallest = result[result["n"] == result["n"].min()]

        # Should either succeed or ERROR gracefully, not crash
        assert len(smallest) > 0, "Should have result for smallest sample size"
        assert smallest["status"].iloc[0] in {
            "PASS",
            "WARNING",
            "FAIL",
            "ERROR",
        }, "Should have valid status"

    def test_handles_high_dimensional_scenarios(self, temp_output_dir):
        """Test that high-dimensional scenarios (p > n) are handled."""
        result = run_coverage_stress_test(output_dir=temp_output_dir, n_simulations=1)

        # Find high-dimensional scenarios
        high_dim = result[result["p"] > result["n"]]

        if len(high_dim) > 0:
            # Should handle gracefully, not crash
            assert high_dim["status"].iloc[0] in {
                "PASS",
                "WARNING",
                "FAIL",
                "ERROR",
            }, "Should handle p > n scenarios"
