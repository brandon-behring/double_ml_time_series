"""
Shared test configuration: 4-tier system with tiered parameters.

Tier definitions:
    tier1: Unit tests — no estimation, <100ms each, pure logic
    tier2: Integration — light estimation, multi-component, <60s each
    tier3: Validation — moderate MC/bootstrap, statistical properties, <5min each
    tier4: Full replication + stress — full MC, published results, <30min each

Running tiers:
    pytest -m tier1                      # 30s smoke test
    pytest -m "tier1 or tier2"           # 2min pre-push
    pytest -m "tier1 or tier2 or tier3"  # 30min nightly
    pytest                               # Full suite (~2h)
"""

import os

# Force sequential execution in tests to avoid multiprocessing hangs
# under Python 3.13's stricter inspect.signature() in forked workers.
# Must be set BEFORE any src/ imports that read DML_N_JOBS at module level.
os.environ.setdefault("DML_N_JOBS", "1")

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from dml_ts.validation.bootstrap_config import BootstrapConfig

# ---------------------------------------------------------------------------
# Tier-specific parameter configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierConfig:
    """Parameters for a specific test tier.

    Attributes:
        n_simulations: Number of Monte Carlo simulations (0 = no MC)
        n_bootstrap_ci: Bootstrap iterations for confidence intervals
        n_bootstrap_bias: Bootstrap iterations for bias estimation
        n_estimators: Number of estimators for RF/ensemble methods
        n_jobs: Parallelism level (-1 = auto, 1 = serial)
        timeout: Per-test timeout in seconds
    """

    n_simulations: int
    n_bootstrap_ci: int
    n_bootstrap_bias: int
    n_estimators: int
    n_jobs: int
    timeout: int

    def bootstrap_config(self) -> BootstrapConfig:
        """Create a BootstrapConfig matching this tier."""
        return BootstrapConfig(
            n_bootstrap_bias=self.n_bootstrap_bias,
            n_bootstrap_ci=self.n_bootstrap_ci,
        )


TIER_CONFIGS = {
    "tier1": TierConfig(
        n_simulations=0,
        n_bootstrap_ci=0,
        n_bootstrap_bias=0,
        n_estimators=0,
        n_jobs=1,
        timeout=10,
    ),
    "tier2": TierConfig(
        n_simulations=3,
        n_bootstrap_ci=5,
        n_bootstrap_bias=5,
        n_estimators=10,
        n_jobs=1,
        timeout=60,
    ),
    "tier3": TierConfig(
        n_simulations=50,
        n_bootstrap_ci=50,
        n_bootstrap_bias=100,
        n_estimators=50,
        n_jobs=1,
        timeout=300,
    ),
    "tier4": TierConfig(
        n_simulations=200,
        n_bootstrap_ci=500,
        n_bootstrap_bias=1000,
        n_estimators=100,
        n_jobs=48,
        timeout=1800,
    ),
}


# ---------------------------------------------------------------------------
# Timeout enforcement via pytest-timeout markers
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Apply per-tier timeouts and enforce tier-marker discipline.

    Tests marked with @pytest.mark.tierN receive timeout = TIER_CONFIGS[tierN].timeout.
    More specific tier markers override less specific ones.

    Collection FAILS if any test lacks a tier marker — an unmarked test would be
    invisible to the tiered CI gates and only run in full-suite contexts (see
    CONTRIBUTING.md). The `--collect-only` step in the CI lint job is the
    enforcement point that sees every test regardless of `-m` filtering.
    """
    unmarked: list[str] = []
    for item in items:
        # Find the most specific tier marker on this test
        has_tier = False
        for tier_name in ("tier4", "tier3", "tier2", "tier1"):
            marker = item.get_closest_marker(tier_name)
            if marker is not None:
                timeout = TIER_CONFIGS[tier_name].timeout
                # Only set timeout if not already explicitly overridden
                if item.get_closest_marker("timeout") is None:
                    item.add_marker(pytest.mark.timeout(timeout))
                has_tier = True
                break

        if not has_tier:
            unmarked.append(item.nodeid)

    if unmarked:
        preview = "\n  ".join(sorted(unmarked)[:20])
        more = f"\n  ... and {len(unmarked) - 20} more" if len(unmarked) > 20 else ""
        raise pytest.UsageError(
            f"{len(unmarked)} test(s) lack a tier marker (tier1/tier2/tier3/tier4). "
            f"Every test must declare its tier — see CONTRIBUTING.md:\n  {preview}{more}"
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tier2_bootstrap_config() -> BootstrapConfig:
    """Light bootstrap config for integration tests."""
    return BootstrapConfig.tier2()


@pytest.fixture
def tier3_bootstrap_config() -> BootstrapConfig:
    """Moderate bootstrap config for validation tests."""
    return BootstrapConfig.tier3()


@pytest.fixture
def tier2_config() -> TierConfig:
    """Full tier2 parameter set."""
    return TIER_CONFIGS["tier2"]


@pytest.fixture
def tier3_config() -> TierConfig:
    """Full tier3 parameter set."""
    return TIER_CONFIGS["tier3"]


@pytest.fixture
def tier4_config() -> TierConfig:
    """Full tier4 parameter set."""
    return TIER_CONFIGS["tier4"]


@pytest.fixture
def real_401k_data() -> pd.DataFrame | None:
    """Load cached 401(k) dataset from test/fixtures/.

    Returns None if the cached file does not exist (network tests skipped).
    """
    csv_path = Path(__file__).parent / "fixtures" / "401k_data.csv"
    if not csv_path.exists():
        pytest.skip(
            f"Cached 401(k) data not found at {csv_path}. Run: python test/fixtures/download_401k.py"
        )
    return pd.read_csv(csv_path)
