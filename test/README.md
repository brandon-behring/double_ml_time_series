# Test Suite: 4-Tier System

## Tier Definitions

| Tier | Purpose | Timeout | Trigger |
|------|---------|---------|---------|
| **tier1** | Unit tests. No estimation, <100ms each. Pure logic, data structures, interfaces. | 10s | Every save / pre-commit |
| **tier2** | Integration. Light estimation (single DML run, small data). Multi-component paths. | 60s | Pre-push / PR |
| **tier3** | Validation. Moderate MC/bootstrap (50 sims, 50 bootstraps). Statistical properties. | 300s | Nightly / manual |
| **tier4** | Full replication + stress. Full MC (200+ sims, 500+ bootstraps). Published results. | 1800s | Weekly / release |

## Running Tests

```bash
# Tier 1 smoke test (~12s)
pytest -m tier1

# Pre-push check (~2min)
pytest -m "tier1 or tier2"

# Nightly validation (~30min)
pytest -m "tier1 or tier2 or tier3"

# Full suite including replication (~2h)
pytest
```

## Tiered Parameters

| Parameter | tier1 | tier2 | tier3 | tier4 |
|-----------|-------|-------|-------|-------|
| `n_simulations` | 0 | 3 | 50 | 200+ |
| `n_bootstrap_ci` | 0 | 5 | 50 | 500 |
| `n_bootstrap_bias` | 0 | 5 | 100 | 1000 |
| `n_estimators` (RF) | N/A | 10 | 50 | 100 |
| `n_jobs` | 1 | 1 | 1 | 48 |

Use `BootstrapConfig.tier2()` / `.tier3()` factory methods for consistent test parameters.

## Cached Data

The 401(k) dataset is cached at `test/fixtures/401k_data.csv` (~500KB).
To regenerate: `python test/fixtures/download_401k.py`

## Adding New Tests

1. Add `@pytest.mark.tierN` to every test class or method
2. Use tiered parameters from `test/conftest.py` (`TIER_CONFIGS` dict)
3. For bootstrap tests, use `BootstrapConfig.tier2()` or `.tier3()`
4. Timeouts are enforced automatically per tier via `conftest.py`
