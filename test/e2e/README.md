# End-to-End Tests

E2E tests verify complete validation workflows from start to finish.

## Purpose (Hub Pattern: testing.md Layer 5)

E2E tests verify the complete user journey:
- Full 7-method validation battery execution
- LaTeX report generation
- Figure output
- Results persistence

## Running

```bash
pytest test/e2e/ -v --timeout=300
```

Note: E2E tests may take longer (5+ minutes for full Monte Carlo).

## Test Scenarios

1. **Full Validation Run**: Execute all 7 validation methods
2. **Report Generation**: Verify LaTeX output compiles
3. **Reproducibility**: Same seed produces identical results

## Adding Tests

1. Name files `test_[scenario].py`
2. Use realistic parameters (may need `@pytest.mark.slow`)
3. Clean up generated files after test
