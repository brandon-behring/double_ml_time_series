# Integration Tests

Integration tests verify multi-module workflows in the DML validation pipeline.

## Purpose (Hub Pattern: testing.md Layer 4)

Integration tests verify that modules work together correctly:
- DGP generator → Bias validation pipeline
- LASSO diagnostic → Bootstrap workflow
- Multiple baseline comparisons in sequence

## Running

```bash
pytest test/integration/ -v
```

## Test Structure

Each test should:
1. Use realistic data sizes (not toy examples)
2. Verify end-to-end data flow
3. Check intermediate outputs
4. Use explicit assertions about correctness

## Adding Tests

1. Name files `test_[workflow].py`
2. Document the workflow being tested
3. Include setup/teardown for stateful operations
