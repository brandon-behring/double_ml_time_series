# /validate-dml

Run the DML validation test suite with coverage.

## Actions

1. Run pytest on the validation module:
```bash
pytest test/validation/ -v --cov=src/validation --cov-report=term-missing
```

2. Report:
   - Total tests passed/failed
   - Coverage percentage
   - Any uncovered lines

3. If coverage < 80%, flag for attention

## Expected Output

```
Test Results:
- Passed: X
- Failed: Y
- Coverage: Z%

[Details of any failures]
```
