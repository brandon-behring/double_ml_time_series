# /build-latex

Compile the LaTeX book with zero-error verification.

## Actions

1. Run make to compile:
```bash
cd /home/brandon_behring/Claude/double_ml_time_series && make
```

2. Check for errors in main.log:
```bash
grep -E "^!" main.log || echo "No errors found"
```

3. Check for warnings:
```bash
grep -c "Warning" main.log
```

4. Report PDF size and page count:
```bash
ls -lh main.pdf
pdfinfo main.pdf | grep Pages
```

## Expected Output

```
Build Status: SUCCESS/FAILURE
- Errors: 0 (required)
- Warnings: N
- PDF: main.pdf (XXX KB, YY pages)
```

## Zero-Error Requirement

This project requires **zero LaTeX errors**. If errors exist:
1. Read main.log for error context
2. Fix the source file
3. Rebuild until clean
