# Contributing

This is a book-grade companion repository for a Double Machine Learning
manuscript — not a production library. `docs/CURRENT_STATUS.md` is the only
current status source.

## Development setup

The repo venv is uv-managed:

```bash
uv venv venv --python 3.13
VIRTUAL_ENV=venv uv pip install -e ".[dev,docs]"
venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
```

`uv.lock` is the reproducibility source of truth. The `temporalcv` dependency is
a git tag pin (`@v2.0.0`) — environments need git access to resolve it.

## Gates (run from the repo venv)

```bash
venv/bin/python -m ruff check dml_ts/ test/ examples/
venv/bin/python -m ruff format --check dml_ts/ test/ examples/
venv/bin/python -m mypy dml_ts/ --ignore-missing-imports --no-strict-optional --explicit-package-bases
venv/bin/python -m pytest -m "tier1 or tier2" --no-cov -q
for f in examples/*.py; do venv/bin/python "$f"; done
venv/bin/python -m sphinx -b html -W --keep-going docs/sphinx docs/sphinx/_build/html
```

Book (when `chapters/` changes): `lualatex -shell-escape main.tex` + `biber` —
fatal TeX errors are blocking, and any `chapters/*.tex` edit must re-stamp its
MDX twin (`scripts/check_tex_mdx_drift.py` enforces this).

## Test tiers

Every test must carry exactly one tier marker:

- `tier1` — unit, no estimation, <100ms (pre-commit hook)
- `tier2` — integration, light estimation, <60s (pre-push hook + PR CI)
- `tier3` — validation, moderate MC/bootstrap, <5min (nightly)
- `tier4` — full replication/stress, <30min (weekly)

## Error-handling policy

Never fail silently. Degenerate numerics (non-finite SEs, empty clusters,
skipped windows) must raise or warn loudly — fabricating neutral-looking output
(t=0, p=1.0) is the bug class this repo has been bitten by twice (#7, #9).

## Versioning policy (pre-PyPI)

This package has no external consumers; the book's code listings are the
compatibility contract and move atomically with the code in the same PR.
Under this policy, **minor versions may remove internal or companion APIs**
(strict semver resumes at the PyPI debut). Versions must agree across
`pyproject.toml`, `docs/sphinx/conf.py`, `CITATION.cff`, and
`dml_ts.__version__` — `test/test_version.py` enforces this. Direct-URL (git)
dependencies are not accepted by PyPI; the temporalcv pin converts to a normal
version constraint at its PyPI debut.

## Commits

See the commit pattern in `STYLE.md`. One focused PR per concern; PRs adding or
changing inference code get adversarial review (correctness, overclaim,
silent-failure, test-adequacy) before merge.
