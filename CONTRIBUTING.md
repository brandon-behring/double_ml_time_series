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

`uv.lock` records the locked resolution (refresh with `uv lock` after dependency
changes); the install commands above and CI resolve fresh from `pyproject.toml`,
which is why the gate tools are pinned exactly there (`ruff==0.15.16`,
`mypy==2.1.0`). The `temporalcv` dependency resolves from PyPI
(`temporalcv>=2.0.0,<3`); `uv.lock` records the exact version, and the
golden-parity suite gates behavioral drift on any upgrade.

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

Every test must carry a tier marker — enforced at collection time: the
`test/conftest.py` hook raises `pytest.UsageError` listing any unmarked tests
(the CI lint job's bare `--collect-only` step covers the full tree). Class- and
method-level markers may overlap; the highest tier found on the item wins —
tier4 > tier3 > tier2 > tier1:

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
compatibility contract, and the policy is that they move atomically with the
code in the same PR (the standing pre-policy violations in chapters 6 and 8
are tracked in #12).
Under this policy, **minor versions may remove internal or companion APIs**
(strict semver resumes at the PyPI debut). Versions must agree across
`pyproject.toml`, `docs/sphinx/conf.py`, `CITATION.cff`, and
`dml_ts.__version__` — `test/test_version.py` enforces this. The temporalcv
dependency converted from a git tag pin to a normal version constraint
(`>=2.0.0,<3`) at temporalcv's PyPI debut (2026-06-12, dml_ts v1.1.1);
`dml_ts` itself remains pre-PyPI, but no direct-URL dependency blocks a
future upload anymore.

## Commits

See the commit pattern in `STYLE.md`. One focused PR per concern; PRs adding or
changing inference code get adversarial review (correctness, overclaim,
silent-failure, test-adequacy) before merge.
