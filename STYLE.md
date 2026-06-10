# Style

Conventions for this repo. The canonical cross-project playbook lives in the
personal hub (`lever_of_archimedes/patterns/` — library-design playbook,
universal-vs-unique boundary, git commit pattern); this file is the
repo-local distillation.

## Python

- **ruff** for lint + format, line length 100, `target-version = py311`
  (config in `pyproject.toml`; `ruff==0.15.16` pinned exactly in the dev extra —
  CI installs `.[dev]` and pre-commit runs the venv's ruff, so all three gates
  share that single pin).
- **mypy 2.1.0** with the flags used in CI
  (`--ignore-missing-imports --no-strict-optional --explicit-package-bases`).
- Type hints on all function signatures; NumPy-style docstrings on public
  functions.
- **Fail loud**: raise explicit `ValueError`/`TypeError` with context; never
  convert numerical failure into neutral-looking output. Warnings
  (`warnings.warn(..., stacklevel=2)`) only for survivable conditions the user
  must know about.
- Imports use the `dml_ts` namespace (`from dml_ts.<subpkg> import ...`);
  never reintroduce `src.*`.

## Boundaries

- Causal interpretation (estimators, known-theta DGPs, assumption tracking)
  lives here; universal time-series mechanism (CV splitters, HAC, simulators,
  numeric validators) belongs upstream in `temporalcv` — upstream-first when in
  doubt.
- `dml_ts/production/` is research/demo utilities only; never describe this
  repo as production software.

## Tests

- Real tests only — no stubs or vacuous assertions (`>= 0.0` on a rate is the
  canonical anti-pattern; it masked #7).
- Every test carries a tier marker; where class- and method-level markers
  overlap, the most specific (highest) tier wins — `test/conftest.py` resolves
  tier4 > tier3 > tier2 > tier1 (see CONTRIBUTING.md).
- Exact-value pins for formula contracts (rtol <= 1e-10), behavioral brackets
  for scale sanity, `pytest.raises`/`pytest.warns` for the fail-loud paths.

## Book

- LaTeX `chapters/*.tex` is canonical; `web/` MDX is a drift-guarded port —
  change both in the same commit.
- Escape underscores in prose (`\_`); fatal TeX errors block.
- Code listings in chapters are the public compatibility contract — keep them
  executable against the current package. (Known violations: chapter 6's
  cluster-SE listing and chapter 8's PanelDML output block, tracked in #12.)

## Commits

`type: description` (feat/fix/refactor/test/docs/chore/migrate), body explains
why, trailer:

```text
Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```
