# Land the backlog, with PanelDML redesigned first

The end-to-end review plan is complete (commit `66aef23`, issues #59 to #66). This plan replaces it.

## Context

The working tree of `double_ml_time_series` holds work from several sessions that was never
committed: 104 modified tracked files (+5,962/−10,136, forming 14 change-sets), 57 untracked paths
(420 files, 26 MiB: 5 package modules, 2 scripts, 45 test files, 5 audit folders), one unpushed
commit (`66aef23`), one pushed commit that never became a pull request (`0cca908`, #41 PR-A), and a
second worktree with a local-only branch (`audit/public-data-extensions-2026-11-21`, 4 commits).

Why it must be landed, and carefully:

- Nothing protects it. One mistaken command loses months of work.
- The only state ever tested is all of it together (1,173 tier1+tier2 tests pass). `66aef23`
  already depends on two uncommitted files, so it cannot be reproduced from the repository.
- The repository is **public**, so whatever lands is published.
- The review found a defect in the uncommitted `PanelDML` redesign: it removes unit effects from X
  before the learners see it (`_panel.transform_panel_fold`, called from
  `temporal_plr_dml.py:1007-1042`), which leaves bias about +1.04 and 0.3% coverage when
  confounding is nonlinear in X.

Outcome wanted: `main` holds the tested work in reviewable pull requests, each green on the full
CI gate; PanelDML lands only after a redesign that passes a rule fixed in advance; a final check
proves nothing was lost.

## Decisions

Yours, already made:

| Decision | Choice |
|---|---|
| Slicing | Themed pull requests, files whole where possible, each passing the full gate |
| PanelDML | Redesign and validate first; no PanelDML change lands before that |
| Merging | I stop before each merge and merge only on your go-ahead for that pull request |

Taken by default (change any at approval):

| Decision | Default | Why |
|---|---|---|
| Order | Preserve the tree, land what does not depend on PanelDML (Part I), then the redesign (Part II), then the rest (Part III) | Part I is independent of the redesign and shrinks the risk at once |
| Environment | Each slice worktree gets its own `venv/` (`uv venv venv --python 3.13`, `uv pip install -e ".[dev,docs,full]"`) | Hooks, Makefile and CONTRIBUTING all name `venv/`, so no tracked file changes; and `.venv` installs the package from the dirty tree, which would let a slice pass by importing files it does not contain |
| Audit folders under "redesign first" | Land their PanelDML-free subset now; hold 26 files of the 11-19 folder and 2 of the 11-21 folder | Landing them whole would put 7 scripts on `main` that cannot run and publish numbers from the rejected design |
| `66aef23` (end-to-end review) | Deferred whole to Part III | Its test imports `_panel` and calls the new PanelDML API |
| Third-party-derived data (Cricsheet records, GSS aggregates, SPF medians, journal ledger) | Before PR-4, read each source's published terms and record URL and date in `dataset_ledger.yml`; a file lands only if redistribution is permitted; otherwise it is held out and I ask | Public repository; five files carry no licence note |
| Absolute `/home/...` paths in two archived 11-21 receipts | Land unchanged | They are evidence, and the path shows only the public GitHub handle |
| `.claude/settings.json` (drops `Write(**)` from the allow list) | Not landed by me; stays a local change | A permission setting is your call |
| Adversarial review of inference-code PRs | A fresh Claude agent plus Gemini through `/ask-gemini`, given the diff and not my conclusions; Codex not used | As in the last pass |
| Closing issues | An issue is closed only when the review's stored check for it no longer reproduces on the merged `main`; otherwise the PR says "refs #N" | Several backlog changes touch a tracked defect without fixing it (#62, #61, #64 reproduce on the tree too) |
| Version | No release in Part I; `[Unreleased]` in `CHANGELOG.md` grows with each PR; 1.2.0 is decided after Part III | The breaking PanelDML change is in Part III |

## Part 0 — Preserve the tree (first, before anything else)

No commit is ever made in the dirty tree: a commit there makes pre-commit stash 104 files.

1. Build a snapshot commit of the whole tree (tracked changes and untracked files) with a temporary
   index, so the working tree and the real index are untouched and no hook runs:
   `GIT_INDEX_FILE=<outside repo>`, `git read-tree HEAD`, `git add -A`, `git commit-tree`, then
   `git update-ref refs/snapshots/backlog-2026-11-24 <commit>`.
2. Verify: file count equals `git ls-files | wc -l` plus 420; `git status` in the tree is unchanged.
3. `git bundle create ~/Claude/.backups/double_ml_time_series/backlog-2026-11-24.bundle` with the
   snapshot ref, the local branch at `66aef23` and the public-data branch; `git bundle verify`.
4. The snapshot ref lives outside `refs/heads` and is never pushed (no `--all`, no `--mirror`).
5. Run gitleaks once over the snapshot content, so PR-0's allowlist is complete.
6. Write the repository copy of this plan, `docs/plans/active/2026-11-24_backlog_landing.md`
   (it lands with PR-0 and carries the checklist).

## Part I — Land what does not depend on PanelDML

Every slice is built in its own worktree, `~/Claude/.worktrees/double_ml_time_series-<slug>/`, from
`origin/main`, by taking files from the snapshot
(`git restore --source=refs/snapshots/backlog-2026-11-24 --staged --worktree -- <paths>`).
`web/package*.json` is never taken (`origin/main` is ahead there).

| PR | Branch | Carries | Chapters | Refs | Review |
|---|---|---|---|---|---|
| PR-0 | `chore/landing-tooling` | `.gitleaks.toml` allowlist for `docs/research/(bib\|dataset)_ledger.yml`; `Makefile` PDF prerequisites; `book.yml` path filter; the plan's repository copy | — | — | none |
| PR-A | `fix/validation-fail-loud-41a` | The existing remote head `0cca908`, opened as it is; the local branch (which has `66aef23` on top) is never pushed | — | #41 | CI only |
| PR-1 | `fix/core-numerics-42` | `dml_ts/dml/{_utils,fwl,robinson,double_ml,dynamic_g_estimation}.py`; the 15 non-panel hunks of `temporal_plr_dml.py`; hunk 1 of `test/dml/test_temporal_plr_dml.py`; `test/dml/test_estimator_properties.py` | — | #42 | adversarial |
| PR-2 | `fix/validation-inference` | Bias-test correction (7 validation modules, `_simulation_inference.py`, 5 tests), its 6 scripts and 2 notebooks, bootstrap with fixed folds (`_bootstrap_folds.py`), 401(k) benchmarks, the two simulation generators, `test_binary_treatment_audit.py`, the fail-loud tests that belong to these files | 3 | #41, #61, #63 | adversarial |
| PR-3 | `fix/applications-pipeline` | Sensitivity relabelled as a log-gamma curve; pinned data loaders; atomic registry and storage (`_persistence.py`, `filelock`); the insurance pipeline and monitor; hunks 1 to 5 of `insurance_dgp.py` (row alignment, `treatment_history`); `test_scientific_oracles.py`, `test_storage.py`; `ASSUMPTIONS.md` hunk 1; 11-19 `chapter10_pipeline.py` and its CSV | 4, 9, 10 | #44, #60, #62, #64 | adversarial |
| PR-4 | `docs/november-audits` | `dml_ts/validation/effect_forecast.py` and its test; the mathematical review folder and its test; the 11-19 folder minus 26 held files, with 28 tests; the 11-21 folder minus 2 held files, the ACIC folder, 2 tests; ledgers and dataset notes; audit-index rows for what landed; `docs/CURRENT_STATUS.md` rewritten for the Part I state | 2, 5, `main.tex` | #65 | claims against evidence, by a fresh agent |
| PR-5 | `audit/public-data-extensions-2026-11-21` | The second worktree's 4 commits, rebased on `main` (its audit-index hunk will conflict and is resolved by hand); then that worktree is removed | — | — | licence check |

Each PR adds its own bullets to `CHANGELOG.md`; the snapshot's text is used where a bullet is
wholly about that PR and adapted where it mixes panel claims.

Consequences of "redesign first", stated so they are not surprises:

- **Seven files are split by hunk, not taken whole:** `temporal_plr_dml.py` (hunks 3 to 17 now;
  1, 2 and 18 are PanelDML), `test_temporal_plr_dml.py`, `insurance_dgp.py`,
  `test_validation_fail_loud.py` (by test), `ASSUMPTIONS.md`, `docs/audits/README.md`,
  `bib_ledger.yml`. The cuts fall on hunk boundaries; each is applied with
  `git apply --check` first.
- **`main` will hold new Chapters 2, 3, 4, 5, 9, 10 beside old Chapters 6 and 8.** A scripted check
  of that mixed state found no broken reference or citation key. Chapter 4's reproduction block
  cites `retail_check.py`, which is held until Part III; Chapter 8 keeps old numbers while the
  generator's row order is corrected under it.
- **Chapters ride with their code**, so prose in Chapters 3 and 4 cites audit files one or two PRs
  before those files land.
- **Nine tests wait for Part III**, among them `test_published_results.py` (it checks Chapters 6
  and 8 too).

Deliberate deviations from the snapshot in Part I (each a separate commit, each with a test):

1. **Baselines get a coverage verdict (PR-2).** In the tree, `OLSWithControls.validate()` returns
   PASS at 79.3% coverage and `NaiveOLS` passes, because the baselines never judge coverage
   (review rows F06d, S001). Route them through `classify_bias_and_coverage`
   (`bias_validation.py:81`), which the DML validator already uses.
2. **Two statements in Chapter 2 corrected (PR-4):** `chapter_02.tex:236` (individual consistency
   when the true effect is zero) and `:982` against `:273` (the claim that `cv = n` is optimal);
   ported by hand to the MDX, then re-stamped.
3. **Hand-written Part I text** in `CHANGELOG.md` and `docs/CURRENT_STATUS.md`, which says that
   PanelDML awaits redesign.

### Procedure for each pull request

1. Build the slice; assert `dml_ts.__file__` lies inside the worktree.
2. Run the gate locally (commands under Verification). If a slice fails because it needs a file
   from a later slice, the allowed repair is to move that file or hunk between slices and record
   it. Any code edit that is not in the snapshot is a deviation: I stop and ask unless it is one of
   the three listed above.
3. Commit per change-set (the hooks run, since `venv/` exists there), push, open the PR, wait for
   CI.
4. For PR-1 to PR-3: adversarial review; findings fixed as follow-up commits.
5. **Stop.** Report the gate output, the CI result and the review findings; merge on your go-ahead
   (merge commit, as PRs #28 to #47 were).
6. After the merge: run the review's stored check for each referenced issue against `main`;
   close or leave open by the rule above and verify the issue's state; update the plan's
   checklist; remove the worktree (`git fetch --prune`, confirm the merge, `git worktree remove`).

For PR-3, also run the tier3 class `TestValidateDGPRecovery` locally: the old PanelDML call on
re-aligned rows is a combination nobody has run. If it fails, I stop and ask.

### Exit from Part I

- **Conservation check.** `git diff --name-only <snapshot> origin/main` must be a subset of a
  written manifest: the Part III held files, the seven split files, the hand-written documents, the
  PR-0 files, `web/package*.json`, and the files touched by the three deviations. Held tracked files
  must equal `origin/main`'s old content; held new files must be absent.
- Save `git diff origin/main <snapshot>` as the Part III residual patch.
- Re-enable the `Nightly Tests` workflow (disabled for inactivity) and trigger it once by hand.

## Part II — Redesign PanelDML (a gated project)

What is kept from the uncommitted redesign: the fold designs, the three inference options,
`PanelDMLResult`, and `panel_standard_error` (checked to 1e-12 against statsmodels). What is
redesigned: how the nuisance learners see X, and `forward_date`'s use of training-period unit
means under feedback (review row M14).

1. **Design gate (I stop here for your choice).** Read in full the two sources the ledger already
   holds as verified: Clarke and Polselli, "Double Machine Learning for Static Panel Models with
   Fixed Effects" (The Econometrics Journal, 2025), and Fuhr and Papies, "Double Machine Learning
   meets Panel Data" (arXiv:2409.01266). Lay out the candidate designs with their assumptions,
   beside the one repair already measured (the basis built before demeaning removes the bias,
   +0.024 against +1.040, but covers 0.919). One option at this gate is to land
   `panel_standard_error` early, which would release the retail, minimum-wage and design-probe
   files.
2. **Protocol, frozen and hashed before any run.** Designs: a linear nuisance as control, quadratic
   and interaction confounding in X, unit effects correlated with X, two-way effects, an unbalanced
   panel, feedback, few units. Arms: `main`'s PanelDML, the snapshot's, each candidate, and
   true-nuisance. A pass rule fixed in advance, taken from the review's protocol; at least 1,000
   replications near nominal; the cost listed before the run.
3. **Implementation** with property tests, formula contracts and fail-loud tests; the choice of
   nuisance treatment is an explicit argument, never a silent default.
4. **Validation:** the study, an independent challenge, and adversarial review.
5. **Decision gate.** If a candidate passes, Part III begins. If none does, I report and you decide
   between a restricted, documented scope and holding PanelDML.

Reused: `simulator.py` and `scorecard.py` helpers (`learner`, `date_folds`, seeds from
`SeedSequence`, the byte-reproducible writer), `mutations.py` `rebind()`, and the panel designs in
`second_audit_checks/area1.py`, all in `docs/audits/2026-11-22_end_to_end_review/` (read from the
snapshot until that folder lands).

## Part III — Land what waited (outline; detailed when Part II ends)

| PR | Carries |
|---|---|
| PanelDML | `_panel.py` as redesigned, the 3 held hunks of `temporal_plr_dml.py`, both `__init__` files, `TestPanelDML`, golden snapshots, example, `mc_panel_dml_table.py`, user guide, `ASSUMPTIONS.md` hunk 2, Chapter 6 |
| Insurance panel and falsification | `insurance_dgp.py` hunks 6 to 8 (with the slice at the true regime period, row F02c, corrected), `mc_insurance_pricing.py`, `validate_falsification.py` (rows S302, S304 addressed), `nightly.yml`, Chapter 8 |
| Audits, remainder | The 26 held 11-19 files and 2 held 11-21 files, 9 held tests |
| End-to-end review | `66aef23` cherry-picked, with its panel rows regenerated; the other tool's audit folder (your decision then: its 200 `file:///home/...` links make its verifier machine-bound); final status documents; the finished plans moved to `docs/plans/archive/` |

Artifacts regenerated after the redesign, each by its own script: golden panel keys;
`panel-draws.csv` and Chapter 6's table (`mc_panel_dml_table.py`); the three insurance CSVs and
Chapter 8's tables (`mc_insurance_pricing.py`); `stress-*` (`validate_falsification.py`);
`adversarial/failures.json`, `adversarial/empirical.json`, `classical/results.json`, the turnout
numbers; `design-probes.json`; the review's `mutations.csv`, `kill-classes.csv` and the panel rows
of `second-audit-checks.csv`.

Last: the dirty tree is retired only after a tree built from it with a temporary index equals the
snapshot's, and `main` passes the final conservation check. Only paths the snapshot lists as added
are removed; `git clean` is never used. The snapshot ref and the bundle are kept until then.

## Hazards

- `git push origin` writes to two repositories (the canonical one and a mirror).
- PR CI installs `.[dev]` only, so EconML tests skip there and tier3 runs nightly; the local gate
  uses `.[dev,docs,full]`.
- `docs/` and `scripts/` are namespace packages: a slice run from the wrong directory imports the
  dirty tree's files.
- Coverage must stay at or above 70% on every partial state; nothing but the full tree has been
  measured.

## Verification

Local gate, run in each slice worktree with its own `venv/` (the CI commands of `tests.yml`,
`docs.yml`, `book.yml`, `deploy-web.yml`):

```bash
venv/bin/python -m ruff check dml_ts/ test/ examples/ && venv/bin/python -m ruff format --check dml_ts/ test/ examples/
venv/bin/python -m mypy dml_ts/ --ignore-missing-imports --no-strict-optional --explicit-package-bases
venv/bin/python -m pytest --collect-only -q && venv/bin/python scripts/check_tex_mdx_drift.py
venv/bin/python -m pytest -m tier1 --timeout=10 --no-cov -q
venv/bin/python -m pytest -m "tier1 or tier2" --timeout=60 --cov-fail-under=70 -q
for f in examples/*.py; do venv/bin/python "$f"; done
venv/bin/python -m pytest --doctest-modules dml_ts/ --override-ini="addopts="
venv/bin/python -m sphinx -b html -W --keep-going docs/sphinx docs/sphinx/_build/html
make                                        # when a chapter changes
(cd web && npm ci && npm run validate)      # when an MDX file changes
```

Then, across the plan:

- Part 0: snapshot file count, unchanged `git status`, `git bundle verify`.
- Each PR: gate output and CI run shown before the merge request; issue states checked after.
- End of Part I: the conservation check against the manifest; nightly run observed once.
- Part II: protocol hash predates the first run; study output byte-identical on a second run.
- End of Part III: `git diff <snapshot> origin/main` shows only listed deviations and regenerated
  artifacts.

The closing report of each part names every skipped step, every deviation and every claim not
verified.

## Not in this plan

Fixes for defects on `main` that the backlog does not already contain (#61, #62, #64, #66); the
default learner and fold decisions (#51, #53); series-mode g-estimation (#50); the gap map's
missing book sections; unifying `venv/` and `.venv` across the documents.

## Critical files

- `dml_ts/dml/temporal_plr_dml.py`, `dml_ts/dml/_panel.py` (untracked)
- `dml_ts/validation/insurance_dgp.py`, `dml_ts/validation/bias_validation.py`
- `test/validation/test_validation_fail_loud.py`
- `docs/audits/2026-11-19_time_series_dml/external/run.py`
- `.gitleaks.toml`, `.pre-commit-config.yaml`, `CHANGELOG.md`, `docs/CURRENT_STATUS.md`

---

# Execution record

Everything above this line is the plan as approved on 2026-11-24. Everything below is added while
the plan runs. Paths are relative to the repository root unless they start with `~`.

## Checklist

Part 0

- [x] Snapshot ref built and verified
- [x] Bundle written and verified
- [x] gitleaks run over the snapshot content
- [x] Repository copy of the plan written (this file, lands with PR-0)

Part I (a box is ticked when the pull request is merged and its issues are checked)

- [x] PR-0 `chore/landing-tooling` (#67, merged 2026-11-25 as `04b96db`)
- [x] PR-L `ci/lock-dependencies` (#73, merged 2026-11-25 as `f56fa08`; #70 stays open)
- [x] PR-A `fix/validation-fail-loud-41a` (#68, merged 2026-11-25 as `87d9191`; #41 stays open)
- [x] PR-1 `fix/core-numerics-42` (#69, merged 2026-11-25 as `257022d`; #42 stays open)
- [x] PR-1b `fix/silent-wrong-slopes-72` (#74, merged 2026-11-25 as `19a8911`; #72 closed, #42 stays open)
- [x] PR-2 `fix/validation-inference` (#78, merged 2026-11-25 as `dc917cf`; #41, #61, #63 stay open)
- [x] PR-1c `fix/series-labels-and-constant-scored-rows` (#79, merged 2026-11-27 as `c84ce72`; #75, #76 closed)
- [x] PR-3 `fix/applications-pipeline` (#80, merged 2026-11-26 as `8feb54c`; #60 closed, #44, #62, #64 stay open)
- [ ] PR-1d `fix/one-input-boundary` (#82, open, held for the owner's decision; #77; the input rule in every estimator but `PanelDML`)
- [x] PR-4 `docs/november-audits` (#83, merged 2026-11-26 as `4e61710`; #65, #70, #86 stay open)
- [x] PR-5 `audit/public-data-extensions` (#87, merged 2026-11-26 as `2415b95`)
- [ ] PR-6 `fix/monitor-alerts-64` (#89, open; #64)
- [x] Conservation check against the manifest (2026-11-27, against `c84ce72`: 234 paths, none unlisted)
- [x] Part III residual patch saved (2026-11-27: 179 patches, local backup folder)
- [ ] `Nightly Tests` re-enabled and observed once (dispatched once on 2026-11-27, tier 3 passed; the schedule does not fire, #88)

Part II

- [x] Design gate (owner's choice, 2026-11-27)
- [x] Protocol frozen and hashed (2026-11-27; SHA-256 in Addendum 6)
- [ ] Implementation
- [ ] Validation
- [ ] Decision gate (owner's choice)

Part III

- [ ] PanelDML
- [ ] Insurance panel and falsification
- [ ] Audits, remainder
- [ ] End-to-end review
- [ ] Final conservation check; dirty tree retired

## Part 0 record (2026-11-24)

- **Snapshot.** `refs/snapshots/backlog-2026-11-24` is commit `024708c`, tree `c560e48`, parent
  `66aef23`. It holds 818 files: 398 tracked plus 420 untracked. The working tree equals the
  snapshot tree (compared through the temporary index), and the output of `git status` was
  byte-identical before and after.
- **Bundle.** `~/Claude/.backups/double_ml_time_series/backlog-2026-11-24.bundle`, 45,410,212
  bytes, SHA-256 `f62ebff18a4c7025b86b24b8a066d162e2cec7e807f7c8278eb4518cb25ac1a2`. It carries
  the snapshot ref, `fix/validation-fail-loud-41a` at `66aef23` and
  `audit/public-data-extensions-2026-11-21` at `3dbfa63`. `git bundle verify` passed, and a fetch
  from the bundle into an empty repository gave the same tree id and 818 files.
- **Not in the snapshot.** Git-ignored files are not covered: `results/` (768 MB of run logs and
  receipts), `output/` (49 MB), `logs/`, build outputs and caches. They stay in the dirty tree.
  This plan never removes them.
- **gitleaks.** Version 8.30.0, run on the exported snapshot content with `main`'s configuration,
  reported 130 findings. All 130 are the `generic-api-key` rule firing on a `bibkey:` line:
  82 in `docs/research/bib_ledger.yml`, 47 in `docs/research/dataset_ledger.yml` and 1 in
  `docs/research/existing_content_urls.txt` (a file already on `main`). PR-0 allowlists these
  three paths. With the new configuration the hook's own command
  (`gitleaks protect --staged`) reports no leak on the snapshot's two ledgers.

- **Baseline of `main` under today's dependencies.** A fresh environment
  (`uv pip install -e ".[dev,docs,full]"`, the unpinned install that CI also performs) resolves
  numpy 2.5.5, scikit-learn 1.9.1, statsmodels 0.15.0 and temporalcv 2.3.0. The dirty tree was
  tested with numpy 2.4.6, scikit-learn 1.6.1, statsmodels 0.14.6 and temporalcv 2.2.0. On
  `origin/main` (`9a3aac4`) the fresh environment gives: ruff clean; 868 tests collected; tier1
  364 passed; tier1 + tier2 683 passed with 74.90% coverage; examples, doctests, Sphinx and the
  book build pass; **mypy fails with one error** at `dml_ts/validation/insurance_dgp.py:244`
  (numpy 2.5 types `0.1 * ndarray.mean()` as `float64`, and the other branch assigns a `float`).
  The same error appears on the full snapshot. The last CI run on `main` was on 2026-10-19.

## Slice manifest

The plan above gives counts. The lists below are the ones those counts refer to. They were
produced by read-only analysis in the planning session and are checked again when each slice is
built.

### Files split by hunk

Hunk numbers refer to `git diff origin/main refs/snapshots/backlog-2026-11-24 -- <file>`, except
for `insurance_dgp.py`, whose hunks are numbered against `0cca908` (PR-A).

| File | Lands in Part I | Waits for Part III |
|---|---|---|
| `dml_ts/dml/temporal_plr_dml.py` | hunks 3 to 17 (PR-1) | hunks 1, 2, 18 (imports, `PanelDMLResult`, `PanelDML`) |
| `test/dml/test_temporal_plr_dml.py` | hunk 1, the rolling-window count (PR-1) | hunk 2, `TestPanelDML` |
| `dml_ts/validation/insurance_dgp.py` | hunks 1 to 5 (PR-3) | hunks 6 to 8, `validate_dgp_recovery` |
| `test/validation/test_validation_fail_loud.py` | by test (PR-2) | the `n_sims` test |
| `ASSUMPTIONS.md` | hunk 1 (PR-3) | hunk 2, the PanelDML bullets |
| `docs/audits/README.md` | rows for landed folders (PR-4) | the remaining rows |
| `docs/research/bib_ledger.yml` | the working-tree entries (PR-4) | the entries added by `66aef23` |

### Whole files that wait for Part III

- Package: `dml_ts/dml/_panel.py`, `dml_ts/__init__.py`, `dml_ts/dml/__init__.py`.
- Tests: `test/golden/snapshots/estimators.json`, `test/golden/test_golden_parity.py`,
  `test/dml/test_formula_contracts.py`.
- Examples, scripts, workflow: `examples/example_panel_dml.py`, `scripts/mc_panel_dml_table.py`,
  `scripts/mc_insurance_pricing.py`, `scripts/validate_falsification.py`,
  `.github/workflows/nightly.yml`.
- Documents: `docs/sphinx/user_guide/time_series_dml.rst`, Chapters 6 and 8 (`.tex` and `.mdx`).
- Commit `66aef23` and the other tool's audit folder.

### The 26 held files of `docs/audits/2026-11-19_time_series_dml/`

- **Scripts (7):** `method_checks.py`, `retail_check.py`, `classical/replay.py`,
  `external/questions/minimum_wage.py`, `external/turnout/replay.py`, `adversarial/failures.py`,
  `adversarial/empirical.py`.
- **Artifacts (13):** `panel-draws.csv`, `insurance-panel-draws.csv`,
  `insurance-panel-summary.csv`, `insurance-rolling.csv`, `stress-draws.csv.gz`,
  `stress-summary.csv`, `retail-results.csv`, `retail-folds.csv`, `retail-baselines.json`,
  `adversarial/failures.json`, `adversarial/empirical.json`, `classical/results.json`,
  `external/turnout/household_counts.csv`.
- **Reports (6):** `09-panel-retail-insurance.md`, `panel-inference.md`,
  `10-property-tests-and-falsification.md`, `adversarial/failures.md`, `adversarial/empirical.md`,
  `questions/minimum-wage.md`.

Two more files of this folder, `chapter10_pipeline.py` and `chapter10-pipeline.csv`, land early
with PR-3 because Chapter 10 cites them.

### The 2 held files of `docs/audits/2026-11-21_evaluation_methods/`

`reproduction/design_probes.py` and `reproduction/design-probes.json`.

### The 9 held tests

| Test | Reason |
|---|---|
| `test/validation/test_classical_benchmarks.py` | imports `classical/replay.py` |
| `test/integration/test_question_minimum_wage.py` | executes `minimum_wage.py` at import |
| `test/integration/test_adversarial_failures.py` | calls the new PanelDML API |
| `test/integration/test_adversarial_empirical.py` | needs `empirical.py` and `empirical.json` |
| `test/integration/test_external_turnout_time_series.py` | imports the turnout replay |
| `test/validation/test_paired_stress.py` | imports `scripts.validate_falsification` |
| `test/validation/test_evaluation_audit_contracts.py` | imports `design_probes` |
| `test/integration/test_published_results.py` | its Chapter 6 and 8 cases need Part III |
| `test/validation/test_matched_temporal_experiments.py` | asserts `classical/replay.py` exists |

### Dependencies between the pull requests

- PR-2 needs PR-A: it renames an argument that PR-A's test file uses, and it carries PR-A's
  follow-up tests.
- PR-3 needs PR-1 (`theta_via_fwl(treatment=)` in the pipeline) and PR-2
  (`ValidationResult.storage_metadata` in the storage module).
- PR-4 needs PR-1 to PR-3: its audit scripts call the corrected functions, and one of its tests
  parses Chapter 9.
- PR-5 comes last: its audit-index hunk conflicts with PR-4's rewrite of the same section.

## Deviations and moves

Recorded here as they happen: every file or hunk moved between slices, and every edit that is not
in the snapshot.

- PR-0: the allowlist also names `docs/research/existing_content_urls.txt`. The plan named the two
  ledgers only; the scan found one match of the same kind in this third file.
- PR-0: one code edit that is not in the snapshot. `insurance_dgp.py:242` wraps the competitor-lag
  term in `float(...)` so that mypy passes under numpy 2.5 stubs. Without it the type-check job
  fails on every pull request. The generator's output is bit-identical before and after (checked
  for the three realism levels and three seeds). It is its own commit. The check is mypy itself;
  no unit test was added, because the edit changes a static type and no value.
- PR-0: `book.yml` gains two steps that are not in the snapshot: `actions/setup-python` 3.13 and
  `pip install "latexminted==0.6.0"`. The reason is obstacle 5 of the addendum below. It is its
  own commit, and the check is the `book` job itself.
- PR-1: first built on the PR-0 branch, not on `origin/main`, because the commit hook runs mypy
  whenever a Python file is staged and mypy fails on `origin/main` (see the baseline above).
  Nothing was pushed in that stacked form. PR-1 is rebased onto `origin/main` after PR-0 merges,
  and every later slice is built from `origin/main` as planned.
- Hooks: `pre-commit` is found through `PATH="$PWD/venv/bin:$PATH"` in each slice worktree,
  because the shim on this machine looks for `.venv/bin/pre-commit` and the worktrees use `venv/`.
  No file changes for this.

- PR-1, after the adversarial review: three edits that are not in the snapshot, in one commit
  with their tests (`test/dml/test_numerical_guards.py`, a new file).
  1. **The "no treatment variation" bound.** The snapshot raises when
     `norm(T_tilde) <= norm(T) * n * eps`. That bound accepts exact collinearity whenever the
     controls have a non-zero mean, because the rounding error of the fit grows with the level
     of the controls: with controls at level 1000, `fwl_estimate`, `double_ml` and
     `robinson_estimator` returned a slope of order 1e10 to 1e12 in 455 of 500 runs, where
     `main` raises. The bound is now the larger of `1e-6 * norm(T - mean(T))` and
     `sqrt(eps) * norm(T)`. It was chosen by measuring four candidate rules on residuals the
     library produces (script `rule_grid.py`, kept with the review files): it catches every
     degenerate case that any candidate catches, and it refuses healthy data only when the
     residual spread is below 1.5e-8 of the treatment's level. The signature of
     `theta_via_fwl` is unchanged, so later slices are not affected.
  2. **A guard in `PanelDML.fit` (9 lines).** `main`'s PanelDML passes the demeaned treatment
     to the shared check, so a treatment that the fixed effects absorb leaves rounding residue
     with no level to compare it with: the snapshot's bound returned a fit in 40 of 40 runs
     (10% with p < 0.05), where `main` raises in 40 of 40. The guard compares the demeaned
     treatment with the bound of the treatment before demeaning. This touches PanelDML before
     its redesign; without it PR-1 would change PanelDML's behavior, which the plan forbids.
     Part II's protocol must include an absorbed treatment as a case that raises.
  3. **R² of a constant target.** The snapshot returns 0.0 only when the computed variance is
     exactly zero. The mean of `np.full(n, 0.1)` is not 0.1, so R² became rounding noise
     (1.0, 0.0, -1295 or -1.3e31 for the same target). The three helpers now return 0.0 when
     the centered norm is at most `norm(y) * n * eps`.

  Consequence for the conservation check: `_utils.py`, `fwl.py`, `dynamic_g_estimation.py`,
  `temporal_plr_dml.py` and `CHANGELOG.md` differ from the snapshot by these edits, and
  `test/dml/test_numerical_guards.py` is not in the snapshot.

## Addendum, 2026-11-25: obstacles of the first day and their repairs

The plan above stands. This section records what stopped the work on 2026-11-24 and how it
resumed.

### State on the morning of 2026-11-25 (checked read-only)

- Part 0 was intact: the snapshot ref had no missing object, the bundle verified with the SHA-256
  recorded above, and the dirty tree equalled the snapshot by content hash.
- #67 (PR-0) was open with 4 commits; every check passed except `book`.
- #68 (PR-A) was open at `0cca908`; every check passed except `lint` (mypy).
- PR-1 existed locally only. Its code commit `54ceae8` was intact; its changelog commit was lost.

### Obstacles and repairs

| # | Obstacle | Cause | Repair |
|---|---|---|---|
| 1 | The last local commit (`effa80f`, PR-1's changelog) was gone, `fix/core-numerics-42` pointed at a missing object, and the session's scratch files (gate script, hunk filter, gate logs) were empty | The machine stopped uncleanly at 15:52:40 on 2026-11-24: the journal ends with no shutdown sequence, and seven object files written in the last seconds were zero bytes. The journal does not show why it stopped. | The seven empty files were deleted and nothing else; the branch was set back to `54ceae8` (its previous reflog value); the index of that worktree was reset, which left the changelog edit as a working-tree change. `git fsck --full` then reported no error and the bundle verified. |
| 2 | mypy fails on `main` and on the snapshot | CI installs unpinned dependencies; numpy 2.5.5 stubs type `0.1 * ndarray.mean()` as `float64` | The `float(...)` cast in PR-0 (see the deviations above) |
| 3 | No Python commit is possible on `origin/main` | The commit hook runs mypy | Slices wait for PR-0; PR-1 is rebased after PR-0 merges |
| 4 | `pre-commit` is not found in a slice worktree | The shim looks for `.venv/bin/pre-commit`; the worktrees use `venv/` | `PATH="$PWD/venv/bin:$PATH"` for commits and pushes |
| 5 | `book` fails on #67 | `ubuntu-latest` is Ubuntu 26.04, whose Python is 3.14. TeX Live 2025's bundled `latexminted-0.6.0` wheel defines `ArgParser.__init__(self, *, prog)`; `argparse.add_parser` in Python 3.14 passes `color=`, so `latexminted` raises `TypeError` and minted cannot highlight. The PyPI build of `latexminted 0.6.0` takes `**kwargs` and runs under Python 3.14. The job had last run in June. | In `book.yml`: `actions/setup-python` 3.13 and `pip install "latexminted==0.6.0"`, staying on `ubuntu-latest`. The fallback, if that run fails, is `runs-on: ubuntu-24.04`. |
| 6 | `lint` fails on #68 | Obstacle 2; #68 does not contain the cast | After PR-0 merges, #68 is closed and reopened, so that CI tests a new merge commit. A re-run would reuse the old one. The branch stays at `0cca908`. |
| 7 | The backlog was never tested with the versions CI installs | `uv.lock` pins numpy 2.4.6, scikit-learn 1.6.1, statsmodels 0.14.6, temporalcv 2.2.0 and econml 0.16.0; a fresh install gives 2.5.5, 1.9.1, 0.15.0, 2.3.0 and 0.17.1. No workflow uses the lock. `main` and PR-1 pass under the new versions. | A probe, described below, before any further slicing |
| 8 | The pre-push hook runs no tests | It calls `pre-commit run --hook-stage pre-push` with no ref range, so every hook reports "no files to check" | Not repaired in this plan. The local gate and CI are the checks. |

### Decisions taken on 2026-11-25 (the owner's)

| Decision | Choice |
|---|---|
| Merging PR-0 | Merged without a further stop once every check is green. PR-A, PR-1 and all later pull requests still stop for a go-ahead. |
| Book workflow | PyPI `latexminted` on `ubuntu-latest`; an `ubuntu-24.04` pin as the fallback |
| Dependency drift | Probe the full snapshot under today's versions; decide only if it fails |

### Dependency probe

The snapshot is exported with `git archive` to a directory under `~/Claude/.worktrees/`, a fresh
environment is built there (`.[dev,docs,full]`), `dml_ts` is asserted to be imported from that
directory, and tier1 + tier2, the examples and the doctests are run. The reference is 1,173 passed
in the tree's locked environment.

The rule was fixed before the run. If everything passes, the work continues unchanged. If anything
fails, each failure is classified (a pinned number that moved, an API change, a timeout) and
re-run in the locked environment to confirm that drift is the cause; the work then stops for a
choice between locking CI to `uv.lock` and regenerating the numbers.

### Probe result (2026-11-25)

Run on the exported snapshot (`024708c`, 818 files) with a fresh environment: numpy 2.5.5,
scikit-learn 1.9.1, statsmodels 0.15.0, temporalcv 2.3.0, econml 0.17.1. The package and the
namespace packages `docs/` and `scripts/` were imported from the export.

| Stage | Result |
|---|---|
| tier1 + tier2 | 1,169 passed, 5 failed, 8 skipped |
| examples | all ran |
| doctests | 50 passed, 12 skipped |

The five failures, classified:

- **Two are artifacts of the export, not drift.** `test_documented_example_runs[git_output]` and
  `[provenance]` in `test/validation/test_mathematical_review_checks.py` call `git rev-parse HEAD`,
  and the export had no git metadata. With git metadata added, the 72 cases of that test pass
  under the new versions.
- **Three are pinned numbers that moved, and scikit-learn is the sole cause.** They pass in the
  locked environment, and they pass in the probe environment when only scikit-learn is set back
  to 1.6.1.
  - `test/integration/test_adversarial_transport.py::test_archived_cricket_separates_effect_estimates_from_yearly_forecasts`:
    the archived `rmse["dml"]` is 12.5298; scikit-learn 1.9.1 gives 12.4767.
  - `test/integration/test_prospective_cricket.py::test_historical_freeze_replays_offline_and_power_is_for_separate_ranking`:
    the hashed issuance does not replay.
  - `test/validation/test_mathematical_review_checks.py::test_two_seed_replay_matches_archived_draws[refute_f3_trees]`:
    the replayed errors differ from the archived draws.

All three belong to PR-4. By the rule fixed before the run, slicing beyond PR-1 waits for the
owner's choice between locking CI to `uv.lock` and regenerating the numbers. PR-A and PR-1 do not
depend on it: both pass under the new versions. The probe covered tier1 + tier2 only; tier3 and
tier4 were not run under the new versions.

The logs are in `~/Claude/.backups/double_ml_time_series/gate-logs/probe-2026-11-25/`. The export
and its environment were removed.

### Other records of 2026-11-25

- The crash repair was carried out as planned; `git fsck --full` reported no error afterwards.
  The gate script and the hunk filter were restored from the previous session's transcript.
- The two hook findings (obstacles 4 and 8) were already tracked in `lever_of_archimedes` as
  #109 and #125. No new issue was filed; this repository's evidence was added to each as a
  comment.
- #68 was closed and reopened after PR-0 merged; all six checks pass at `0cca908`.

### Adversarial review of PR-1 (#69), 2026-11-25

Reviewers: a fresh Claude agent, which ran 12 experiments and a 12-row mutation table, and
Gemini in three scoped calls. Each was given the diff, the tree at the reviewed commit and the
text of #42, and no conclusion of the author. Every finding was re-run before it was accepted.

| Finding | Source | Verdict | Outcome |
|---|---|---|---|
| The bound `norm(T) * n * eps` misses exact collinearity when the controls have a level | Claude | Confirmed: 455 of 500 runs at level 1000 | Fixed (deviation above) |
| PanelDML accepts a treatment absorbed by its fixed effects | Claude | Confirmed: 40 of 40 runs | Fixed |
| R² of a constant target is rounding noise | Claude, Gemini | Confirmed | Fixed |
| Eleven changed behaviors are pinned by no test | Claude, Gemini | Confirmed by mutation | Tests added; 15 of 15 mutations of the guarded behaviors are caught |
| The check refuses a treatment whose variation is tiny beside its level | Claude, Gemini | Confirmed; by design | Documented in the changelog and pinned by a test |
| An all-zero or constant treatment bypasses the check | Gemini | Refuted: raises for every learner tried | none |
| `TemporalPLRDML` ignores a classifier nuisance | Gemini | Refuted: it accepts only the four named regressors | none |
| A classifier cannot be fitted on a float treatment | Gemini | Refuted: 0.0/1.0 targets fit | Pinned by a test |
| A wrapped classifier without `predict_proba` raises `AttributeError` | Gemini | Refuted: it raises the documented `ValueError` | Pinned by a test |
| Changelog: the classifier claim is too wide for `robinson_estimator`; the series-mode entry omits that default results change | Claude | Confirmed (`main` [2.031, 0.031], now [1.998, 0.060]) | Changelog corrected |
| A column-vector treatment or outcome broadcasts silently to a wrong slope | Claude | Confirmed on `main` (slope -2.6e-16 for 2.03) | Not in this plan; tracked issue |
| Estimates change with the units of the treatment when it enters as a feature (ridge with lags; the `"linear"` alias under scikit-learn 1.9) | Claude | Confirmed on `main` (2.05 against 3.08) | Not in this plan; tracked issue |
| `RollingWindowDML` returns empty results with only a warning (`window_size < 30`; invalid inner arguments) | Claude, Gemini | Confirmed: 0 windows fitted, one warning | Not in this plan; tracked issue |

The review files (diff, reviewer scripts, mutation runner, Gemini's answers) are under
`~/Claude/.backups/double_ml_time_series/review-pr1/`.

### Working rules added

- The gate script and the hunk filter live under
  `~/Claude/.backups/double_ml_time_series/landing-tools/`, and gate logs under
  `~/Claude/.backups/double_ml_time_series/gate-logs/`. The session's temporary directory does
  not survive a reboot.
- Commits are made with `git -c core.fsync=all`, and each slice branch is pushed as soon as its
  commits pass the hooks, so that no finished commit exists only on one disk.

## Addendum 2, 2026-11-25: decisions at the first stop, and the order from here

The plan above stands, with the amendments listed here. The first addendum's steps are done.

### State when this addendum was approved (checked 2026-11-25)

| Item | State |
|---|---|
| Part 0 | Intact. Snapshot `024708c` (818 files, tree `c560e48`); the dirty tree equals it |
| PR-0, #67 | Merged as `04b96db` |
| PR-A, #68 | Open at `0cca908`; six checks green |
| PR-1, #69 | Open at `87cd7c3`; six checks green; local gate 775 passed, coverage 75.33%; adversarial review done, three repairs and 76 test cases added |
| Dependency probe | 1,169 passed, 5 failed: 2 export artifacts, 3 pinned numbers that move with scikit-learn 1.9.1 (all PR-4 material); tracked as #70 |
| Issues | Filed #70, #71, #72; evidence added to #42 and #51 |
| The stop of 2026-11-24 | Explained: the kernel log of the next boot records a reset-pin assertion (`BP_SYS_RST_L`) |
| Copies | The bundle, the tools and a tar of the git-ignored `results/`, `output/`, `logs/` on a second disk. No off-machine copy yet |

### Decisions taken on 2026-11-25 (the owner's)

| Decision | Choice |
|---|---|
| Bound of the "no treatment variation" check in #69 | Keep the two-term bound: `max(1e-6·norm(T − mean T), sqrt(eps)·norm(T))` |
| The 9-line guard in `PanelDML.fit` (#69) | Keep it. Part II's protocol gets "an absorbed treatment must raise" as a required case |
| CI versions (#70) | Lock CI to `uv.lock`, plus one scheduled unpinned drift job that blocks nothing |
| Merging #68 and #69 | Both, once they are green under the lock |
| The lock pull request (PR-L) | Merged without a further stop when every check is green; the work stops if any test fails under the locked versions |
| Off-machine copy | The three local-only refs are pushed to the private mirror, by its URL alone, after its visibility reads "private" |
| The two silent wrong-number defects on `main` (#72, series-mode degeneracy) | Fixed now, in one small pull request after #69 (PR-1b) |
| Extra precautions against a repeat of the stop | None |

### Amendments to the plan above

- **Environment.** Each slice worktree's `venv/` is installed from the lock:
  `uv venv venv --python 3.13`, then `VIRTUAL_ENV=$PWD/venv uv sync --locked --active --all-extras`.
  The lock is what CI installs from PR-L on.
- **Part 0, step 4.** The snapshot ref is still never pushed to the public repository (no `--all`,
  no `--mirror`, never through `origin`). It is pushed to the private mirror alone, as a backup
  branch.
- **Decisions.** No PanelDML change lands before the redesign, with one exception: the guard in
  #69 that keeps PanelDML's behavior as on `main`. The go-ahead to merge was given in advance for
  #67, PR-L, #68 and #69; every later pull request stops for its own.
- **Part I, two new pull requests.** PR-L `ci/lock-dependencies` (refs #70, CI only, before #68)
  and PR-1b `fix/silent-wrong-slopes-72` (#72 and #42, adversarial review, after #69). PR-4 also
  refers to #70, which stays open until PR-4's three pinned tests pass in CI.
- **Deviations from the snapshot, added.** (4) From the review of PR-1, in #69: the two-term bound,
  the guard in `PanelDML.fit`, the R² rule for a constant target. (5) PR-L and PR-1b are not in
  the snapshot at all.
- **A held hunk can change a landed caller's behavior** (found in PR-1: the shared check changed
  while PanelDML's own hunk was held, and PanelDML lost its degeneracy check). Every slice that
  changes a shared helper is checked against the callers whose hunks are held.
- **Exit from Part I.** The manifest of the conservation check also lists the PR-L files
  (workflows, `uv.lock`, `CONTRIBUTING.md`) and the files touched by the deviations. The Part III
  residual is built per held file and hunk, applied with `git apply --3way`, and never as one
  `git diff origin/main <snapshot>`: that diff would revert every deviation (the bound, the
  guards, the locked workflow steps). Re-enabling `Nightly Tests` is also the first run of its
  locked install step.
- **PR-3** adds `filelock` to `pyproject.toml`; its lock change is made with `uv lock` there, not
  taken from the snapshot's `uv.lock`.
- **Hazards, added.** A push from the dirty tree runs the pre-push hook there, as a commit runs
  the pre-commit hook; both are done from a clean slice worktree only. From PR-L on, PR CI
  installs the `dev` extra from the lock, so EconML tests still skip there and tier3 runs
  nightly; the local gate installs every extra. The drift job is a scheduled workflow and is
  disabled by GitHub after 60 days without repository activity, as `Nightly Tests` was.

### Order from here

1. **Mirror backup.** First, because it is the cheapest reduction of risk.
2. **PR-L.** Local gate in a locked environment, CI green, then merge (merge commit).
3. **#68.** Closed and reopened so that CI builds a merge commit with the locked workflow; green;
   merge. #41 stays open ("refs").
4. **#69.** Its worktree environment is rebuilt from the lock and the full gate is run; closed and
   reopened; green; merge. #42 stays open (its state is in a comment there).
5. **PR-1b**, with adversarial review.
6. **PR-2 `fix/validation-inference`**, sliced while PR-1b waits (it needs PR-A only and touches
   none of PR-1b's files).
7. **Stop and report** PR-1b and PR-2. Neither merges without a go-ahead.
8. PR-3, PR-4, PR-5 and the exit from Part I as planned above.

### Mirror backup

- The mirror's visibility is read first through the GitHub API; the push happens only if it
  reads `private: true`.
- The push is made from a clean slice worktree, to the mirror's URL and not to `origin`. It
  creates three branches there: `backup/backlog-2026-11-24-snapshot` (the snapshot `024708c`),
  `backup/e2e-review-66aef23` and `backup/public-data-extensions-3dbfa63`.
- Verification: `git ls-remote <mirror> 'refs/heads/backup/*'` shows the three commits, and the
  same query against the public repository shows nothing.
- Not covered: the git-ignored `results/`, `output/`, `logs/` (on the two local disks only).
- The three mirror branches are deleted at the plan's final conservation check.

### PR-L: lock CI to `uv.lock`

Branch `ci/lock-dependencies` from `origin/main`. It changes no package code.

| File | Change |
|---|---|
| `uv.lock` | The project's own version line, `1.1.2 → 1.1.4`, and nothing else |
| `.github/workflows/tests.yml` (5 jobs), `docs.yml`, `nightly.yml` (2 jobs) | The `pip install -e ".[…]"` step becomes `astral-sh/setup-uv` and `uv sync --locked --extra …`, and `.venv/bin` goes on `GITHUB_PATH`, so every later command stays as it is |
| `.github/workflows/drift.yml` (new) | Weekly schedule and `workflow_dispatch`; unpinned `pip install -e ".[dev,full]"`; prints the resolved versions; runs mypy and tier1 + tier2. It is not a pull-request check |
| `CONTRIBUTING.md` | CI installs from the lock; the local equivalent; how to bump a version |

Not verified until the exit from Part I: the edited `nightly.yml`, because that workflow is
disabled and is re-enabled there. `drift.yml` can be dispatched only once it is on `main`.

### PR-1b: two silent wrong numbers on `main`

Branch `fix/silent-wrong-slopes-72` from `origin/main` after #69 merges.

1. **Column vectors (#72).** `fwl_estimate(y, t[:, None], x)` returns −2.6e-16 for a slope of
   2.03, because `np.sum(Y_tilde * T_tilde)` broadcasts to an `(n, n)` matrix.
   `_utils.validate_lengths` (shared by `double_ml`, `robinson_estimator`, `TemporalPLRDML`,
   `RollingWindowDML`) also requires `Y` and `T` to be one-dimensional; `fwl_estimate` calls it
   in place of its inline copy; `theta_via_fwl` raises unless both residual arrays are
   one-dimensional and of equal shape. A column vector is rejected with a message that names the
   shape and suggests `.ravel()`. It is not flattened: `TemporalPLRDML` already rejects it, and a
   silent reshape is what this repository avoids.
2. **Series-mode g-estimation.** `DynamicGEstimationDML._fit_series` solves the treatment Gram
   system with no degeneracy check: `T = 0.5·x0` returns `theta_t = [4.4e13, 0.21]`, and a
   constant treatment ends in `LinAlgError`. Each lag column is checked with the bound #69
   introduced (`_utils.no_variation_tolerance`), with the message panel mode uses; the
   conditioning warning moves before the solve; a `LinAlgError` becomes a `ValueError` that names
   the treatment Gram.
3. **Tests.** `(n, 1)` outcome and treatment raise in each of the five estimators and `(n,)`
   gives unchanged numbers; in series mode a constant treatment and `T = 0.5·x0` raise for the
   ridge and linear learners, and the golden-parity numbers do not move. A mutation check as in
   #69.
4. `Closes #72` only if the script in #72 raises on the merged `main`.

### Working rules added by the first two days

- Tools and logs live in `~/Claude/.backups/double_ml_time_series/` (`landing-tools/`,
  `gate-logs/`, `review-pr1/`); the session's temporary directory is lost at every restart.
- Nothing is written into the dirty tree, and no command run from it uses a relative redirect.
  Parallel jobs are separate background commands. After any work near it, the status hash is
  compared.
- Reviewers get the diff and an export of the commit, never a live worktree. Every review finding
  is re-run before it is accepted (four claims of one reviewer on #69 were false). Scratch
  scripts do not catch exceptions broadly.
- `journalctl -k -b <n> | grep "reset reason"` names the cause of an earlier stop.

### Record of 2026-11-25, after the approval

- **Mirror backup: done.** The mirror read `private: true`; the three branches were pushed from
  the clean `fix/core-numerics-42` worktree; `ls-remote` shows `024708c`, `66aef23` and `3dbfa63`
  on the mirror and no `backup/*` or `refs/snapshots/*` ref on the public repository. The dirty
  tree's status hash was unchanged afterwards.
- **A second unclean stop,** at 11:17:56, between the mirror push and PR-L. The kernel log of the
  next boot gives the same cause as on 2026-11-24 (`system reset pin BP_SYS_RST_L was tripped`).
  Nothing was lost: no commit was in flight, the dirty tree's status hash was unchanged, and the
  tools directory was intact.
- **The lock.** `uv lock` (uv 0.13.7, no `--upgrade`) reports one change, `dml-ts 1.1.2 → 1.1.4`,
  as the dry run did, but the file it writes differs in 46 lines: it also raises the lock format
  from `revision = 3` to `5`, reorders `resolution-markers`, and drops redundant markers in the
  four forked Sphinx entries. A comparison by content (`landing-tools/lockdiff.py`) finds the
  same 112 packages, sources, hashes and dependency edges. Because the plan allowed the version
  line only, the lock in PR-L is `main`'s lock with that one line changed; `uv lock --check`
  accepts it and leaves it untouched. The format rewrite will appear with the next plain
  `uv lock` (PR-3's `filelock`) and is to be a commit of its own there.
- **`setup-uv` is pinned by commit.** Its repository has no floating `v10` tag, only full
  versions; `v10.4.0` is commit `1db35dd8eee11c5894872e0b383ae6c406cf738d`, and its README pins
  the same way. The uv version is pinned too (`0.13.7`, the version that checked the lock).
- **Three edits beyond the table above,** each a consequence of replacing pip:
  `cache: 'pip'` is removed from `actions/setup-python` in all three workflows (with no pip
  install the cache directory does not exist, and `setup-python@v6` then prints a warning on
  every job); `uv.lock` joins the path filter of `docs.yml`, since the lock now decides the
  documentation's environment; and the comment on the `doctest` job no longer gives a numpy
  downgrade as its reason, since under the lock an extra cannot move a version. In `nightly.yml`
  the held Part III hunk still applies with `git apply --3way --check`.
- **PR-L merged (#73, `f56fa08`).** Local gate in the locked environment: 683 passed, coverage
  74.86%, examples, doctests and Sphinx pass. CI: six checks green; the job logs show uv 0.13.7,
  the `setup-python` interpreter (CPython 3.13.18) and the lock's versions (scikit-learn 1.6.1,
  numpy 2.4.6, statsmodels 0.14.6, temporalcv 2.2.0). On `main`'s own lock, `uv sync --locked`
  failed as intended before the change ("The lockfile at `uv.lock` needs to be updated").
- **First drift run** (dispatched by hand on `main` after the merge): scikit-learn 1.9.1, numpy
  2.5.5, statsmodels 0.15.0, temporalcv 2.3.0, econml 0.17.1; mypy clean; 683 passed. So `main`
  itself does not depend on the lock yet; the three numbers that move are in PR-4's tests (#70).
- **#68 and #69 merged under the lock** (`87d9191`, `257022d`), each closed and reopened first so
  that CI built a new merge commit: 691 and 783 passed, coverage 75.91% on the second. #69's
  branch also passed the full local gate in an environment rebuilt from the lock (775 passed,
  75.28%); its earlier gate had run under the unpinned versions. #41, #42, #70 and #72 were
  checked open after the merges.
- **PR-1b, one expectation of the plan corrected by measurement.** The plan expected series mode
  to raise for `T = 0.5·x0` under the ridge learner too. It cannot under the two-term bound: a
  ridge fit leaves shrinkage residue of norm 0.06 where the bound is 8.7e-6. In 300 simulated
  series (truth 1.0) that fit reported a lag-0 effect with median absolute value 15.3, median
  standard error 16.6, and p < 0.05 in 17.7% of the runs. `double_ml` and `robinson_estimator`
  with a ridge learner accept the same treatment, but their standard errors (median 31.7 and
  39.9) never gave p < 0.05. PR-1b therefore covers the exact cases (linear learner, constant
  treatment, linearly dependent lags), and the ridge case is put to the owner at the stop.
- **PR-1b, what #72 also covers.** With both the outcome and the treatment as `(n, 1)`,
  `double_ml` returned 1.93 where the flat inputs give 2.04, and `robinson_estimator` 1.93 with a
  standard error of 0.003; `RollingWindowDML` returned zero windows for any column vector.
  `TemporalPLRDML` and `PanelDML` already raised, late, from the HAC routine.
- **PR-1b opened as #74; CI found what the local gate did not.** The first version turned a
  `LinAlgError` from the solver into the promised error. For an alternating treatment the Gram
  matrix is singular only up to rounding: locally the solver met a zero pivot, on CI's machine it
  did not and the later inversion raised a raw `LinAlgError`. The dependence of the lags is now
  tested on their residuals (scaled singular values), not through a solver exception. Rule for
  later slices: a guard must not rest on whether LAPACK meets an exact zero.
- **Adversarial review of #74** (a fresh Claude agent and Gemini; Gemini's first, wide call
  timed out with no text, two narrow calls answered; every claim re-run). Accepted and repaired:
  the series guard looked only at residuals, so a lag constant on every scored row and lags that
  a nonlinear learner does not reproduce exactly passed it; no more scored rows than lags ended
  in an unrelated error; no test pinned the size of the bound; a 0-d outcome raised `TypeError`;
  the message's hint failed on a DataFrame; three changelog sentences were incomplete. Rejected
  after re-running: two of Gemini's claims. The series guard is now one helper,
  `_refuse_unidentified_lags`. Final head `7089c0f`: local gate 882 passed, coverage 76.02%;
  six CI checks green; 29 of 29 mutations caught; the reviewer's scan of dependent lags gives
  200 of 200 refusals where it had found 36.
- **Three issues filed from that review, for defects that predate #74:** #75 (a pandas `Series`
  with a non-default index is read by label in `double_ml`, `fwl_estimate` and
  `robinson_estimator`; P1), #76 (`TemporalPLRDML` returns a significant estimate when the
  treatment is constant on every scored row; P1), #77 (flattened matrices in
  `DynamicGEstimationDML`, truncated panel identifiers, masked arrays; P2). The evidence on the
  approximating-learner case is a comment on #42. None of the three is in this plan; whether #75
  and #76 are fixed before PR-3 is the owner's decision at the stop.
- **PR-2 opened as #78.** The 41 files of the slice were taken whole (for each, `main` equals
  the content the snapshot was built on) and passed the gate at the first attempt; the five
  commits of snapshot work reproduce the gated tree `d7e992c`. One test of the snapshot is held
  with the insurance hunks (`test_validate_dgp_recovery_rejects_nonpositive_nsims`).
- **Deviation 1 had a consequence the plan did not list.** With the baselines judging coverage,
  Chapter 3's Experiment 3.1 no longer prints what its table shows (PASS and WARNING become FAIL
  and FAIL), and its text said the OLS labels use the bias test alone. The listing, the table
  and two paragraphs were changed in both editions, in the same pull request. The baselines'
  bias verdict also changes, because they now share the DML validator's rule. Rule for later
  slices: before a deviation, search the chapters for what the changed function prints.
- **A commit message with a wrong reason** (`d296a91`, the 401(k) p-value) was corrected in the
  message of `d59f35b`, since commits are not amended here without the owner's word.

## Addendum 3, 2026-11-25 (evening): decisions at the second stop

The plan above stands. Steps 1 to 7 of addendum 2 are done.

### Decisions taken at the second stop (the owner's)

| Decision | Choice |
|---|---|
| Merging #74 and #78 | Both |
| #75 (a pandas `Series` read by label) and #76 (`TemporalPLRDML`, a treatment constant on the scored rows) | One small pull request before PR-3 (PR-1c) |
| A treatment the controls determine, under a learner that only approximates it (17.7% significant in series mode under ridge) | Deferred to #42 and #51; documented only |
| The clause "a bias above 0.15 is a FAIL", which the baselines now share (20.7% false bias verdicts for naive OLS in Experiment 3.1) | Kept; decided under #56 |

### Order from here

1. Merge #74, then #78, with their stored checks on `main`.
2. PR-1c, with mutation check and adversarial review.
3. PR-3 `fix/applications-pipeline`, sliced while PR-1c is in CI and review.
4. Stop and report PR-1c and PR-3. Neither merges without a go-ahead.
5. Then the licence check of the third-party-derived files, PR-4, PR-5 and the exit from Part I.

### Amendments for PR-3, found while planning it

- **`insurance_dgp.py` has nine hunks against `main`, not eight.** The second
  (`_generate_competitor_treatment`) is the reverse of PR-0's `float(...)` cast and is never
  applied. PR-3 takes hunks 1, 3, 4, 5 and 6; hunks 7 to 9 (`validate_dgp_recovery`) stay held.
- **`uv.lock` is made with `uv lock`, in two commits:** the format rewrite alone (revision 3 to
  5, no package moved), then `filelock`.
- **The listings of Chapters 4, 9 and 10 are run on the slice** before the chapters are taken,
  since `test_published_results.py` is held and nothing else checks them.

### Record of the second stop's first step

- **#74 merged as `19a8911`.** On the merged `main` the script of #72 raises in `fwl_estimate`,
  `double_ml` and `robinson_estimator`, and the flat inputs give 2.0335 and 2.0411. #72 closed
  through its bare closing line. The series-mode item is struck in the status comment on #42.
- **#78 merged as `dc917cf`,** after it was closed and reopened so that CI tested it merged with
  #74 (eight checks green). On the merged `main`, `OLSWithControls` and `NaiveOLS` on
  `DGPGenerator(n=200, p=5, true_effect=1.0, random_state=42)` with 300 simulations report
  coverage 0.76 and 0.34, bias PASS, coverage FAIL, status FAIL. #41, #61 and #63 are open.
- **The state of #75 and #76 on that `main`,** before PR-1c (`gate-logs/pr1c/stored_checks.py`):
  `double_ml` returns 1.8198 for a shuffled frame and 2.0181 for a sorted one where the arrays
  give 2.0299 and 2.0383; with labels 50..349 against 0..299 `fwl_estimate` returns -0.1539,
  `robinson_estimator` -0.1535 and `double_ml` a `KeyError`; `TemporalPLRDML` returns 2.09 with a
  standard error of 0.12 for a treatment that is 1 on the first 60 of 400 rows.

### Record of PR-1c's review (2026-11-25, night)

Reviewed by a fresh Claude agent (diff and export, 78 tool calls, a reversed-diff copy as
"before") and by Gemini in two calls. The agent's report said not to merge the first version
(`19e5951`); its evidence was re-run and holds. The pull request was reworked.

- **What the first version got wrong.** Addendum 3 prescribed `np.asarray(..., dtype=float)` for
  `Y`, `T` and `X`. For `X` that was a regression: `robinson_estimator` had passed a DataFrame
  or a sparse matrix to the nuisance model as given, and the conversion broke pipelines that
  select columns by name or encode a string column, refused sparse controls in `double_ml`
  too, and changed one categorical case silently (1.9905 to 2.0211). For `Y` and `T` it made
  three kinds of input return a number that had raised before: timedeltas and datetimes
  (slopes of 7.3e3, 7.3e6, 7.3e9 by storage unit), complex values (imaginary part dropped),
  and masked arrays in `fwl_estimate`.
- **What the rework does** (two departures from Addendum 3's wording, both to be confirmed at
  the stop):
  1. Only `Y` and `T` are read, by value, and a dtype that is not a number is refused with
     the argument's name (datetime, timedelta, complex, text, masked entries). Two-dimensional
     `X` is not converted; a list becomes an array and a one-dimensional `X` one column.
  2. Two pandas inputs with different indexes are refused, where the addendum expected "the
     positional slope": Gemini's case (`fwl_estimate(df.y, shuffled.t, X)` = 0.17 against
     2.03) shows that pairing by position is wrong when the labels are the same units in
     another order.
- **Texts corrected.** Only `double_ml` indexed by label (shuffled 1.82, sorted 2.018, reversed
  1.888 against 2.03 to 2.04; a filtered frame raised `KeyError`); `fwl_estimate` and
  `robinson_estimator` were wrong only for different labels (-0.15). The slope of a treatment
  constant on the scored rows is not meaningless: over 400 draws it is centered on the truth
  with a standard error about 30% too small (coverage 0.845).
- **Limits of the scored-rows check, measured and not repaired here.**
  - It refuses numerical constancy only. Neighbouring designs are fitted with the same numbers
    as before: a switch at row 71 (coverage 0.843), three treated rows (0.777), a switch at row
    398 (0.292), one treated row (0.075). Filed as #81.
  - `PanelDML`: the refusal reaches it only when the demeaned treatment is exactly constant
    on the scored rows. A constant non-zero dose in every scored unit leaves rounding residue
    (1.1e-16) and is fitted: 7.47 with a standard error of 3.48, and 4.52 with 1.13, for a
    slope of 2.
  - `RollingWindowDML`: a fit whose windows are all skipped returns empty arrays with a
    warning (the open #42 item). One input that returned a single window before (2.06 with
    0.35) now reaches it.
- **Part II protocol, two required cases added:** a panel whose treatment varies only in units
  that the split never scores, and a panel whose scored units each have one constant dose.
  Each must raise or be estimated with a valid interval. The cause is unit-major rows handed
  to a time-series split: the block that is trained on and never scored is the first units.
- **Evidence for #77,** there before and unchanged: `cross_fit_predictions` and `theta_via_fwl`
  still read labels when called directly (8.25 maximum difference; -0.034 against 1.954), and
  the exported `fwl_residualize` returns a labelled `Series`.
- **Mutation check:** 29 reversions, among them the four that survived the first version's
  tests (the check only for `n_lags=0`, only for one split strategy, exact in place of
  numerical constancy, integer treatments read as category codes); all 29 are caught.

### Record of PR-3 and its review (2026-11-25 and 26)

PR-3 (`fix/applications-pipeline`, #80) was sliced as Addendum 3 describes: 31 files whole,
`insurance_dgp.py` by hunk (1, 3, 4, 5, 6 of 9), `ASSUMPTIONS.md` hunk 1, `uv.lock` made with
`uv lock` in two commits. Chapter listings were run on the slice before the chapters were taken.
The stored check of #60 passes on the branch; #44, #62 and #64 stay open.

Reviewed by Gemini in four calls (21 findings, 4 false on re-run) and by a fresh Claude agent
(15 groups of findings, all re-run, all hold; it found the core numerics correct).

- **Repaired on the branch, each with a test that a mutation check shows to bite (15 of 15):**
  a moved or renamed `ResultStorage` returned no results and no error; a store or registry opened
  with a relative path pointed at another directory after `os.chdir`; published files were mode
  0600; an error after the index update could delete an indexed payload; `frequency="MS"` or
  `"ME"` with a year-over-year transform raised `KeyError`; an interrupted OJ download propagated
  unwrapped; the changelog claimed more relabelling of the sensitivity module than the backlog did.
- **Not repaired, recorded on issues:** the monitor's alerts (#64), the loader's stale rows for
  business-day series requested daily (#62), degenerate designs that the pipeline fits and the
  two-step promotion (#41), tests that cannot fail (#56, #66).
- **Two decisions raised at the stop.**
  1. Chapter 8 on the aligned generator rows. Its `PanelDML` listing prints -0.8553 with a
     standard error of 0.0330 (reproduced on `main`); on the branch the same call gives -0.8531
     with 0.0175 and an interval that excludes the true -0.8. Its exercise covers the truth in 21%
     of 100 draws on the branch and 56% on `main`; the chapter prints 93.0%. On `main` the
     generator pairs one product's outcome with another's index, so the printed numbers come from
     misaligned rows.
  2. Through the pipeline a randomized binary treatment is reported as critical, where it was
     "ok" or "warning": the branch hands the monitor an R² where it handed accuracy, and the
     thresholds (0.50 and 0.30) are `main`'s.
- **For Part III:** Chapter 8 is recomputed there; `validate_dgp_recovery`'s docstring promises 93
  to 97% coverage and its tests assert only `abs(bias) < 1.0`.
- **Lessons.** A fix the plan prescribes can regress valid inputs (PR-1c converted `X`); compare
  before and after on a matrix of input kinds. A mutation list should hold the conditions a guard
  must not depend on. A held chapter can stop reproducing when a generator it reads is repaired.

## Addendum 4, 2026-11-26: decisions at the third stop, one input rule, and the next stretch

Recorded when the stop of PR-1c (#79) and PR-3 (#80) was closed. The plan above stands; this
section says what was decided and what is executed next.

### Decisions taken at the third stop (the owner's)

| Decision | Choice |
|---|---|
| #80 and Chapter 8 | Merged as it was. Chapter 8's printed numbers stop reproducing until Part III recomputes them; the changelog says so and PR-4's status document repeats it |
| The monitor's "critical" status for a randomized treatment | Repaired under #64 after the merge, not in #80 |
| #79 | Reviewed against a long-term standard (the input contract below). Its two departures from Addendum 3 stand: the controls are not converted, and different labels are refused |
| Index rule, long term | Two pandas inputs with different indexes are refused, in every estimator |
| Sequence | Independent review of the reworked #79; merge if no regression for a valid input is found; then a follow-up pull request ("PR-1d") carries the rule to the other estimators. `PanelDML` waits for its redesign |

### The input contract (written once)

Measured on 2026-11-26 in the locked environment: for a treatment with the same labels in another
order, `fwl_estimate`, `robinson_estimator` and `double_ml` on #79 refuse, while `TemporalPLRDML`,
`RollingWindowDML` and `PanelDML` return -0.18, -0.03 and -0.18 for a slope near 2.1, with no
error. statsmodels 0.14.6 refuses the same input (`index.equals`); scikit-learn 1.6.1, EconML
0.16.0, `DoubleMLData.from_arrays` 0.11.3 and linearmodels 7.0 pair by position (about -0.11).

1. **One boundary.** Every public estimator in `dml_ts/dml/` reads its inputs through one
   function (`_utils.as_validated_arrays`), with one contract and one test matrix (#77).
2. **`Y` and `T`** are read by value as floats. What conversion would misread is refused by the
   argument's name: datetime, timedelta, complex, text, masked entries, anything not
   one-dimensional.
3. **Rows are paired by position, and unequal pandas indexes are refused.** Labels are never used
   to reorder rows: a time-series estimator depends on row order. Known cost: an input rebuilt
   without its index (a fresh `RangeIndex` beside a date index, rows in the same order) is refused
   too; the message names `.to_numpy()` as the way through.
4. **Controls.** Handed to a learner unchanged (`robinson_estimator`, `double_ml`): not converted,
   rows selected by position. Computed on by the estimator (`fwl_estimate`, `TemporalPLRDML`,
   `DynamicGEstimationDML`): converted to floats under the refusals of point 2.
5. **Identifiers** (units, periods, groups) are compared as given or refused, never cast. This
   part belongs to the PanelDML redesign.
6. **Later, not in this plan:** an entry that takes one DataFrame and column names (the form
   `DoubleMLData` uses). It is put to the Part II design gate as an option.

### Order from here

1. Merge #80; run the stored check of #60 on `main`; close #60 if it passes.
2. Independent review of the reworked #79; merge on its outcome; stored checks of #75 and #76.
3. PR-1d `fix/one-input-boundary` (refs #77), from `origin/main` after step 2.
4. Licence check, then PR-4 `docs/november-audits`.
5. PR-5 (the public-data branch's 4 commits), stacked on PR-4's branch.
6. Stop and report PR-1d, PR-4 and PR-5. None merges without a go-ahead.
7. Then the exit from Part I and the Part II design gate.

Added to the Part II protocol: `PanelDML`'s inputs pass the same boundary (unequal indexes
refused; identifiers compared as given or refused, so `1.00 ... 1.29` cannot become one unit),
and the design gate lists a frame-based entry as an option.

### Record of step 1 (2026-11-26)

- #80 merged as `8feb54c`; the merged tree equals the reviewed head's (`4e24d6b`).
- The review's stored check of #60 (`pipeline_check.py`, 200 seeds per design), run on the merged
  `main`, is byte-identical to the result recorded for `7e8fa51`: no errors, SE/SD 0.94 to 0.96
  and coverage 0.945 to 0.97 in all six designs. #60 was closed by hand; #44, #62 and #64 stay
  open. The slice worktree was removed.

### Hazards added by this stop

- A stop report must say which version a review covered. "Reviewed" was written for #79 when only
  its first version had been reviewed independently.
- A parametrization that leaves an estimator out hides a failure (`double_ml` with a DataFrame of
  controls). A matrix lists every estimator, and a known failure is an explicit expectation.
- A rule applied at some entry points is a contract the caller cannot rely on. A new input rule
  is written once and its remaining entry points are listed in the same pull request.

### Record of the second review of PR-1c (2026-11-26)

The reworked #79 (`6f31d25`) was reviewed by a fresh Claude agent, given the diff and two
exports and none of the author's conclusions (61 tool calls, a 2,932-cell before-and-after
matrix, 59 mutations), and by Gemini in one narrow call (9 findings; 2 false on re-run). Every
finding below was re-run before it was accepted.

**The rule for the merge was not met.** The go-ahead held if no valid input regressed and no
wrong number was returned by the three functions. The review found wrong numbers that the rework
had introduced, so #79 was repaired on its branch and waits for a new go-ahead.

- **New misread, repaired.** The reworked reader took an object column to floats with
  `np.asarray(values, dtype=float)`. NumPy and pandas cast a timezone-aware datetime, and object
  arrays of `np.complex128`, `np.timedelta64`, `np.datetime64`, `bytearray` or `memoryview`
  values, without an error: a timezone-aware outcome returned 7,347.87 (seconds), 7.3e6
  (milliseconds) and 7.3e12 (nanoseconds) where the floats give 2.04, and `main` raised on all
  of these (33 cells of the matrix). An object column is now judged by its elements: real
  numbers and missing values pass, anything else is refused by the argument's name and the
  element's type. An integer too large for a float is refused by name as well.
- **Equal values under other level names, repaired.** `Index.equals` ignores names. An outcome
  indexed by (unit, period) and a treatment indexed by (period, unit) over a square panel have
  equal tuples in other rows: `double_ml` raised `KeyError` on `main` and returned 0.045 for a
  slope of 1.85 on the reworked branch. Indexes whose levels are named differently are refused.
- **The refusal's advice, reworded.** The message recommended aligning by label first. For rows
  that are in the same order under other labels that mispairs them (1.9531 against 2.0335);
  `.to_numpy()` is now named first.
- **The intended cost, confirmed and now stated in the changelog.** `fwl_estimate` and
  `robinson_estimator` refuse calls that were paired correctly before: the same rows under
  other labels (10 of 168 index combinations).
- **False statements, corrected in the changelog, the docstrings and the tests.**
  - "A DataFrame keeps its column names and a sparse matrix stays sparse" holds for
    `robinson_estimator` only. `double_ml` raises `KeyError` for every DataFrame of controls
    (before and after), and `fwl_estimate` does not take a sparse matrix. A test pins the
    `KeyError` until PR-1d selects rows by position.
  - The changelog said that a constant non-zero dose in a panel "is still fitted". That holds
    when each scored unit has a dose of its own, the case the first review measured (the
    demeaned treatment is residue near 1e-16 that differs between units), and for 129 of 200
    shared doses in an unbalanced panel. It is false for one dose shared by all scored units of
    a balanced panel: every unit leaves the same residue, so the treatment is constant there
    and all 200 doses tried are refused. "Is now refused" for the zero-outside panel holds for
    unit effects only: under time and two-way effects it still returns 4.50 and 4.43 for a
    slope of 2.
  - A test docstring gave 4.4 with 1.0 for five treated units; `main` returns 4.32 with 3.16.
    Another said the three split strategies score different rows; two of them score the same.
  - The 84.5% coverage belongs to ridge with `n_lags=0`; with the default `n_lags=1` the same
    design gives 52.8%. "Exact constancy" was written where the check is numerical. The
    message counted "the other" rows without the lag rows. `pd.NA` in a list is refused, not
    read as NaN.
- **Tests that did not bite.** 18 of the reviewer's 59 mutations survived. Tests were added
  for 14 of them: text other than `str` and text after the first element, unsigned integers, a
  ragged input, the conversion of a nullable column with a missing value, indexes of dates,
  strings and repeated labels, a treatment against controls when the outcome has no labels,
  the check with three lags and with a gap, the row counts of the message, and the edge cases
  of the constancy helper. Not tested: a non-finite level without its guard gives the same
  verdict (comparisons with NaN are false), `<` in place of `<=` differs only when the spread
  equals the bound exactly, and two message mutations that the reviewer called benign.
- **Not repaired, by scope.** `TemporalPLRDML`, `RollingWindowDML` and `PanelDML` have neither
  guard (PR-1d, and the redesign for `PanelDML`). An index that was reset after reordering
  cannot be detected. Masked controls are read under the mask by every entry point and numeric
  text in controls is parsed by scikit-learn (PR-1d). `RollingWindowDML` returns an empty
  result with a warning when every window is refused (#42).
- **Gemini's two false claims:** that `None` in a list is refused (it is read as NaN), and that
  an integer index beside the same labels as floats is refused (`Index.equals` is true).
- **Lesson.** Whether a container can be cast is not whether it holds numbers. A reader that
  falls back on a cast must first look at what it is casting; the matrix of input kinds needs
  the kinds that libraries cast silently (timezone-aware dates, object arrays of scalars).

## Addendum 5, 2026-11-26 (evening): decisions at the fourth stop

The full text of the addendum (decisions, order, rules) is in the session's plan file, of which
a copy is in the landing's local backup folder (`plans/`). Its decisions are recorded here with
what was done. The record of the fourth stretch itself is on #82's branch (`88407d8`) and lands
with it.

### Decisions taken at the fourth stop (the owner's)

| Decision | Choice |
|---|---|
| Merging | One more independent review of the final heads of #79 and #82; they merge if it finds no regression for a valid input and no wrong returned number. #83 and #87 merge once the decisions are applied and CI is green |
| Licence holds | The three ACIC files and the six fixture rows land; `gdp-survey.csv` and the two receipts that embed it stay held |
| Controls of `robinson_estimator` and `double_ml` | With one of the package's own learners, datetime, timedelta, complex and text controls are refused by name; numeric controls reach the learner unchanged; a learner the caller supplies gets the controls untouched |
| Receipts with output of held `PanelDML` scripts | Kept as landed |
| Defaults | #87's README loses the first paragraph of its handoff note; in `double_ml` the check applies whenever any built-in learner sees X; #64 is one small pull request that stops for a go-ahead; the exit record and the Part II design note go in one documents-only pull request |

### Record of step 1: the controls decision (#82)

- `b8dae4c` adds `refuse_non_numeric_controls` (`dml_ts/dml/_utils.py`) and calls it in
  `robinson_estimator` and `double_ml` when a built-in learner receives X. It only checks: the
  learner is handed the caller's object.
- `88407d8` adds the test that a mutation showed missing (masked controls with a supplied
  learner) and drops a default that cannot occur.
- Before and after, against `85374d7`: no number changed in the 968 cells of
  `before_after_matrix.py`. In the 731 cells of `before_after_learners.py`, 48 went from a
  number to a refusal of exactly the decided kind, no other number changed, and none of the
  215 cells with a supplied learner changed.
- Gate on `88407d8`: 2,051 collected, tier 1 + 2 1,866 passed, coverage 83.59%; 63 of 63
  mutations caught.

### Record of step 3: #83 and #87

- #83: the four files restored from the snapshot (`02f1379`), with `DATA_SOURCES.md`, the
  dataset ledger, the status document and the changelog. `verify.py` matched 3,200 keys
  (largest difference 6.93e-14); a fresh `run.py` reproduced `draws.csv` and `by-design.csv`
  byte for byte. Gate and CI green; merged as `4e61710`.
- #87: the README's handoff paragraph removed (`a3d5bc0`), #83's branch merged in (`8fb7ac6`).
  Closed and reopened so that CI ran against `main`; merged as `2415b95`.

### The interruption

At 20:17 EST on 2026-11-26 the machine was powered off after a global out-of-memory event,
while step 2 (the review of the final heads) and the PR-6 mutation run were in progress. 71 Python processes were running, with 85 GiB resident in all (about 1.2 GiB each, by the kernel's out-of-memory report); another session's log attributes them,
by inference, to this repository's review agents, and the review's own driver started 20
workers per state for three or four states. The stop report of this addendum was never
written; Addendum 6 completed its steps, apart from the merge of #82, which is held.

## Addendum 6, 2026-11-27: review of the work, the guard, and the exit from Part I

The full text is in the session's plan file (copy in the local backup folder, `plans/`).

### Findings of the review

1. The interruption left three things at risk: PR-6 on one disk, Addendum 5's record and the
   design note outside the repository, and the merge rule of #79 and #82 unmet while half of
   their review was unjudged.
2. The workflow's parallelism can take the machine down: the design note's cost line assumed 64 workers, about 77 GiB of the machine's 123 GiB at the 1.2 GiB per process measured on 2026-11-26, on a machine that other work shares.
3. The input-contract work has reached diminishing returns: each review round of #79 found a
   regression in the previous repair. After #82 the contract is frozen; new edge cases become
   comments on #77.
4. The status document contradicted itself (a June half with its own baseline and roadmap
   under the November section), and `CLAUDE.md`'s baseline was from 2026-05-02.
5. Tier 3 and tier 4 had not run in CI for 39 days.
6. Readers see known errors: Chapter 6 (#84), Chapter 8's numbers, #65.
7. The design note's method is sound; its cost line needs a memory budget.
8. Two older plans were stale (content expansion, the Track 3 handoff).
9. #46's overflow item can be closed by a scaffold bump to `^4.26.2` (the fix, scaffold #171, was published in 4.26.2; no 4.26.1 was released). Its other items stay open.

### Decisions taken on 2026-11-27 (the owner's)

| Decision | Choice |
|---|---|
| Part II design gate | The note's bundle: B and C, with A; the nuisance treatment a required argument; `forward_date` out of the first release; `fit_frame` after the estimator passes; `panel_standard_error` early, in its own pull request; the protocol as drafted, with the cost recomputed under a memory cap |
| Chapter 10's monitor listings | Updated in both editions inside PR-6 |
| Manuscript | A parallel track during Part II; known-issue callouts in Chapters 6 and 8 now |
| The working copy | Retired at the exit from Part I: verified and moved aside, not deleted before the close of the landing; a fresh clone takes its path |

### The guard (applies to every run)

- A run that starts more than one Python process goes under
  `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0`.
- At most 16 concurrent Python processes per job and one heavy job of this repository at a
  time, started only with at least 60 GiB available.
- Review agents: at most 4 workers per state and three states; mutation runs at most 8 jobs.
- A new virtual environment needs at least 20 GiB free on the disk.

### Record

- **Secured.** PR-6 pushed to the private mirror as a backup branch; both session plans copied
  to the local backup folder; the second-disk copy refreshed.
- **Final-heads review of #79 and #82.** A fresh agent judged the unjudged differences and
  finding scripts; each finding was re-run. One regression in #82: a one-field structured float
  array as Y or T (`dtype=[("a", "f8")]`), which `main` and #79 read (2.109180), is refused by
  `TemporalPLRDML`, `RollingWindowDML` and `DynamicGEstimationDML`. #82's changelog says that two kinds of call returned a number before and are refused now; object arrays with datetime or timedelta columns beside numeric ones were a third. No new wrong number was found: the two the judge found are wrong on `main` as well (`PanelDML` with identifiers that are not integers, #77; the R² at extreme scales, which #79 turns from `main`'s 0.0 into NaN, #90). #79 was closed and reopened for CI against `main`, merged
  as `c84ce72`; `stored_checks.py` passes on `main`, and #75 and #76 are closed. #82 is held for
  the owner: (a) a documentation commit that lists the refusal, then merge, or (b) a code repair
  and a new review. Pre-existing on `main` and filed: the R² at extreme scales (#90).
- **PR-6 (#89).** Chapter 10's listings rewritten in both editions and checked against the
  package (0 mismatches in 1,721 cases). A fresh-agent review of `56b73cb` found eleven points;
  each was re-run and answered (the pull request's description lists them). Gate on `41f5bc9`
  green (1,559 passed in tier 1 + 2, coverage 86.03%), 48 of 48 mutations caught, CI green.
  Waits for a go-ahead.
- **Conservation check.** The 234 paths in which the snapshot differs from `main` (`c84ce72`)
  were classified one by one (a fresh agent, then re-checked): none is unlisted. Held whole
  files equal the pre-landing `main`; held new files are absent from `main`; the held parts of
  split files equal the pre-landing `main` apart from recorded deviations.
- **Part III residual.** 179 patches in the local backup folder (`part3-residual/`): 143 for new files, 12 for whole files, 20 for the held hunks of 8 split files (one per file and one per hunk), and 4 for the two cricket tests held for #86 (likewise). Each passes `git apply --3way --check` against `c84ce72`. Applied, the new and whole patches reproduce the snapshot's 155 files, and the hunk patches give `main` with the held hunks. One hunk differs from the snapshot's side: it keeps
  `is_numerically_constant` for #79's refusal. The snapshot's `PanelDML` does not carry
  `main`'s absorbed-treatment guard (#69): Part II must carry that case.
- **Nightly.** Dispatched once (run 46024713734): tier 3 passed in 43 minutes; tier 4 runs only
  on the Sunday schedule. The schedule has not fired since 2026-10-19: #88.
- **The working copy.** Its tree equals the snapshot's (`c560e48`) and its status hash is
  unchanged; its ignored folders are archived on the second disk, and the licence-held files
  and local settings are copied aside. The move was not made: an editor had the folder open.
  A fresh clone is staged and takes the path when the move is made.
- **Protocol.** `docs/plans/active/2026-11-27_paneldml_protocol.md`, SHA-256
  `bd5bf7407ef92cac706569eff9fd1cdaf54411d36c3c9b7fb32b6f939ad00bd2`, frozen in this pull
  request before any study run.
