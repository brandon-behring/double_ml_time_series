# What was audited

The starting checkout contained 307 tracked paths on `fix/validation-fail-loud-41a` at `0cca908463719d46dbe43a44c17a6d2dd81ceb89`. Local `main` and the local `origin/main` ref both identified `fc1c7dec993b65e5c93dc995af5e264ac850480f`, with 306 paths. The checkout was one commit ahead of that baseline. The audit uses these immutable references. SHA-256 of the initial `git diff --binary HEAD` bytes was `fc51d08e6a4d0ecfbc0087c1ca7cc109d2cc6f649761f12fc7a4c6c0fcc11bb2`.

## Coverage at the start of the audit

| Area | Paths | Work represented |
|---|---:|---|
| Repository root | 17 | Book entry point, active bibliography, package metadata and lock, build rules, style, license, contribution guidance, assumptions and changelog. |
| [Package](../../../dml_ts/) | 41 | Package root, 11 estimator/inference files, 19 validation files, five research pipeline files, three data files and two sensitivity files. |
| [Tests](../../../test/) | 57 | 41 test modules plus fixtures, import markers, instructions and golden reference data. The 9,915-row 401(k) fixture enables offline comparison. |
| [Scripts](../../../scripts/) and [templates](../../../templates/) | 18 + 3 | Simulation and diagnostic drivers, validation and empirical replication, document conversion, build and provenance checks, and scaffolds. The Chapter 6 table has a named runner. |
| [Examples](../../../examples/) and [notebooks](../../../notebooks/) | 8 + 2 | Eight script entry points and two historical validation notebooks. See [teaching material review](03-manuscript-and-content.md) for notebook import drift. |
| [Chapters](../../../chapters/) and [shared style](../../../shared/) | 16 + 1 | Ten numbered TeX chapters, one Julia appendix, secondary bibliography, a template and three historical AsciiDoc files. |
| [Web edition](../../../web/) | 23 | Ten MDX ports, academic profile/configuration, lockfile, assets, upstream issues and operating instructions. |
| [Documentation](../../) | 101 | 36 historical archive files, 33 research files, 15 Sphinx files, 12 plans, three prior audit/index files, the status page and an external patterns-directory symlink. The symlink identifies a dependency in another checkout. |
| [Frozen results](../../../results/) | 11 | Three impact CSVs, one Lasso diagnostic, five method-comparison artifacts and two bias-simulation artifacts. |
| CI and local agent instructions | 5 + 4 | Five workflows cover package checks, book, web, documentation and nightly runs. Three local command descriptions and one local settings file support authoring. |
| **Total** | **307** | All tracked paths are accounted for before this audit was added. |

## Which version contains which work

- The November 2025 work established the validation harness and recorded the early simulation results. The January 2026 commit `75750ee` introduced Chapter 5's Monte Carlo table. Neither the table nor the displayed generator identifies the original seeds or a saved draw-level result. [Chapter 5 reconstruction](04-reproducibility.md)
- The May audits documented major changes in claims and project organization. Subsequent commits added panel-mode dynamic g-estimation (`9ef7880`), fixed the extra HAC scaling (`36adac1`) and panel SE scaling (`be72f23`), migrated core primitives to temporalcv, completed the web chapters and added the doctest gate. The June status table remains a historical measurement. [Previous audit](../2026-05-30_repo_quality_methodology_audit.md) · [Changelog](../../../CHANGELOG.md)
- Committed `main` includes the v1.1.4 release, fail-loud core estimator and monitor changes, and exact-value SE tests. The feature commit adds validation failure handling and regression cases across bias testing, bootstrap diagnostics, time-series DGPs and insurance recovery. Its only additional tracked path is `test/validation/test_validation_fail_loud.py`.
- The initial working diff added 86 lines and removed 10 across six files. Four validation modules refine nonfinite bias SE handling, degenerate bootstrap draws, near-boundary innovation warnings, and failed/empty insurance recovery runs. The associated test file gains 42 lines. One unrelated local settings deletion is excluded from the substantive findings. The two versions' test counts appear in [reproducibility](04-reproducibility.md).
- Unmerged local refs include web-scaffold updates, a code-width preview, and `origin/draft/dml-confounding-lens`. The latter adds an interactive-demo strategy in its merge-base comparison. These are branch context, outside the selected checkout/main comparison. The local `gh-pages` ref contains generated publication work.

## What the saved results establish

The November 14, 2025 method comparison covered six configurations, seven estimators and 20 replications per method and configuration. Its 42 detailed rows support comparisons within that design. Even at true 95% coverage the Monte Carlo standard error at 20 replications is about 4.9 percentage points. Selecting the best of several methods adds selection variability. The other bias-validation run describes 100 replications at each of 81 parameter combinations. Neither run matches the 200-replication Chapter 5 design. The three November 21 impact tables document earlier binary-treatment, simulation-comparability and covariate investigations. [Method-comparison metadata](../../../results/method_comparison/findings.json) · [Bias-validation metadata](../../../results/simulations/BIAS_VALIDATION_metadata_20251114_103701.json) · [Impact tables](../../../results/impact_analysis/)

The working directory also contained a generated book PDF, two earlier chapter PDFs, Sphinx and TeX build products, coverage and pytest output, and ignored historical logs. One local 401(k) log records a random-forest estimate of $7,946.16 and marks it a match against $9,127 under a 15% tolerance. That label illustrates why the published table and actual specification require an independent check. The local outputs are absent from a clean clone. [401(k) replication](07-401k-replication.md)

For an exact path inventory, run `git ls-tree -r --name-only 0cca908463719d46dbe43a44c17a6d2dd81ceb89`. To isolate the committed validation work, run `git diff --stat fc1c7de..0cca908`. To inspect the six pre-existing local edits independently of the audit, run `git diff 0cca908 -- .claude dml_ts test`.
