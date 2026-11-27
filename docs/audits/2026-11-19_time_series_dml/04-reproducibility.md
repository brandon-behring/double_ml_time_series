# Reproducibility of the book, code and quantitative claims

The ten web chapters, the 209-page book, the Sphinx site and all eight Python examples built or ran in the November 19 audit. Chapter 6's published panel table reproduced to displayed precision. On November 20 both editions of Chapter 5 replaced their interval-coverage table with results from the two seeded experiments below.

## Chapter 3 — power to detect simulation bias in the November 19 snapshot

The November 20 revision reports the power of the corrected one-sample test in both editions. [Seeded results and historical verdicts](08-bias-test-correction.md)

I ran the literal Python block for Experiment 3.6 with NumPy's legacy RNG seeded at 20261119. At 100 independent estimates per trial, 998 of the 1,000 trials detected the injected mean of 0.05 at the 5% level. The book reports 17%. At 250, 500, 1,000 and 2,000 estimates, all 1,000 trials detected the bias; the table gives 34%, 56%, 80% and 95%. The code specifies a per-estimate SD of 0.10, so the signal at 100 draws is five standard errors. The exact two-sided power using the code's population-SD convention is 0.998655 at 100 draws. Replace the table and its minimum-simulation-count advice, or change the design and regenerate it. This is a second, independent issue from the package's bootstrap bias p-value. [Experiment 3.6](../../../chapters/chapter_03.tex) · [Bias test](02-code-and-methodology.md)

## Chapter 5 — matched-score reconstruction

The table introduced in commit `75750ee` reported 78% and 93% empirical coverage for nominal 95% intervals. Its original run configuration and draws could not be recovered. The replacement uses the chapter's published data generator and a second design with known score persistence. Each design generates 200 independent series of 500 observations with true effect 2. Coverage counts how often a two-sided normal interval contains 2. [Print chapter](../../../chapters/chapter_05.tex) · [Web chapter](../../../web/src/content/chapters/chapter05-temporal-plr-dml.mdx) · [Simulation driver](chapter5_check.py) · [Results by replication](chapter5-draws.csv)

| Data and estimator | Scored rows | Mean bias | Empirical SD | Mean SE | 95% coverage | MC standard error of coverage |
|---|---:|---:|---:|---:|---:|---:|
| Chapter data, shuffled fivefold forest PLR, i.i.d. SE | 500 | −0.0292 | 0.0567 | 0.0507 | 175/200 (87.5%) | 2.34 pp |
| Chapter data, forward forest PLR, zero-lag score SE | 415 | −0.0457 | 0.0714 | 0.0694 | 179/200 (89.5%) | 2.17 pp |
| Same forward fit and scores, automatic Bartlett HAC | 415 | −0.0457 | 0.0714 | 0.0689 | 177/200 (88.5%) | 2.26 pp |
| Persistent scores, forward ridge PLR, zero-lag score SE | 415 | +0.0008 | 0.1076 | 0.0484 | 121/200 (60.5%) | 3.46 pp |
| Same persistent-score fit, automatic Bartlett HAC | 415 | +0.0008 | 0.1076 | 0.0872 | 178/200 (89.0%) | 2.21 pp |

The first three rows use the chapter's treatment AR coefficient of 0.3 and error AR coefficient of 0.2, with seeds 0–199. The shuffled fit uses the three raw controls. The forward forest adds one treatment lag and uses five expanding splits, gap zero and 100-tree forests with learner seed 42. Changing from shuffled to forward also changes the controls and the scored sample. Within each forward pair, the fitted coefficient and the 415 out-of-fold scores are identical on every draw. Only the bandwidth changes, from zero to the automatic choice of seven. The table gives the Monte Carlo SE of each coverage rate in percentage points (pp).

**Why serial dependence matters.** The influence score multiplies the residualized treatment by the fitted structural error and divides by the mean squared residualized treatment. Its long-run variance adds twice the sum of the positive-lag score autocovariances to its one-period variance. The HAC standard error weights the estimated autocovariances and takes the square root of that estimated variance divided by the scored sample size. A positive weighted sum widens the interval and a negative one narrows it. [Newey and West (1987), equation 5](https://users.ssc.wisc.edu/~kdwest/publications/1980/A%20Simple%20PSD%20HAC%20Covariance%20Matrix.pdf)

In the chapter process, conditioning on the first treatment lag and current controls leaves an independent treatment innovation. With the true nuisance functions the score is proportional to that innovation times the outcome error. Every positive-lag score autocovariance is zero even though the outcome error is persistent. The fitted scores' mean lag-one correlation was **0.008**. The forward fit's mean bias was −0.0457 under both variance estimates. Changing the standard error can change the width of an interval while leaving its center equally biased.

In the second design, two independent stationary, unit-variance AR(1) processes with coefficient 0.8 supply the residual treatment and the structural outcome error. Their product has population score correlation $0.64^k$ at lag $k$. Forward ridge regressions use two exogenous controls and zero treatment lags. The fitted scores' lag-one correlation averaged **0.621**. The population long-run variance is 4.56 times the one-period variance. Applying the seven-lag Bartlett weights to the exact population autocovariances gives a smaller multiplier of 3.36. This illustrates why a finite bandwidth can still leave the mean SE (0.0872) below the empirical SD (0.1076).

**How many intervals changed?** On the chapter series, one interval gained coverage under HAC and three lost it, a change of **−1.0 pp** with paired Monte Carlo SE **1.0 pp**. On the persistent-score series, 57 gained coverage and zero lost it, a change of **+28.5 pp** with paired Monte Carlo SE **3.2 pp**. To reproduce each paired SE, subtract the lag-zero coverage indicator from the HAC indicator within each replication and divide the sample standard deviation of those 200 differences by the square root of 200. Multiply by 100 for percentage points. [Paired observations](chapter5-draws.csv)

The November 20 rerun reproduced the November 19 summary for all five rows. The CSV records the replication index, estimate, SE, number of scored observations, score autocorrelation, bandwidth and coverage indicator. Replication index `r` uses RNG seed `r` in the chapter process and `10000 + r` in the persistent process. A blank score autocorrelation denotes the shuffled fit. To repeat the comparison, run the driver from the repository root in the verification environment below. The workers split the independent replications, with one thread per fit.

```sh
audit=docs/audits/2026-11-19_time_series_dml
draws=$(mktemp)
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DML_N_JOBS=1 \
  python "$audit/chapter5_check.py" --workers 16 --repetitions 200 --draws "$draws"
cmp "$draws" "$audit/chapter5-draws.csv"
```

The rebuilt print table (Table 5.1) and web table each matched all 30 numeric cells recomputed from the archived draws. The two editions passed the source-drift check. All 73 tests in the temporal PLR, inference and chapter-provenance test files passed.

## Chapter 6 — original simulation driver at the November 19 snapshot

The original table below documents the positional folds. The November 20 chapter reruns the same 100 panels using whole-unit folds and archives all 300 intervals. [Revised results and commands](09-panel-retail-insurance.md#panel-design-and-the-ordering-check)

`PYTHONPATH=. DML_N_JOBS=1 python scripts/mc_panel_dml_table.py` ran 100 simulations with 40 units, 15 periods and true effect 2. Each line matches Chapter 6 after rounding. Pooled DML returned bias **+0.481**, mean SE **0.040**, coverage **0%**. Individual and two-way fixed effects both returned bias **+0.009**, mean SE **0.046**, coverage **90%**. At 90 successes in 100 trials the MC standard error of coverage is 3.0 percentage points. Generalization to ragged panels or calendar holdouts needs additional evidence. [Original runner](../../../scripts/mc_panel_dml_table.py) · [Table](../../../chapters/chapter_06.tex) · [Panel counterexamples](02-code-and-methodology.md)

## Verification environment and gates

The initial tracked working files and a separate `git archive main` were copied to temporary directories. Both imports were run from their own checkout with `PYTHONPATH=.`. The shared environment used CPython 3.13.17, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.6.1, temporalcv 2.2.0, EconML 0.16.0 and Sphinx 9.1.0. The package built from the clean snapshot reports 1.1.4. A frozen sync installed the dependencies, but a fresh online `uv lock --check` fails because the lock needs updating; its local-project record still declares 1.1.2. Refreshing a disposable copy with the installed uv changed the root package version and lockfile representation, while the dependency names and versions stayed the same. The original working directory also has an ignored older egg-info directory. Inspect package metadata in a clean installation when comparing releases.

| Gate | Fresh result |
|---|---|
| Collection | **880** on the dirty checkout; **868** on committed main. |
| Tier 1 | **374 passed**, 506 deselected on the checkout. |
| Tiers 1 and 2 | Checkout **695 passed**, 185 deselected, coverage **75.48%**; main **683 passed**, 185 deselected. Warnings were present in both runs. |
| Full checkout suite | **880 of 880 distinct test cases passed** across the tier 1/2 run, 183 higher-tier non-network cases and 20 further network-marked cases. Eighteen cases occurred in both the first two groups. The union of their pytest node IDs matched the complete collection. The Lasso diagnostic alone emitted 220 convergence warnings. |
| Formula contracts and insurance unit/integration tests on main | **40 passed**, four deselected with the tier filter. |
| Package doctests | **50 passed**, 19 skipped. |
| Ruff, format, mypy, port provenance | Lint passed, 99 files formatted, mypy passed on 41 modules, all ported chapter hashes matched. |
| Eight examples and strict Sphinx build | All examples exited successfully. `sphinx -b html -W --keep-going` succeeded. |
| Book | Clean `make -B all` produced **209 pages**, with **266 overfull** and **12 underfull** box warnings. The first attempt failed to write the Lua font cache. Setting a writable `TEXMFCACHE` and rebuilding resolved it. |
| Web | npm ci from the lockfile, Node 22.20.0, npm 11.7.0. Build lifecycle validated **10 chapters**, Pagefind indexed **15 pages**. |

For checkout reconstruction and tests, use the recorded versions in [the inventory](01-inventory-and-history.md). From the copied checkout:

```sh
uv sync --frozen --python 3.13 --extra dev --extra docs --extra full --extra fred
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD" DML_N_JOBS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest --collect-only -q
python -m pytest -m tier1 -q
python -m pytest -m 'tier1 or tier2' --cov=dml_ts -q
python -m pytest -m '(tier3 or tier4) and not network' -q
python -m pytest -m 'network and not (tier1 or tier2)' -q
python -m pytest --doctest-modules dml_ts/ -q
ruff check dml_ts test examples && ruff format --check dml_ts test examples
mypy dml_ts --ignore-missing-imports --no-strict-optional --explicit-package-bases
python scripts/check_tex_mdx_drift.py
for f in examples/*.py; do python "$f" || exit 1; done
sphinx-build -b html -W --keep-going docs/sphinx /tmp/dml-audit-sphinx
mkdir -p /tmp/dml-audit-texcache
TEXMFCACHE=/tmp/dml-audit-texcache make -B all
(cd web && npm ci && npm run build)
```

**Incremental build defect.** The Makefile's PDF dependency list names chapters 1–2 and the secondary chapter bibliography. A dry run treating Chapter 10 or the active root bibliography as newer produced “Nothing to be done” with an existing PDF. The clean build above validates the current sources; the incremental dependency list can leave an author reading a stale PDF. The book workflow's path triggers also omit the root bibliography. [Build rules](../../../Makefile) · [Book workflow](../../../.github/workflows/book.yml)
