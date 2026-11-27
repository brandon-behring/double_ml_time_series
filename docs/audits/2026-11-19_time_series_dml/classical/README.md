# Classical econometric benchmarks

The randomized earnings contrast and the ten-firm investment slopes reproduce at published precision. A direct, same-fold software comparison also reproduces the earnings DML coefficient and its standard error after aligning variance conventions. For the historical fiscal application, using the authors’ automatic bandwidth resolves the earlier discrepancy with the **published** standard errors. The separate output workbook still gives different values. [Measured results and fold assignments](results.json) · [Empirical replay](replay.py)

## Randomized job training

Dehejia and Wahba’s National Supported Work RE74 sample contains 185 treated and 260 randomized controls. Mean 1978 earnings were $6,349.14 and $4,554.80 in 1982 dollars. The treatment contrast is **$1,794.34**. Two legitimate SEs answer different variance assumptions. The repo’s ordinary least-squares calculation gives **$632.85** using pooled residual variance and reproduces the paper’s $1,794 ($633). The arm-specific difference-in-means formula gives **$671.00**, matching a treatment regression with HC2 inference. The source release also offers different LaLonde samples and nonexperimental comparison groups. The replay checks the treatment code and size of each experimental arm before fitting. [Dehejia and Wahba, Table 3](https://users.nber.org/~rdehejia/papers/dehejia_wahba_jasa.pdf) · [Source files, attribution and terms](https://users.nber.org/~rdehejia/nswdata2.html)

We fit ridge regressions for the outcome and treatment using all eight supplied pretreatment covariates, with scaling learned inside each of five shuffled folds. Both the repo and DoubleML 0.10.1 estimated **$1,750.78**. The largest held-out prediction differences were $2.8 × 10⁻¹² for earnings and 5.6 × 10⁻¹⁷ for treatment, rounded up. DoubleML reports an SE of $667.57 using the empirical second moment. The repo reports $668.32 using the sample variance. Multiplying the reference SE by √(445/444) produces agreement to machine precision. On this split, the native adjusted SE is **0.40% smaller** than the unadjusted arm-specific SE. One randomized cross-section with one partition provides little evidence for a general precision advantage. [Reference splitting interface](https://docs.doubleml.org/stable/guide/resampling.html) · [Exact folds and measured deviations](results.json)

## Grunfeld investment

We used Baltagi’s ten firms from the original eleven-firm release, dropping American Steel, for 1935–1954. Each row reports the investment slopes and classical homoskedastic SEs in parentheses. Only the first row has a published target here. The remaining rows check the implementation against an independent regression using explicit fixed-effect dummies. [Kleiber and Zeileis, data and versions](https://www.zeileis.org/grunfeld/) · [Baltagi example and R output](https://www.zeileis.org/grunfeld/Baltagi-2005.Rout)

| Fixed effects and sample | Firm-years | Firm value | Capital |
|---|---:|---:|---:|
| Firm, complete | 200 | 0.110124 (0.011857) | 0.310065 (0.017355) |
| Firm and year, complete | 200 | 0.117716 (0.013751) | 0.357916 (0.022719) |
| Firm, unbalanced | 171 | 0.127972 (0.012817) | 0.310746 (0.018647) |
| Firm and year, unbalanced | 171 | 0.133864 (0.014641) | 0.360590 (0.024060) |

The deletion rule depends only on firm identifier and year. Across the four samples/specifications, the repo’s within transformation and independent dummy regression differed by at most 1.7 × 10⁻¹⁴ in slope. Permuting the rows changed slopes by at most 2.3 × 10⁻¹⁶. The public Frisch–Waugh–Lovell regression also returned the dummy regression’s coefficients and classical SEs. These checks validate fixed-effect elimination, including the repeated projections needed in an unbalanced two-way panel. An investment effect would require an argument for the exogeneity of capital or firm value. [Replay and deletion rule](replay.py)

## Known impulse responses

In the simulated system, a one-unit randomized shock changes the current outcome by one and the effect then decays by a factor of 0.6 per date. OLS local projections and the repo’s temporal ridge PLR used the same scoring dates, the prior outcome as a control and the same outcome horizon. Three expanding folds use a horizon-length gap so every training outcome has matured before the scored block. Each cell summarizes 500 independent paths. [Simulation and interval construction](impulse.py) · [Jordà’s local projections](https://www.aeaweb.org/articles?id=10.1257%2F0002828053828518)

| Dates | Horizon | True response | Mean OLS / PLR | 95% interval coverage, OLS / PLR |
|---:|---:|---:|---:|---:|
| 300 | 0 | 1.0000 | 0.9983 / 1.0007 | 94.2% / 93.6% |
| 300 | 1 | 0.6000 | 0.5979 / 0.6219 | 92.6% / 93.0% |
| 300 | 4 | 0.1296 | 0.1206 / 0.1619 | 91.6% / 91.0% |
| 1,200 | 0 | 1.0000 | 1.0009 / 1.0006 | 94.6% / 94.0% |
| 1,200 | 1 | 0.6000 | 0.5991 / 0.6040 | 95.2% / 94.6% |
| 1,200 | 4 | 0.1296 | 0.1285 / 0.1371 | 94.8% / 94.6% |

At horizon four, temporal PLR’s bias fell from +0.0323 (Monte Carlo SE 0.0059) to +0.0075 (0.0027). OLS had smaller RMSE in both samples at this horizon. The finite-sample precision and coverage differences matter even in this correctly specified linear design. Two sample sizes allow an empirical check of how error changes with information. [Summary and Monte Carlo SEs](impulse-summary.csv) · [Individual draws](impulse-draws.csv.gz) · [Settings and serial replay](impulse-run.json)

## Historical fiscal multipliers

Published Table 1 estimates a cumulative output-to-spending ratio, using military news as an instrument for cumulative government spending. The denominator, normalization by potential GDP, and four lags of news, spending and output all enter the replay. IV identification still requires an appropriate first stage and exclusion restriction. [Published Table 1](https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_JPE_published.pdf) · [Author archive, data, Stata script and results workbook](https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_replication_codes.zip)

| Horizon and effective sample | Raw-data coefficient (SE) | Published coefficient (SE) | Output workbook coefficient (SE) |
|---|---:|---:|---:|
| 8 quarters, 1891Q1–2014Q1, 493 rows | 0.663714 (0.067114) | 0.66 (0.067) | 0.661911 (0.068221) |
| 16 quarters, 1891Q1–2012Q1, 485 rows | 0.713361 (0.043575) | 0.71 (0.044) | 0.711940 (0.045476) |

The authors’ Stata command specifies robust inference with automatic bandwidth and omits the small-sample correction. We translated the Bartlett selector from the published `ivreg2` source. It uses the structural residual multiplied by the sum of the nonconstant instruments to choose the bandwidth. Both regressions used 28 nonzero lags, corresponding to bandwidth 29. We then formed the sandwich for the IV slope from the residualized news instrument times the structural residual. Using each projection horizon as the lag count instead gave SEs 0.070220 and 0.042908. Adding the parameter-count correction at the automatic bandwidth gave 0.068088 and 0.044218. Summing nine quarters in the two-year specification changed the point estimate to 0.668961. The underlying Stata binary was unavailable. A separate matrix-IV regression and HAC covariance implementation check the sandwich on seeded data. [Baum, Schaffer and Stillman’s estimator and source](https://ideas.repec.org/c/boc/bocode/s425401.html) · [Independent sandwich test](../../../../test/validation/test_classical_benchmarks.py)

Both the original Excel inputs and the alternative CSV supplied by the authors reproduce the *paper’s* two slope and two SE cells to rounding. Their coefficients agree across input formats within 2 × 10⁻¹³. The archived results sheet is headed 1890Q2–2015Q4. Its effective observations are unavailable, and the inputs’ missing early news and four required lags place the first usable origin in 1891Q1. The provenance of the workbook’s full-precision values remains unresolved. [Raw fits and workbook values](results.json)

## Reproduction

From the repository root, run the commands below. The empirical entry point verifies the hashes of all inputs and runs the pinned DoubleML and Excel reader in a `uv` environment layered over the locked project. The replay fetches restricted-use inputs directly from the authors into the local cache. The checked-in output contains aggregate statistics and fold indices.

```bash
audit=docs/audits/2026-11-19_time_series_dml/classical
uv run --locked python "$audit/replay.py" --cache-dir /tmp/dml-classical-data
uv run --locked python "$audit/impulse.py" --workers 8 --repetitions 500 --output /tmp/dml-impulses
uv run --locked python -m pytest -q test/validation/test_classical_benchmarks.py
```
