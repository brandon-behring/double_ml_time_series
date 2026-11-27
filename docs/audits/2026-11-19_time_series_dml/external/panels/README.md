# Published benchmarks for state panels and daily time series

Two small author-hosted archives let us check fixed effects, weighting and uncertainty against published economic research. The checks compare estimates and standard errors independently, allowing half a unit in the last printed decimal. The scripts also compare the repository’s Frisch–Waugh–Lovell (FWL) estimator with an independent implementation wherever the paper fits a compatible linear specification. A match establishes that we can reproduce that calculation on the authors’ sample. Predictive performance, causal identification and interval coverage require separate evidence.

## Wolfers (2006), unilateral divorce reform

Table 2 estimates annual divorces per 1,000 people at eight durations since a state introduced unilateral divorce. The sample contains 1,631 observed state-years across 51 jurisdictions from 1956 through 1988. Each regression includes state and year indicators, uses state population weights and compares each duration to the authors’ reference category (`years_unilateral == -99`). The published standard errors come from the model-based weighted regression. [Published Table 2, p. 1808](https://users.nber.org/~jwolfers/Papers/Divorce%28AER%29.pdf) · [Data appendix](https://users.nber.org/~jwolfers/data/DivorceDataAppendix.pdf) · [Author’s data and Stata programs](https://users.nber.org/~jwolfers/data.php)

| State trends | First two years, estimate (SE) | Year 15 onward, estimate (SE) | Adjusted R² |
|---|---:|---:|---:|
| None | 0.266673 (0.084886) | −0.507382 (0.080471) | 0.930967 |
| Linear | 0.342386 (0.062331) | −0.210312 (0.119239) | 0.973163 |
| Linear and quadratic | 0.301931 (0.053788) | 0.251014 (0.204976) | 0.982220 |

The test checks all 24 coefficients, 24 standard errors and three adjusted R² values against Table 2. At the first and last durations in each specification it also multiplies the outcome and **every** design column (including the intercept) by the square root of state population, then runs `fwl_estimate` and requires agreement with statsmodels WLS to `1e-8`. The observed sign change after 15 years makes an omitted quadratic interaction easy to catch. A forward-fold DML effect or a state-clustered standard error would answer a separate question.

The source program is `Divorce rate analysis.do` in the author’s Stata 9 ZIP. The archive SHA-256 is `7fc6fdce65688e6d522e224a547e62c321ce6bdfe77adbf64f7df344653b4bd4`. Its `Divorce-Wolfers-AER.dta` member has SHA-256 `c43fd642171407ce7021d89e40dc64c7a14b24c5d2ff55de0a950d002c842d80`. [Download the author’s archive](https://users.nber.org/~jwolfers/data/DivorceDataAppendix%28Stata9%29.zip)

## Dahl and DellaVigna (2009), movie attendance and same-day assaults

The authors link national movie attendance in three ratings categories to the log of reported assaults in participating agencies on Fridays, Saturdays and Sundays in 1995–2004. The coefficient unit is log points per additional million viewers in the indicated category, conditional on the other audiences and specified controls. Each weekend forms one uncertainty cluster. The following figures come from the pinned daily data and six-hour blocks. [Published Tables II and III, pp. 697 and 699–700](https://eml.berkeley.edu/~webfac/dellavigna/e101a_f09/moviescrimeQJEProofs2009.pdf) · [Author’s replication page](https://econweb.ucsd.edu/~gdahl/movies-code.html) · [Author’s Stata program](https://econweb.ucsd.edu/~gdahl/movies-code/main.do)

| Strongly violent audience specification | Days | Weekends | Estimate | Weekend-clustered SE |
|---|---:|---:|---:|---:|
| Daily OLS, year indicators, Table II (1) | 1,563 | 522 | 0.032404 | 0.005347 |
| Daily OLS, calendar, holiday, weather and TV controls, Table II (6) | 1,563 | 522 | −0.009103 | 0.002577 |
| Daily IV, same controls, Table II (7) | 1,563 | 522 | −0.010625 | 0.003095 |
| Noon–6 p.m. IV, Table III A (2) | 1,563 | 522 | −0.003023 | 0.004975 |
| 6 p.m.–midnight IV, Table III A (3) | 1,563 | 522 | −0.013039 | 0.004893 |
| Following midnight–6 a.m. IV, Table III A (4) | 1,562 | 521 | −0.019155 | 0.006018 |

The daily OLS comparison runs the native FWL estimator twice. Adding seasonality and the other controls reverses the sign of the strong-audience slope. For inference the script sums score contributions within each weekend. With `G` weekends, `n` observations and `k` linearly independent coefficients including the three exposures, the finite-sample variance multiplier is `G/(G−1) × (n−1)/(n−k)`. The package FWL estimator reports a homoskedastic SE.

For the IV benchmark, the authors predict current attendance from the **following weekend’s** attendance. The test partials out the nuisance controls, solves the three just-identified instrument equations and compares all three audiences across all four time windows. It also checks all nine first-stage coefficients and their clustered SEs in Table III B. For example, the strong-audience instrument’s own coefficient is 0.914488 (SE 0.019573). The next-week information is available for historical IV estimation. A prospective DML test should exclude it from features available on the prediction date and keep the days of each weekend together. The afternoon estimate provides a timing comparison with uncertainty around its estimated effect.

The ZIP SHA-256 is `f57e7320c4a2e85523267fb412f890ef48c2a82c7858758f40d0d9f2122be81b`. The daily file `fulliblockday.dta` hashes to `f1dfeca3846d3c1ffc63a692aedaac247c9d5669fd0b17a741264507d258bcbd` and `fulliblock.dta` hashes to `fa2007b80ed8de4f822ef8606027673bc7593551041e2ff026baaac58985630a`. Both are in the same ZIP. The Stata program reviewed for these specifications has SHA-256 `41894cdc57acabcd230c866a98e6cca91b69883855db4262b806a2a4727bb686`. [Download the author’s data](https://econweb.ucsd.edu/~gdahl/movies-code/movies.dta.zip)

## Run from the repository root

With the project environment active, either CLI command prints the unrounded results as JSON and exits with a failure if a comparison misses its target. Running both pytest cases also checks the public CLI output. Each fit uses one BLAS thread, allowing independent replays to run concurrently.

```sh
panels=docs/audits/2026-11-19_time_series_dml/external/panels/replay.py
export DML_TS_BENCHMARK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/dml-ts/benchmarks"
python "$panels" wolfers --cache-dir "$DML_TS_BENCHMARK_CACHE"
python "$panels" dahl --cache-dir "$DML_TS_BENCHMARK_CACHE"
DML_TS_EXTERNAL_REPLAY=1 python -m pytest -q test/integration/test_external_panels.py
```

The author pages offer the archives for download. I found no explicit data redistribution license on those pages. The scripts fetch from the authors, verify cached archives before reading them, and keep the downloaded data outside the repository.
