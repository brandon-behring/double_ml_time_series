# Household water conservation and state beer taxes

## Ferraro and Price (2013), Table 2, model A

Cobb County assigned households within meter routes to conservation letters or a control group. The outcome is thousands of gallons used from June through September 2007, measured on the bills issued in July through October. Each coefficient compares one assigned letter with the control group, conditional on the same two pre-treatment consumption totals. The reference fit has one row per premise, an intercept, three arm indicators, no route effects and HC1 heteroskedasticity-robust standard errors. [Published paper and Table 2](https://www.sdu.dk/-/media/files/om_sdu/institutter/iti/forskning/nato%2Barw/literature/using%2Bnonpecuniary%2Bstrategies.pdf) · [Versioned data and code](https://doi.org/10.7910/DVN1/22633)

| Assignment | Premises | Published slope (HC1 SE) | Replayed slope (HC1 SE) |
|---|---:|---:|---:|
| Control, group 4 | 71,643 | Reference | Reference |
| Technical advice, group 3 | 11,675 | −0.24 (0.19) | −0.237215 (0.192965) |
| Weak social appeal, group 1 | 11,675 | −0.99 (0.17) | −0.990100 (0.174944) |
| Peer comparison, group 2 | 11,676 | −1.74 (0.17) | −1.738677 (0.168627) |

We confirmed that the original 158,731-row file yields 106,669 unique analysis premises. The replay drops unassigned households, the archive’s billing-error flag, and the three catastrophic leaks listed in the author code. All four outcome bills, all six June–November 2006 baseline bills, and both April and May 2007 baseline bills must be present. The letter labels above follow the raw `group` variable. [Authors’ Stata code](https://dataverse.harvard.edu/api/access/datafile/2446187)

The offline fixture stores the joint cross-products of the outcome and design, the regression score cross-products, and treatment-arm counts. A Cholesky factor preserves the projections used by the repository’s `fwl_estimate`. The HC1 sandwich uses the original household count. Restoring the classical variance denominator yields a peer-comparison SE of 0.178544. The raw replay independently fits household-level OLS with statsmodels HC1 and checks the two-decimal paper values within 0.005. Expressing the outcome in gallons multiplies every arm’s slope and both SEs by 1,000. A separate check multiplies the peer indicator by ten and verifies that its slope and both SEs divide by ten.

The full-data command also estimates a separate cross-fitted PLR using only the 83,319 peer-comparison and control households, the two baseline consumption totals and the original archive row order. Five shuffled folds, ridge learners and seed 42 give −1.736851 thousand gallons with an influence-function SE of 0.168955. That number is specific to this analysis. A monthly extension would place all records for a premise in the same fold and account for repeated household outcomes and calendar shocks. Household fixed effects would absorb the time-invariant letter indicators, so a before-and-after design would require interactions with the intervention period.

Harvard Dataverse v1.1 specifies CC0 1.0. The original Stata download has SHA-256 `7e42f75dcbd6ff42bc463d9e2039a71d85d1a0683181a5c02910c02055ad908e`. The source program `Cons_Ed_FP.do` has SHA-256 `fc68b290c6b0b572dad7941830f26dbc7770b357827a14634111498eea038f57`. The checked-in fixture contains only aggregate matrices and counts. [Version and license metadata](https://dataverse.harvard.edu/api/datasets/54879/versions/1.1) · [Original data](https://dataverse.harvard.edu/api/access/datafile/2446186?format=original)

## Chetty, Looney and Kroft (2009), Table 6, column 1

The alcohol application estimates an observational state-year demand equation. The dependent variable is the annual change in log per-capita beer consumption. Its regressors are changes in log gross excise and sales tax price factors, the change in log population and year fixed effects. The source drops Hawaii and West Virginia. Taking one-year differences within state and filtering missing observations leaves 1,607 rows across 49 states or jurisdictions from the 1970–2003 panel. [Published paper and Table 6](https://www.korykroft.com/files/Chetty_Kroft_Looney_AER_2009.pdf) · [AEA replication archive V1](https://doi.org/10.3886/E113312V1)

| Tax term | Printed slope (SE) | Replayed slope | Conventional SE | State-clustered CR1 SE |
|---|---:|---:|---:|---:|
| Excise | −0.88 (0.17) | −0.876781 | 0.169076 | 0.132105 |
| Sales | −0.20 (0.30) | −0.199201 | 0.300621 | 0.293508 |

The table note and text specify clustering by state. In the supplied `alcohol_analysis.do`, the command for column 1 is `areg` with absorbed year effects and no cluster option. Both slopes and the conventional SEs round to the published numbers. Explicit state clustering with the usual CR1 finite-sample correction and full year-indicator rank yields the last column. The replay tests those results separately, using native FWL for the excise slope and conventional SE and statsmodels for the original OLS specification and state clusters. The grocery experiment in the same paper has only placeholder observations in its two Stata files. [Published paper](https://www.korykroft.com/files/Chetty_Kroft_Looney_AER_2009.pdf) · [Author-hosted ZIP](https://rajchetty.com/wp-content/uploads/2021/04/CLK_data_code_AER.zip)

The opt-in replay downloads the author’s ZIP, SHA-256 `b96c51c96f4a9489fe55cf4b6d377e76018b4019822f6c7b6127ee2cd1c3270e`. It reads `alcohol_data.dta`, SHA-256 `86b1461f17a0b3ea78c7713d386694c4f33239595f66f46e9a440a1b2172b0fa`. The Stata program names the input `alcohol_consumption_data.dta`. The versioned AEA deposit includes `LICENSE.txt`. The replay keeps the author ZIP in a local cache. [AEA archive contents](https://www.openicpsr.org/openicpsr/project/113312/version/V1/view) · [Author’s data page](https://rajchetty.com/publications/)

## Commands

Run from the repository root in the project Python environment.

```sh
python -m pytest -q -m "not network" test/integration/test_external_taxes_water.py

replay=docs/audits/2026-11-19_time_series_dml/external/taxes_water/replay.py
cache="$HOME/.cache/dml-ts/benchmarks"
python "$replay" water --cache-dir "$cache"
python "$replay" beer --cache-dir "$cache"
python "$replay" water --cache-dir "$cache" --write-fixture /tmp/water_moments.json

DML_TS_EXTERNAL_REPLAY=1 DML_TS_BENCHMARK_CACHE="$cache" \
  python -m pytest -q test/integration/test_external_taxes_water.py
```
