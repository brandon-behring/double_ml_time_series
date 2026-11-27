# Did more exposed states see larger farm wage gains after bracero exclusion?

For two states whose 1955 exposure differed by ten percentage points, the estimated change in their hourly wage gap after the program ended was **−0.36 cents per hour** in 1965 dollars. The 95% state-clustered interval runs from **−1.21 to +0.50 cents**. This reproduces the first column of Clemens, Lewis and Postel’s Table 1. The estimate concerns changes in relative wages across states. A wage change common to all states is absorbed by the calendar-quarter effects. [Published paper](https://doi.org/10.1257/aer.20170765) · [Authors’ manuscript, Table 1](https://www.ethang.host.dartmouth.edu/clemens_lewis_postel_AER_manuscript.pdf)

## Published estimate and earlier placebo dates

The table uses the paper’s units, **1965 dollars per hour per unit of baseline exposure**. Multiply an estimate by ten to express it in cents per hour for a ten-percentage-point exposure difference. Each placebo uses eight quarters on each side of January 1. Each interval is pointwise, based on 46 state clusters and 45 degrees of freedom. The final column applies Holm’s adjustment to the three placebo tests together.

| Cutoff | Period used | State-quarters | Estimate | Clustered SE | 95% interval | Two-sided p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| January 1965, published policy specification | 1948Q1–1971Q2 | 4,324 | −0.03564 | 0.04260 | [−0.12144, +0.05015] | .4071 | — |
| January 1958, placebo | 1956Q1–1959Q4 | 736 | +0.00991 | 0.01477 | [−0.01983, +0.03965] | .5057 | .5057 |
| January 1959, placebo | 1957Q1–1960Q4 | 736 | +0.04393 | 0.02147 | [+0.00070, +0.08717] | .0466 | .0931 |
| January 1960, placebo | 1958Q1–1961Q4 | 736 | +0.03863 | 0.01608 | [+0.00624, +0.07103] | .0205 | .0615 |

Two individual placebo tests crossed the 5% threshold. All three adjusted p-values exceed .05. The positive slopes show how the observed wage gap moved before exclusion, and the overlapping windows share observations. The earliest exploratory calculation used three years on each side of January 1959. It yielded **+0.04312** across 1,104 state-quarters, with a 95% interval of [+0.00318, +0.08305]. That earlier look informed this retrospective choice of dates, so the three-test adjustment does not account for selection of the research question. [Executable estimates](../external/forward/bracero.py)

## Exposure and timing

The authors average the monthly 1955 Mexican share of each state’s hired seasonal workforce. A recorded workforce with missing Mexican employment receives zero Mexican workers under their cleaning rule. Months without a positive workforce denominator contribute no fraction. The resulting sample has 23 states with zero exposure, 17 below 20%, and six at or above 20%. The underlying data put Nebraska in the high group (32.3%) and Arkansas in the low group (8.0%). The main analysis uses the continuous share. Hourly wages enter once per quarter, deflated to January 1965 prices. The regression includes state and quarter-by-year fixed effects. Its state-clustered covariance permits serial correlation within each state. [Original wage and employment data](https://dataverse.harvard.edu/api/access/datafile/3240617?format=original) · [CPI input](https://dataverse.harvard.edu/api/access/datafile/3240623?format=original) · [Author analysis code](https://dataverse.harvard.edu/api/access/datafile/3240615?format=original)

Federal restrictions began tightening in March 1962. The program ended in December 1964. All three placebo windows close by 1961Q4, whereas the published pre-1965 period includes the tightening. The policy interpretation relies on the wages of states with different exposure following comparable trends in the absence of exclusion. Each placebo asks whether earlier changes were correlated with exposure. Only one nationwide termination occurred. A forward fold trained exclusively before termination sees zero variation in the post-1964 treatment interaction, and the shared shocks estimated from later observed wages were unavailable before the cutoff. [Policy chronology and identification, sections I and III](https://www.ethang.host.dartmouth.edu/clemens_lewis_postel_AER_manuscript.pdf)

The public files combine historical sources and carry observation dates. Individual records lack first-publication or operator-receipt timestamps. The Harvard archive released this version on October 15, 2018. The results are retrospective reconstructions on that public version. The replay verifies the archived program, both input digests, the published sample size and the original rounded coefficient and standard error. [Archive](https://doi.org/10.7910/DVN/17M4ZP) · [Source and inference tests](../../../../test/integration/test_bracero_benchmark.py)

```bash
audit=docs/audits/2026-11-19_time_series_dml
uv run --locked python "$audit/external/forward/bracero.py" --cache-dir "${DML_TS_BENCHMARK_CACHE:-/tmp/dml-external-bench-cache}"
```
