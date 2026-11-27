# Notebook and source-file reproduction outside the study registry

Both original notebooks stopped before completion. The validation workflow also labeled an assumed 95% coverage rate as its result. After correcting the examples, isolated Jupyter kernels completed all 22 code cells. Separate checks recovered the Fulton fish estimates from the dated teaching release, matched the full 2018 journal roster against fresh Crossref responses, and reconstructed PROGRESA attendance counts through two file formats. Each check establishes a different kind of evidence. [Notebook replay](additional/notebooks.json) · [Fulton](additional/fulton.json) · [Journal roster](additional/journals.json) · [PROGRESA](additional/progresa-raw.json)

## Validation notebook

The version at commit `0cca908` failed in its import cell because the `src.validation` namespace had moved. Both functions labeled “naive” and “with controls” included all confounders. The notebook reported PASS and 0.95 coverage regardless of the fits and used quantiles of estimation errors as a confidence interval for mean bias. On the fixed example dataset, the treatment-only slope is **0.748** and the adjusted slope is **2.144**, against a true effect of 2.0. These are errors for a single realization. The generator redraws the signs and magnitudes of the confounder coefficients across realizations, so mean signed error can conceal serious omitted-variable error on individual datasets. [Corrected notebook](../../../notebooks/example_validation.ipynb) · [Data-generating process](../../../dml_ts/validation/dgp_generator.py)

The corrected run computes an OLS interval for the adjusted treatment coefficient in **each of 1,000** independent datasets. **957/1,000** intervals cover the known effect (exact binomial $p=0.345$ against 95%). Estimated mean bias is **+0.00389**, with a Monte Carlo 95% interval of **[−0.00103, +0.00881]**. The declared ±0.01 bias tolerance and separate coverage check yield PASS in this example. The serial run and four-worker run use the same per-draw seeds. All 1,000 point estimates and pairs of interval limits agreed to the specified numerical tolerance. The data have independent observations and a correctly specified linear conditional mean. These results cannot establish time-series DML coverage. [Before/after execution record](additional/notebooks.json)

## Historical bias-results notebook

The original stopped at the highest-absolute-bias ranking because it passed a Series where pandas requires a column key. Fixing the ranking and creating missing output directories allowed all four figures and both LaTeX tables to be generated in a temporary directory. The notebook reads **81 archived aggregates representing 8,100 historical fits**. This replay recomputes summaries from those aggregates. The original absolute-bias thresholds give **43 PASS, 25 WARNING and 13 FAIL**. The largest absolute bias is **0.4031**, at $n=1,000$, 20 covariates, treatment effect 3 and confounding strength 2. The three sample sizes 500, 1,000 and 2,000 produced 4, 5 and 4 FAIL labels respectively out of 27 settings each. That evidence contradicts the notebook's former recommendation that $n\ge1,000$ ensures a PASS. Five folds was the only fold count used. [Corrected notebook](../../../notebooks/method1_bias_validation_results.ipynb) · [Archived fits and metadata](../../../results/simulations/BIAS_VALIDATION_results_20251114_103701.csv) · [Execution record](additional/notebooks.json)

The archival field labeled coverage ranges from 0.09 to 0.30. Those intervals used the standard error of the Monte Carlo **mean** around each individual estimate. The revised notebook removes this field from its summaries of estimator coverage and its exported table. Recalibrating a real per-fit interval requires access to the estimates and appropriate uncertainty from each fit, or new runs. The earlier audit independently rescored the point-estimator bias evidence using the archived moments. [Historical scoring audit](../2026-11-19_time_series_dml/08-bias-test-correction.md)

## Fulton Fish Market

New downloads matched the byte hashes of both public releases. The MIT author's Stata file contains **97 rows and 14 fields**, with weekday indicators but no calendar date. The pinned teaching release contains **111 dated trading days** from 1991-12-02 to 1992-05-08, spanning 32 stormy, 34 mixed and 45 fair days. A common sample definition matters when testing replication of published quantities. [Author data](https://economics.mit.edu/sites/default/files/inline-files/fish.dta) · [Dated teaching file](https://github.com/ccolonescu/PoEdata/blob/c41bee8fb53ccdec53961fc9fd85b7b84bd17d1c/data/fultonfish.rda) · [Independent extraction and calculation](additional/fulton.py)

After residualizing log price, log quantity and the storm indicator on a constant and four weekday indicators on the teaching data, the OLS coefficient is **−0.56255**, the storm first stage **+0.34641**, the reduced form **−0.38777** and their IV ratio **−1.11942**. These reproduce the repository's dated-file benchmark. Across five successive date blocks, storm counts are **9, 10, 8, 5 and 0**. The last block still permits scoring a price or quantity prediction. It has no variation for estimating its own storm-IV effect. Information from earlier blocks could support an explicit transport model, conditional on IV validity and stability. Source provenance and instrument support prevent treating the five block slopes as five independently verified demand elasticities. [Measured calculations](additional/fulton.json) · [Original simultaneous-equations study](https://doi.org/10.1111/1467-937X.00141)

## Three 2018 journal rosters

Using an empty cache and Crossref's print metadata restored all **215 entries** in the repository's ledger across the stipulated volumes of AER (125), AEJ Applied (50) and QJE (40). The existing coding classifies 158 as empirical. Its temporal targets count 84 historical, 41 later measurements of the original exposure, 23 model counterfactuals, 5 outcome predictions, 3 effect projections and 2 external-applicability studies. The fresh metadata validates membership and bibliographic identity. The target counts come from rerunning the existing human labels. Independent rereading or double coding would be needed to assess their reliability. Three journal volumes, heterogeneous questions and a nonrandom scope cannot identify the field-wide frequency of causal mistakes or missed temporal validation. [Roster receipt](additional/journals.json) · [Coding and scope](../2026-11-19_time_series_dml/forward/journals.md) · [Crossref metadata provenance](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)

## PROGRESA raw files and original Fortran

The checksum-pinned Todd–Wolpin archive contains 14,039 estimation child-wave records and 19,500 in the expanded file, including 5,461 additional 1998 treated records. There are 3,923 1998 controls in the estimation data. Among matched common rows, the father's age differs in 3,872 and the 1998 flag in 425. The direct age-12–15 attendance reconstruction from the expanded data agrees with the original compiled Fortran reader's counts. [Raw-file result](additional/progresa-raw.json) · [Archive](https://doi.org/10.3886/E116240V1) · [Study](https://doi.org/10.1257/aer.96.5.1384)

| Group | Treated attending | Controls attending | Difference, percentage points | Village-clustered 95% interval |
|---|---:|---:|---:|---:|
| Girls | 166 / 223 | 117 / 176 | +7.96 | [−2.55, +18.48] |
| Boys | 200 / 262 | 132 / 182 | +3.81 | [−6.00, +13.61] |

The authors' budget subroutine returned grade-seven subsidies of 1,745.15 and 1,662.05 in its deflated peso units for girls and boys. This independently checks the reader, arm totals and that part of the policy calculation. It is a partial program replay. The structural model's full estimation and the timing of its original freeze are separate claims. In particular, predicting from a 1998 comparison group uses later information than issuing a prediction before policy rollout. DML fitted solely to pre-program zero exposure has no treatment variation from which to learn a subsidy effect. [Fortran driver](../2026-11-19_time_series_dml/external/forward/progresa_raw_check.f90) · [Design and paper-version comparison](../2026-11-19_time_series_dml/forward/progresa.md)

## Reproduction commands

From the repository root, use a project Python environment with Rscript for Fulton and gfortran for the PROGRESA program checks. The notebook runner uses two additional Jupyter dependencies in an isolated environment. It creates clean temporary notebook working directories, and the optional git reference executes the uncorrected versions before the current sources.

```bash
audit=docs/audits/2026-11-21_evaluation_methods
forward=docs/audits/2026-11-19_time_series_dml/external/forward
uv run --locked --with nbclient --with ipykernel --with seaborn python "$audit/additional/notebooks.py" --output /tmp/eval-notebooks.json --original-ref 0cca908463719d46dbe43a44c17a6d2dd81ceb89
uv run --locked python "$audit/additional/fulton.py" --cache-dir /tmp/eval-fulton-cache --output /tmp/eval-fulton.json
uv run --locked python "$forward/journals.py" --verify-roster --cache-dir /tmp/eval-print-rosters
uv run --locked python "$forward/progresa_raw.py" --cache-dir /tmp/eval-progresa --check-fortran
```
