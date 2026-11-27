# Rechecking the published 401(k) estimates

The repository’s default random forest gives **$7,025** and its lasso gives **$6,705** on the tracked 1991 Survey of Income and Program Participation sample. The corresponding fivefold homogeneous eligibility estimates in Chernozhukov et al. (2018), Table 2, Panel B, are **$9,247** and **$8,187**. All 9,915 rows and all 14 columns of the tracked CSV match the pinned author Stata file value for value in their original order. The discrepancy therefore survives a direct check of data identity. [Published paper, §6.2](https://doi.org/10.1111/ectj.12097) · [Author manuscript, Tables 2–3](https://www.nber.org/system/files/working_papers/w23564/w23564.pdf) · [Tracked fixture](../../../test/fixtures/401k_data.csv) · [Repository method audit](02-code-and-methodology.md)

The completed historical-code refit gives **$9,243.65 for forest** and **$8,182.27 for rigorous lasso** in fivefold partially linear regression. These are the medians of 100 fresh partitions. The largest two point-estimate differences in the full comparison arise in the interactive “Best” column. Across all 200 interactive partitions, its estimate and standard error exactly equaled those of rigorous lasso, as the public R selection code implies. The original split assignments were unavailable. [Full comparison](401k-summary.csv) · [Pinned author score and learner selection](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/Moment_Functions.R)

## Three economic questions in the two tables

| Panel | Treatment and comparison | Interpretation and identifying assumption |
|---|---|---|
| Table 2A | Eligibility (`e401`), interactive outcome regressions by eligibility | ATE of employer-provided access on net financial assets. Identification assumes conditional exchangeability and overlap. The author code drops observations outside the open propensity interval (0.01, 0.99). |
| Table 2B | Eligibility, partially linear regression using pooled cross-fitted residuals | Constant marginal effect under the partially linear model. Under heterogeneous effects and conditional exchangeability the coefficient generally weights effects by the conditional treatment variance. |
| Table 3 | Participation (`p401`), with eligibility as the instrument | Complier effect on net financial assets, requiring instrument relevance, conditional independence, exclusion and monotonicity. Among 3,682 eligible records, 2,594 participate. Participation is zero in all 6,233 ineligible records. The author code trims the estimated instrument propensity. |

The nine controls are age, income, education, family size, marital status, two-earner status, defined-benefit pension, IRA participation and homeownership. Trees and neural nets use the raw controls. Rigorous lasso uses powers through 6, 8, 4 and 2 for the first four variables and two-way interactions between terms in the R formula. The authors fit six nuisance specifications and add a seventh column selecting a predictor for each nuisance function. The Ensemble combines rigorous lasso, boosting, random forest and neural net using inner validation. This cross-sectional example tests score implementation and empirical provenance. Panel/time-series validation requires repeated observations or randomized periods with a specified time-dependent assignment law. [Pinned Table 2 source](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/401K.R) · [Pinned Table 3 source](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/401K-LATE.R)

## Published targets and fresh estimates

<!-- begin numeric output -->

**600 of 600 joint jobs returned all seven finite estimates and standard errors.** Of the 42 cells, 42 have a finite refit and 42 have 100 usable partitions (42 from joint runs and 0 from single-learner runs).

Among the 42 point estimates with 100 usable partitions, the median absolute difference from publication was $42.00. The largest was $671.61 for 2A · Eligibility, interactive (2 folds, Best, joint), with $6,874.39 fresh and $7,546 published. The largest absolute split-adjusted SE difference was $756.29 for 3 · Participation, LATE (5 folds, Lasso, joint), with $2,550.71 fresh and $3,307 published.

**Coverage by panel and run family.** The first numeric column counts completed seven-method jobs. Each learner column gives usable partitions out of 100 and the run family used for its median.

| Panel and folds | Joint jobs | Lasso | Tree | Forest | Boosting | Neural net | Ensemble | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2A · Eligibility, interactive, 2 | 100 / 100 | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint |
| 2A · Eligibility, interactive, 5 | 100 / 100 | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint |
| 2B · Eligibility, PLR, 2 | 100 / 100 | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint |
| 2B · Eligibility, PLR, 5 | 100 / 100 | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint |
| 3 · Participation, LATE, 2 | 100 / 100 | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint |
| 3 · Participation, LATE, 5 | 100 / 100 | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint | 100 joint |

**Point estimates in dollars**, fresh median / published median. A dash marks a cell awaiting a finite fit.

| Panel and folds | Lasso | Tree | Forest | Boosting | Neural net | Ensemble | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2A · Eligibility, interactive, 2 | 6,874 / 6,830 | 7,643 / 7,713 | 7,933 / 7,770 | 7,793 / 7,806 | 7,603 / 7,764 | 7,688 / 7,702 | 6,874 / 7,546 |
| 2A · Eligibility, interactive, 5 | 7,174 / 7,170 | 8,046 / 7,993 | 8,125 / 8,105 | 7,690 / 7,713 | 7,756 / 7,788 | 7,794 / 7,839 | 7,174 / 7,753 |
| 2B · Eligibility, PLR, 2 | 7,700 / 7,717 | 8,669 / 8,709 | 9,046 / 9,116 | 8,786 / 8,759 | 8,887 / 8,950 | 8,948 / 9,010 | 9,028 / 9,125 |
| 2B · Eligibility, PLR, 5 | 8,182 / 8,187 | 8,880 / 8,871 | 9,244 / 9,247 | 9,081 / 9,110 | 9,032 / 9,038 | 9,109 / 9,166 | 9,162 / 9,215 |
| 3 · Participation, LATE, 2 | 8,652 / 8,978 | 11,047 / 11,073 | 11,294 / 11,384 | 11,321 / 11,329 | 11,113 / 11,094 | 11,021 / 11,119 | 10,900 / 10,952 |
| 3 · Participation, LATE, 5 | 8,886 / 8,944 | 11,498 / 11,459 | 11,617 / 11,764 | 11,096 / 11,133 | 11,125 / 11,186 | 11,211 / 11,173 | 11,090 / 11,113 |

<details><summary>Median within-partition standard errors, fresh / published ($)</summary>

| Panel and folds | Lasso | Tree | Forest | Boosting | Neural net | Ensemble | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2A · Eligibility, interactive, 2 | 1,283 / 1,282 | 1,208 / 1,208 | 1,264 / 1,276 | 1,158 / 1,159 | 1,342 / 1,328 | 1,144 / 1,149 | 1,283 / 1,360 |
| 2A · Eligibility, interactive, 5 | 1,201 / 1,201 | 1,191 / 1,198 | 1,235 / 1,242 | 1,155 / 1,155 | 1,238 / 1,238 | 1,132 / 1,134 | 1,201 / 1,237 |
| 2B · Eligibility, PLR, 2 | 1,327 / 1,346 | 1,355 / 1,363 | 1,288 / 1,302 | 1,331 / 1,339 | 1,329 / 1,335 | 1,303 / 1,309 | 1,296 / 1,304 |
| 2B · Eligibility, PLR, 5 | 1,298 / 1,298 | 1,352 / 1,358 | 1,292 / 1,295 | 1,318 / 1,314 | 1,326 / 1,322 | 1,297 / 1,299 | 1,291 / 1,294 |
| 3 · Participation, LATE, 2 | 2,317 / 2,192 | 1,732 / 1,749 | 1,808 / 1,832 | 1,663 / 1,666 | 1,889 / 1,903 | 1,640 / 1,653 | 1,647 / 1,657 |
| 3 · Participation, LATE, 5 | 2,379 / 2,259 | 1,713 / 1,717 | 1,771 / 1,788 | 1,657 / 1,661 | 1,795 / 1,795 | 1,626 / 1,641 | 1,638 / 1,645 |

</details>

<details><summary>Split-adjusted standard errors, fresh / published ($)</summary>

| Panel and folds | Lasso | Tree | Forest | Boosting | Neural net | Ensemble | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2A · Eligibility, interactive, 2 | 1,314 / 1,530 | 1,253 / 1,271 | 1,341 / 1,363 | 1,175 / 1,202 | 1,432 / 1,468 | 1,157 / 1,170 | 1,314 / 1,533 |
| 2A · Eligibility, interactive, 5 | 1,204 / 1,398 | 1,220 / 1,236 | 1,266 / 1,299 | 1,162 / 1,177 | 1,271 / 1,293 | 1,138 / 1,148 | 1,204 / 1,294 |
| 2B · Eligibility, PLR, 2 | 1,363 / 1,749 | 1,394 / 1,427 | 1,339 / 1,377 | 1,347 / 1,382 | 1,372 / 1,408 | 1,319 / 1,344 | 1,314 / 1,357 |
| 2B · Eligibility, PLR, 5 | 1,306 / 1,558 | 1,380 / 1,418 | 1,314 / 1,328 | 1,325 / 1,328 | 1,340 / 1,355 | 1,305 / 1,310 | 1,299 / 1,312 |
| 3 · Participation, LATE, 2 | 2,592 / 3,014 | 1,792 / 1,849 | 1,924 / 1,993 | 1,686 / 1,718 | 2,036 / 2,098 | 1,674 / 1,689 | 1,671 / 1,699 |
| 3 · Participation, LATE, 5 | 2,551 / 3,307 | 1,749 / 1,786 | 1,820 / 1,893 | 1,672 / 1,710 | 1,831 / 1,890 | 1,636 / 1,678 | 1,650 / 1,675 |

</details>

<!-- end numeric output -->

Each cell counts partitions with a finite estimate and standard error. Base-learner estimates can be recovered from their completed outer folds while the joint run continues to fit the ensemble. The aggregator checks the recovered values against the complete joint row when it arrives. [Checkpoint scoring](score_401k_checkpoints.R)

The summary uses the run family with more usable partitions for each cell, preferring a joint fit on a tie. Each reported median uses one run family. The split-level file identifies each run as joint or single learner. A one-partition value is sensitive to fold allocation. The machine-readable comparison includes all 42 published median estimates, their median standard errors, their split-adjusted standard errors, matched fresh values and dollar differences. The adjusted SE is the square root of the median of `se_s² + (theta_s − median(theta))²`, following the paper’s variance aggregation. The CSV also reports the median of the individual square roots, corresponding to the order of operations in the author driver after centering each method separately. With an even number of draws, interpolation can make these values slightly different. [Full numerical comparison](401k-summary.csv) · [Fresh split-level values](401k-draws.csv.gz) · [Aggregation code](summarize_401k.py)

## Corrected Python comparison, November 20

The companion uses one fivefold split with seed 42. It fits a conditional effect that can vary with eleven inputs and then averages the fitted effects over the sample. Two inputs measure wealth. The variable `nifa` records net non-401(k) financial assets and `tw` records total wealth. The published constant-slope specification uses nine adjustment variables. The table gives dollars of net financial assets per household. The last column divides the signed Python–publication gap by the published coefficient. [Journal article, Tables 2 and 3](https://doi.org/10.1111/ectj.12097) · [Executable comparison](../../../scripts/run_401k_replication.py)

| Eligibility fit | Published Table 2B | Fresh author-code median, 100 partitions | Python mean, one partition | Python SE for mean | Python 95% interval | Python minus published |
|---|---:|---:|---:|---:|---:|---:|
| Random forest | 9,247 | 9,243.65 | 7,025.17 | 491.74 | [6,061.38, 7,988.96] | −2,221.83 (−24.0%) |
| Lasso | 8,187 | 8,182.27 | 6,705.35 | 357.09 | [6,005.47, 7,405.23] | −1,481.65 (−18.1%) |

The API now obtains the standard error and interval from EconML's inference for the mean fitted effect. Both estimates fall outside the default 15% distance from the paper. The API reports that descriptive distance as a signed dollar and percentage gap. The fits share observations and their joint sampling covariance is unavailable, so a p-value for that gap cannot be calculated from the two marginal SEs. For a direct fit with participation as treatment, the API returns `NO_BENCHMARK`. The published Table 3 effect for compliers requires eligibility as an instrument. [Companion API](../../../dml_ts/validation/empirical_replication.py) · [EconML mean-effect inference](https://www.pywhy.org/EconML/_autosummary/econml.inference.PopulationSummaryResults.html)

To reproduce the Python column on the tracked fixture, run `python scripts/run_401k_replication.py` from the repository root. Repeating the comparison with nine and eleven effect modifiers reproduced all four mean effects and intervals in the decomposition. The full 42-cell comparison above uses the 4,200 joint draw records. The archive also contains 642 single-learner pilot records. [Covariate sensitivity script](../../../scripts/test_issue_c5_401k_covariates.py) · [Unrounded data](401k-summary.csv) · [Portable draw archive](401k-draws.csv.gz)

The revised chapter built in both formats. Its printed code reconstructed all 42 point-estimate medians from the joint records. The 210-page PDF and web chapter display the verified comparisons. The package passed 711 tier 1/2 tests and 51 doctests, and the Sphinx documentation built. [Chapter 3](../../../chapters/chapter_03.tex) · [Web chapter](../../../web/src/content/chapters/chapter03-validation-framework.mdx)

## Python results at the November 19 audit snapshot

| Repository fit, seed 42 | Fresh mean effect | Reported “SE” | Reported interval | Coded comparator | Actual Table 2B fivefold reference | Difference from actual reference |
|---|---:|---:|---:|---:|---:|---:|
| 500-tree RF | 7,025.17 | 3,208.75 | [736.02, 13,314.33] | 9,127 | 9,247 | −2,221.83 |
| LassoCV and logistic CV | 6,705.35 | 1,938.58 | [2,905.73, 10,504.97] | 9,580 | 8,187 | −1,481.65 |

Both fits called the repository methods with their default arguments and supplied the tracked CSV through the constructor’s `data_path`. The methods add `nifa` and `tw` to the nine published controls, pass all 11 as effect modifiers to EconML and average fitted conditional effects. At the audit snapshot their reported interval averaged the endpoints of individual effect intervals. Averaging that width does not estimate the sampling variance of the mean effect. The RF uses 500 trees, minimum leaf size 10 and a classification nuisance for binary treatment. The authors’ forest uses 1,000 trees, node sizes 5 and 1 and numeric-response treatment predictions. Repository lasso applies cross-validated penalties to the selected unexpanded columns. The default lasso run emitted convergence warnings. A historical ignored November 2025 log records an RF mean of $7,946.16 and a reported SE of $5,021.60. Its unrecorded environment and design preclude attribution of that change. [Repository evaluator](../../../dml_ts/validation/empirical_replication.py) · [Author nuisance fits](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/ML_Functions.R)

Holding the random seed and learner settings fixed, the following decomposition changes the controls and final effect model in sequence. ATE intervals come from EconML's inference for the mean effect. The default point estimates agree with both versions of the repository calls. The averaged CATE limits reproduce the November 19 output. The lasso optimizer raised 56 convergence warnings under the original specification and 46 under each nine-control specification when warnings were recorded individually. [Decomposition driver](decompose_401k.py) · [Unrounded results, inference and fixture digest](401k-decomposition.csv)

| Role of controls in EconML | RF mean [95% ATE interval] | Lasso mean [95% ATE interval] |
|---|---:|---:|
| Eleven controls as effect modifiers, repository design | 7,025 [6,061, 7,989] | 6,705 [6,005, 7,405] |
| Nine published raw controls as effect modifiers | 8,064 [6,118, 10,010] | 3,879 [1,644, 6,113] |
| Same nine as adjustment variables, constant effect | 9,144 [6,556, 11,733] | 4,165 [1,813, 6,518] |

Changing the RF specification closes most of its point-estimate gap in this partition. The remaining difference from $9,247 is $103. The lasso series shows why correcting the final model and the control set cannot establish numerical replication. Its feature basis, penalty rules and split aggregation also differ from the paper.

## Source recovery and numerical fidelity

Wiley lists a 538.2 KB replication ZIP. Its direct and alternate supplement links returned HTML in this audit. The authors also link their public GitHub source, and its pinned checkout contains the data, model functions and two 401(k) drivers. Its seven-commit history and tutorial ZIP contain no archived 100-partition results or original fold assignments. The tutorial is a separate two-repetition illustration. The publisher ZIP contents could differ from the GitHub source. The source file `sipp1991.dta` has SHA-256 `1123d5f0abf6adae1d8e200f756d2a22b0ce0ce30cb228e69342a0098e57b4b2`. All fresh estimates in this report are trained from that file. [Publisher supplement listing](https://onlinelibrary.wiley.com/doi/full/10.1111/ectj.12097) · [Pinned author source and data](https://github.com/VC2015/DMLonGitHub/tree/b91cbf96c01eccd73367fbd6601ecdd7aa78403b)

The author drivers declare `set.seed(1211)` and 100 partitions. They default to two folds, one partially linear or LATE row, and create a 12-worker cluster without registering a foreach backend in the scripts. There is no original fold file or documented per-partition seed stream. Our driver uses seeds 1211–1310 separately at each fold count and panel and calls the author DML2 function for each draw. The original supplied R functions perform model fitting, trimming and pooled estimation. Fresh draws can test a table's statistical and programming implementation even when they differ from its exact original random partitions. [Author moment and ensemble source](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/Moment_Functions.R)

Several source choices matter for numerical interpretation. The empirical methods discussion describes tenfold tuning for boosting, while the 401(k) code passes fivefold CV to `gbm` and selects the CV minimizing number of trees using `gbm.perf`. In the interactive Best path the author's repeated `length(methods)>1` condition overwrites each selected index with `which.min(mean(...))`, a scalar index of one. Consequently its Table 2A Best column uses the first learner, rigorous lasso, on every split. The published Best and lasso columns differ. Changing the seed cannot reconcile those two columns for this source version. The joint run preserves the observed behavior. In the single-learner run the guard is corrected solely to make the redundant Best branch executable, then its duplicate column is discarded. The interactive ensemble function evaluates its control-outcome prediction against the treatment column for control records when choosing mixture weights. The source LATE standard error scales the numerator score by the mean first stage and omits the influence variation of that estimated denominator. A modern implementation of the paper's score can therefore report a different standard error. [Author implementation](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/Moment_Functions.R) · [Author LATE uncertainty calculation](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/ML_Functions.R)

**How the released driver aggregates uncertainty.** The paper centers each learner's estimates on its own median. The R drivers subtract a vector of seven medians directly from a data frame, which recycles the vector down the rows. Using the same 100 fivefold lasso participation fits, the paper's rule yields **$2,550.71**. The driver's expression yields **$3,252.49** in ascending partition order and **$3,321.66** when those rows are reversed. The printed SE was **$3,307**. Across all 42 cells, the median absolute SE difference from publication is $35.10 under the paper's rule and $14.85 under the driver with ascending IDs. This sensitivity helps explain why even close point estimates can accompany materially different SEs. The original split assignments and row ordering are unavailable. The check matched the original R expression in both orderings for all 42 cells. [Run the fixed-draw check](check_401k_se_aggregation.R) · [Published driver](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/401K.R#L113-L121)

Table 3’s Best selector also uses different indices for different operations. It trims on the candidate selected for eligibility-propensity prediction, then weights outcomes with the propensity from the candidate selected for ineligible-household outcomes. It chooses the ineligible outcome regression using the loss for participation among ineligible households, which is structurally zero. The original boosting branch compares the eligible participation prediction with wealth when recording that selection loss, and the LATE ensemble computes two validation losses with swapped outcome arms. These choices affect how Best selects from the nuisance predictions. In one twofold forest partition (seed 1213), 9,912 records survived trimming, the estimated first stage was 0.69526, and the LATE was $12,253.72. Keeping the same data, predictions and point estimate, the code’s numerator-only SE was $1,795.84 and the conventional ratio influence-function SE was $1,802.04. [Pinned scoring and selection implementation](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/Moment_Functions.R) · [Paper, §5.2](https://doi.org/10.1111/ectj.12097)

## Reproduce the audit

Use the pinned source with R, the recommended packages `foreign`, `rpart`, `nnet`, and CRAN packages `sandwich`, `randomForest`, `gbm` and `hdm`. The audit selected `hdm` 0.2.0, available when Wiley first published the paper in June 2017. The driver verifies data and source digests before fitting. It writes an atomic partition CSV on success, an error file on failure, and checkpointed nuisance predictions in the joint run. The audit used R 4.5.2, foreign 0.8-91, rpart 4.1.24, nnet 7.3-20, sandwich 3.1-3, randomForest 4.7-1.2 and gbm 2.3.1. The ensemble evaluates candidate mixture weights via fold-level cross-products to reduce runtime. Near-tied weights are rechecked with the original literal R matrix product in grid order. This resolves a synthetic case with two identical validation predictors in which rounding otherwise changes the winning weights. The accelerated grid and the original enumeration produced the same 23,426 eligible weight vectors in the same order. All eight whole-grid numerical checks chose the same minimizing vectors, including ties. A final 1,250-record nested twofold check for each of PLR, interactive ATE and LATE gave zero discrepancy in point estimates or standard errors and identical final RNG states. Those integration checks used the raw nine-variable basis across learners, 50 trees/boosting iterations and 75 neural-net iterations to test the implementation within a short run. Separate checks retained the default learner iterations and also agreed across all three estimands. Its model-specific tuning and original indexing are retained. When resuming, give the driver the same source, R packages, math libraries, output directory and arguments. For specific rows, set `DML_401K_JOBS` to a CSV with columns `panel,folds,partition`. The driver claims each row exclusively in the output directory and releases ownership at normal completion. If a process was killed, verify that its recorded PID and host are inactive before removing its stale claim. Choose the worker count to fit the available RAM. [CRAN historical hdm releases](https://cran.r-project.org/src/contrib/Archive/hdm/) · [Refit driver](replicate_401k.R)

Within one joint job, the ensemble refits its full-training rigorous lasso on exactly the observations and formula already used in the main learner. The driver reuses that fit only when the original call left the RNG state unchanged and all ordered training and test row IDs, formula, method inputs and parameters agree. On three independently computed full-sample PLR fold pairs the reused fitted outputs matched every saved prediction, residual and fit diagnostic byte for byte, with the same subsequent RNG state. Reuse took 9–12 milliseconds. The corresponding historical-code refits took approximately 163, 241 and 285 seconds from adjacent checkpoint timestamps. A test that changed training row order forced a fresh fit. Other learners follow the original fitting path.

The audit skips two auxiliary fits that the pinned source computes and discards. With `hdm` 0.2.0, the interactive eligibility and LATE scores omit the unconditional rigorous-lasso fit for wealth. LATE also omits the unconditional logistic lasso for participation. All seven published columns use other predictions for those scores. PLR retains both required fits. In a historical-polynomial LATE test on 1,500 training and 500 holdout SIPP records, the original fold took 254.427 seconds and the revised fold took 69.069 seconds. The two discarded models accounted for 174.105 seconds in the original fold. All ten consumed prediction or loss fields and the final RNG state matched exactly. This timing measures one nuisance fold; total replication time also includes the other learners and fold sizes.

End-to-end checks used seed 1211 and the same 1,250 SIPP records, two folds, nine raw controls, 50 trees or boosting iterations and 75 neural-net iterations. For all three panels, the two driver versions produced identical seven-method estimates, standard errors, diagnostic matrices, draw records and final RNG states. Each direction of a mixed-version restart also matched after loading 17 of the 34 call checkpoints from the other version. The checkpoint scorer independently returned byte-identical CSV rows for three full-sample, twofold SIPP partitions after removal of the unused fields. Already-loaded workers finish with their original driver. [Refit driver](replicate_401k.R) · [Checkpoint scorer](score_401k_checkpoints.R)

A real full-sample three-row test returned identical estimates and standard errors under one and three workers. The regression-tree twofold checks were $9,149.75 (PLR, partition 1), $8,005.85 (interactive, partition 2), and $11,807.09 (LATE, partition 3). Thirty recovered base-learner checkpoint results also agreed with rerunning the pinned author score on exactly the same cached outer predictions, covering both five- and two-fold panels where predictions had completed.

From the dated audit directory, a fresh historical-code run can be started with the following commands. Substituting a learner name for `all` computes its own single-learner partition series. The final command can also be run without the checkpoint argument to rebuild both numerical tables from the compressed evidence snapshot.

```sh
git clone https://github.com/VC2015/DMLonGitHub.git /tmp/401k-authors
git -C /tmp/401k-authors checkout b91cbf96c01eccd73367fbd6601ecdd7aa78403b
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript replicate_401k.R /tmp/401k-authors /tmp/401k-checkpoints 100 8 all
python summarize_401k.py /tmp/401k-checkpoints
```

### Parallel execution on the audit workstation

The unit of parallel work is one panel, fold count and partition. Each job produces seven estimates together because the Ensemble and Best specifications use results from the underlying learners. The workstation has **64 physical cores, 128 logical CPUs and 123 GiB RAM**. An initial sample measured about 33 core equivalents in R. A later sample with eight additional independent fits measured roughly 49 core equivalents across all applications. Every observed R fit used one thread. The published boosting configuration also sets `n.cores=1`. Partition-specific seeds permit the scheduler to change concurrency without changing a partition’s inputs. On the full 9,915-row sample, resuming twofold PLR partition 2 from saved nuisance fits reproduced all seven estimates and standard errors from its earlier run exactly, including the Ensemble. [Author tuning configuration](https://github.com/VC2015/DMLonGitHub/blob/b91cbf96c01eccd73367fbd6601ecdd7aa78403b/401K.R) · [Resource-aware dispatcher](dispatch_401k.py)

A paired full-sample fivefold PLR benchmark placed the same split and expanded rigorous-lasso design on two different physical cores. Single-thread netlib BLAS took **321.7 seconds** and single-thread OpenBLAS 0.3.32 took **327.7 seconds**. Both used more than 99% of their allotted core time. Combining that fold with four identical saved folds gave the same $8,269.30 coefficient and $1,280.93 standard error to the cent. This was one paired timing. The audit retained netlib BLAS.

The local dispatcher checks CPU and memory every ten seconds. Its admission target is **50 host-wide busy core equivalents**, with **16 GiB** reserved for the rest of the machine. The initial 2.2 GiB allowance was revised after a five-fold participation fit reached 2.67 GiB. The dispatcher now budgets at least **3.0 GiB for five-fold LATE, 2.4 GiB for other five-fold fits and 2.2 GiB for two-fold fits**. A group’s measured peak plus 10% takes precedence when higher. All 30 live workers retained their process IDs and continued through this scheduler change. It counts existing fits, reserves room for their memory to grow and caps new starts at five per check. A job claims its partition exclusively. Each completed nuisance fit saves its predictions and RNG state. The dispatcher validates the seven learner labels and finite estimates and standard errors, archives failed outputs for inspection, and refreshes this report every two minutes during the workstation run. Completion of the joint replication requires **600 validated partition files and 4,200 finite results**.

The dispatcher recorded completion at **11:04 UTC on November 20, 2026**, with zero failed jobs. An independent check matched all 600 raw partition files to all 4,200 joint estimates and standard errors in the compressed archive. It also recalculated the point medians, median standard errors and split-adjusted standard errors for all 42 table cells. The dispatcher sampled up to 42 audit R processes, and system memory available at its sampled minimum was 40.07 GiB. Local run logs, checkpoints and the final `dispatcher/status.json` remain under `results/audit_2026-11-19/401k/`. [Portable split-level archive](401k-draws.csv.gz) · [Final comparison](401k-summary.csv)
