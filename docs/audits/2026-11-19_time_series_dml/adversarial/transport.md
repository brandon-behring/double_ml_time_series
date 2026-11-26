# Does adjustment improve a current effect estimate or a forecast of a later effect?

These retrospective checks use three existing source archives. Each earlier effect estimate averages over people or matches observed at the time. Carrying that number forward assumes some stability in effects and population. Even perfect estimation in the earlier population leaves that assumption open. The later benchmark is a noisy randomized estimate. The comparisons below were selected after reviewing the data. [Full results, including every historical forecast](transport.json) · [Replay](transport.py) · [Prospective issuances and digests](../prospective/README.md)

## Cricket gives both questions a numerical answer

First, take the same 946 eligible matches in calendar 2025. Train a model for the focal team's win credit on the 3,965 eligible archived matches before 2025 using only pre-toss features. With a fair toss, write `A` for focal assignment, `Y` for win credit, and `b` for a pre-toss prediction. The match score in percentage points is `200(2A−1)(Y−b)`. Its expectation over the toss equals the focal team's treatment effect for any fixed pre-toss prediction. For the two-arm regression, `b` is the mean of the two predicted outcomes. The pooled regression uses the same features and one model trained without toss assignment.

| Adjustment during estimation | 2025 effect (pp) | SE across 158 series (pp) | SD under independent tosses and a sharp null (pp) |
|---|---:|---:|---:|
| Constant win credit of one-half | 3.594 | 3.020 | 3.203 |
| Single pre-trained prognostic ridge model | 2.860 | 2.765 | 2.941 |
| Two pre-trained arm-specific ridge models | 2.717 | 2.860 | 2.947 |

Under the sharp null each match's two potential win credits agree. Holding the fixture slate, opponents, credits and predictions fixed, the randomization SD is `200 × sqrt(sum((Y−b)²)) / n`. This conditional calculation assumes independent fair tosses and rules out changes to later opponents or outcomes through tournament progression or carryover. Both adjustments reduced that conditional SD. The single prognostic model also had the smallest estimated series-clustered SE in this season. Variance reduction supplies a reason to model the outcome even with known assignment. These numbers provide little incentive to split the training data by arm here. [Randomized regression adjustment](https://pmc.ncbi.nlm.nih.gov/articles/PMC5111650/) · [Match design and coverage](../prospective/cricket.md)

Second, forecast an entire *later year's effect* using only earlier calendar years. Score all 11 complete target years from 2015 through 2025 equally. This allows ten calendar years of preceding archive history before the first target. Historical DML uses the original grouped cross-fitting inside that history. The benchmark is the later year's unadjusted toss contrast. RMSE was **12.39 pp** for zero, **12.85** for full history, **12.32** for half-history, **12.53** for DML and **14.72** for the most recent year. Annual counts ranged from 63 to 959. The earliest targets are noisy, the training histories overlap, and the archive omits some matches. We used a single archived snapshot from 2026. The year cutoffs reconstruct match chronology. Historical release availability and completeness of all international fixtures in each year are unverified.

## GSS separates the model from the choice of historical waves

Predict each of the seven eligible waves after 2000 using the preceding eligible waves. All methods score the same later, design-weighted difference between questionnaire forms. Standardized OLS fits a survey-weighted linear model separately in each arm using the same age terms, sex, region, ballot and missing-value coding as the cross-fitted ridge estimator. It averages both predictions over the same eligible respondents. Every target wave receives equal weight when computing RMSE.

| Historical forecasting rule | RMSE across seven later waves (pp) |
|---|---:|
| Equal-weight mean of earlier design-based contrasts | 3.440 |
| Same waves, standardized OLS | 3.453 |
| Same waves, cross-fitted ridge AIPW | 3.467 |
| Latest observed design-based contrast | 3.605 |
| Mean of up to three most recent contrasts | 3.114 |
| Mean weighted by inverse historical survey variances | 3.382 |
| Half the equal-weight mean | 20.794 |
| Zero | 42.429 |

For 2024, full history predicted 43.07 points, OLS 43.14 and AIPW 43.19. The measured contrast was 36.82. Changing the historical window moved the forecast more than changing the adjustment, although even the three-wave rule predicted 41.57. The window and precision-weight rules are exploratory. There are seven waves, a single 2026 data vintage, and substantial overlap in the training histories. The four original prospective forecasts remain the reference for evaluating 2028. [Assignment, survey variance and 2028 target](../prospective/gss.md)

## JTPA shows where a new population enters

Using the corrected earnings decoder and the primary reporting lag, compare earlier and later adults with observed first-full-quarter earnings. Average the within-site offer-minus-control differences using each cohort's own observed site shares. The conditional SE weights the independent within-site Neyman variances by the squared shares. It conditions on the observed mix and does not account for missing wage records.

| Adult cohort | N | Pooled difference (SE), dollars | Site-standardized difference (SE), dollars |
|---|---:|---:|---:|
| Earlier observed applicants | 3,191 | +131.34 (48.65) | +137.94 (48.29) |
| 1989 Q1 holdout | 1,832 | −14.29 (61.77) | −12.69 (60.97) |

Earlier ridge AIPW gave +$138.45, within $0.51 of simple site adjustment. Of the later observed cohort, **428 of 1,832 adults (23.4%)** came from two site codes absent from the earlier observed sample. Reweighting earlier sites cannot estimate effects for these new sites without extrapolation. Calendar maturation, UI linkage and the original wage-code correction further qualify this comparison. For adult offers the design documents support the two-thirds treatment probability. The archived post-unblinding correction and the two reporting lags all refer to the same holdout. [JTPA design and original issuance](../rolling/jtpa.md)

From the repository root, reproduce the numerical output with the command below. It verifies the original data through the study loaders and writes JSON to standard output.

```bash
audit=docs/audits/2026-11-19_time_series_dml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python "$audit/adversarial/transport.py" \
  --cache-dir /tmp/dml-external-bench-cache
```
