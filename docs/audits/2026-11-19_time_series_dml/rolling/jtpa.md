# Could earlier job-training offers predict the next cohort’s earnings effect?

**The later randomized offer effect was −$14.29 per adult**, with a standard error of $61.77 and a 95% interval of −$135.36 to +$106.79. Zero came closest on this cohort. The forecasts actually frozen before evaluation were +$424.69 for the earlier randomized difference and +$341.54 for DML. A source-coding error inflated both. After correcting that error, the retrospective forecasts were +$131.34 and +$138.45, respectively. The apparent advantage of DML over the earlier difference therefore changes sign. [Original study and archive](https://www.upjohn.org/data-tools/employment-research-data-center/national-jtpa-study) · [Sealed results](jtpa-results.json) · [Corrected results](jtpa-corrected-results.json)

## What was frozen

The target is the marginal intention-to-treat effect of **access to JTPA** on UI-covered earnings in the first full calendar quarter after adult random assignment. It includes assigned adults irrespective of receipt of training. The decision date was January 1, 1989. Adults randomized in January–March 1989 supplied a fresh evaluation cohort. The outcome period was April–June 1989. The training population and holdout each give an observed adult equal weight. [Protocol and source fields](jtpa-protocol.json)

The protocol file was written and its SHA-256 recorded before opening participant outcomes. We then read training wages only for mature earlier adults, wrote separate forecast records for the two reporting lags, and recorded their hashes before the first read of holdout wages. The archive’s original Abt UIW file uses calendar-quarter earnings. Zero is a recorded value when a complete state wage report contains no covered work for a person. The original documentation says most state responses lagged by one or two quarters and were sometimes revised. No contemporary 1989 operator snapshot survives in the materials examined. This exercise reconstructs an operator decision using later-archived files. Local hashes document the replay sequence and permit independent checks of file identity. [Source archive and documentation](https://www.upjohn.org/sites/default/files/2019-02/jtpa_national_evaluation.zip) · [Original issuance hashes and counts](jtpa-issuance.json)

The primary training cutoff admits assignments through June 30, 1988. Their next full quarter ended in September, leaving an additional completed quarter before January 1. With two quarters allowed for reporting, the latest training assignment was March 31, 1988. We froze zero, the earlier difference in mean earnings by assigned arm, and the earlier cross-fitted augmented inverse-probability estimate. The last uses the adult design probability of two-thirds and separate ridge regressions for each arm with site, adult gender and age. Five participant folds use only earlier adults. The later benchmark is the direct assignment-arm difference. [Analysis](../external/forward/jtpa_holdout.py) · [Primary forecast record](jtpa-lag1-frozen.json) · [Longer-lag forecast record](jtpa-lag2-frozen.json)

## What the first evaluation found

Forecasts and errors are nominal dollars per adult. A positive error means the forecast exceeded the later experimental estimate. Intervals condition on the particular forecast already issued and use independent-arm Neyman sampling variance in the later cohort.

| Reporting allowance | Forecast | Frozen value | Error | 95% interval for error |
|---|---|---:|---:|---:|
| Either lag | Zero | $0.00 | +$14.29 | [−$106.79, +$135.36] |
| One quarter | Historical | +$424.69 | +$438.97 | [+$317.90, +$560.05] |
| One quarter | DML | +$341.54 | +$355.83 | [+$234.75, +$476.90] |
| Two quarters | Historical | +$694.60 | +$708.88 | [+$587.81, +$829.96] |
| Two quarters | DML | +$524.62 | +$538.90 | [+$417.83, +$659.98] |

**A quarterly missing-data code explains much of the miss.** The initial decoder recognized Stata system missing values but treated a numeric 99,999 as wages. All 271 such rows across the source have 99 in the companion employment count and a missing quarterly flag. They occur in two sites. Of these people, 270 lack that quarter’s earnings in the later unedited UI file and one has a different reported amount. Exactly 25 affected adults entered both training sets, including 19 offers and six controls. The linked and eligible holdout contains zero such wages. The original zero forecast and later randomized estimate remain usable. The two fitted forecasts measure the performance of an implementation with this decoding error. [Source-field reconciliation](jtpa-source-audit.json) · [Preserved sealed scores](jtpa-results.json)

## What changes when the source marker is decoded

We classified the joint wage, employer-count and quarterly-flag pattern as unavailable. Its members leave the observed-wage comparison. Genuine recorded zeros stay in both arms. The correction follows inspection of the later comparison. Both frozen originals remain archived. The following results measure sensitivity to source coding on a previously opened holdout. [Corrected primary record](jtpa-corrected-lag1-frozen.json) · [Corrected longer-lag record](jtpa-corrected-lag2-frozen.json)

| Allowance | Forecast | Corrected value | Error | 95% interval for error | Paired squared-error gain over zero |
|---|---|---:|---:|---:|---:|
| One quarter | Historical | +$131.34 | +$145.62 | [+$24.55, +$266.70] | −21,002 USD² |
| One quarter | DML | +$138.45 | +$152.74 | [+$31.66, +$273.81] | −23,124 USD² |
| Two quarters | Historical | +$173.56 | +$187.85 | [+$66.77, +$308.92] | −35,083 USD² |
| Two quarters | DML | +$177.13 | +$191.41 | [+$70.34, +$312.49] | −36,435 USD² |

Under the primary lag, DML’s squared-error gain versus the historical forecast was **−2,122 USD²** after the correction, compared with **+66,086 USD²** for the original issuance. For the illustrative accuracy margins, none of the forecasts has a 90% error interval entirely inside ±$50 or ±$100. The zero forecast qualifies at ±$250. The corrected historical forecast also qualifies under the primary lag, with a 90% interval of +$44.01 to +$247.23. The corresponding DML interval extends to +$254.35. These sensitivities concern one cohort. A decision about delivering the program would require costs and a relevant benefit horizon. [Complete scores and tolerance indicators](jtpa-corrected-results.json)

## Coverage and comparability

| Cohort | All adult assignments | Assigned in covered sites | Valid linked records | Observed quarterly earnings, corrected | Contributing sites |
|---|---:|---:|---:|---:|---:|
| Earlier, one-quarter reporting lag | 3,965 | 3,253 | 3,216 | 3,191 | 10 |
| Earlier, two-quarter reporting lag | 2,234 | 1,858 | 1,829 | 1,804 | 7 |
| Later, 1989Q1 assignment | 2,655 | 1,873 | 1,834 | 1,832 | 9 |

The later observed-earnings sample comprises 1,237 offered adults and 595 controls. The archive recorded zero covered wages for 444 and 194 respectively. In covered sites, 21 of 1,259 assigned offers and 18 of 614 controls failed linkage validation, and one further wage was unavailable in each arm. The contrast for the full eligible population depends on whether this ascertainment preserves exchangeability. Two later sites (MN and NE) contributed 428 of the 1,832 evaluated adults and had no qualifying observations under the primary training cutoff. The holdout combines a change in assignment date and a change in site composition. A single cohort cannot determine which explains a forecast error or whether an historical training effect declined. [Counts by assignment and site](jtpa-corrected-results.json)

## Reproduce the replay

The command below verifies the source SHA-256, reconstructs both original forecasts in a temporary directory, compares their digests to the published issuance record, scores the holdout, and repeats the corrected analysis. The source archive is 128 MB and contains the original quarterly data, baseline fields and later re-release used for the marker check. Tests perturb later outcomes and verify that a digest mismatch stops scoring before holdout wages are read. [Integration tests](../../../../test/integration/test_jtpa_holdout.py)

```bash
audit=docs/audits/2026-11-19_time_series_dml
uv run --locked python "$audit/external/forward/jtpa_holdout.py" --cache-dir /tmp/dml-external-bench-cache
```
