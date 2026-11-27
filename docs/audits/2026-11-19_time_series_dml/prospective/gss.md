# Can past GSS surveys predict the effect of the questionnaire form in 2028?

Four forecasts were issued on **2026-11-21 at 02:18:33 UTC**, before the planned 2028 fieldwork. The target is the percentage-point effect of assigning the assistance form instead of the welfare form on recording that the country spends “too little.” All four rows await a comparable 2028 randomized experiment. NSF's solicitation provides evidence of a planned 2028 survey. Whether the item will run in the required form remains open. [Frozen forecasts](gss-frozen.json) · [Pending scores](gss-pending.json) · [NSF survey schedule](https://www.nsf.gov/funding/opportunities/general-social-survey-competition/505742/nsf24-537)

| Forecast | Effect, percentage points |
|---|---:|
| Zero | 0 |
| Historical mean | 42.287214013 |
| Half the historical mean | 21.1436070065 |
| DML | 42.441534778 |

The primary comparison is DML against half-history. A 2028 confidence interval wholly above **31.793 pp** favors DML under squared error. An interval wholly below that midpoint favors half-history. The two full-history forecasts differ by just **0.154 pp**. At historical levels of survey precision, a win against half-history would mainly support persistence of a large form effect. A win over unadjusted history on this one wave could depend on which side of their **42.364 pp** midpoint the effect falls. [Scoring and precision](scoring.md)

## What was randomized, and what changed?

The GSS assigns one of three standard ballots and one of two questionnaire forms to sampled respondents. On form X, NATFARE asks about spending on welfare. On form Y, NATFAREY asks about spending on assistance for the poor. Several other questions in the same sequence also change, including earlier items. Randomization identifies the **effect of assignment to the questionnaire form on the recorded response**. Attributing the contrast to a single word would require separately randomizing that wording while holding the preceding questions fixed. Inferring an effect on actual government spending would require another outcome. [NORC 2024 codebook, table 4 and appendix B](https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook.pdf) · [2024 field instrument](https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/GSS2024_Ballot2_English.pdf)

**The codebook label and the administered wording differ.** The cumulative file labels NATFAREY “Assistance to the poor.” The 2000 English questionnaire printed that wording. We checked an English cross-sectional questionnaire in each of 2004, 2006, 2008, 2014, 2021, and 2022 and all three 2024 English ballots. These print “Assistance for the poor.” The 2024 Spanish ballot B prints “Ayuda a los pobres” and “La asistencia pública o welfare.” In the sealed record the phrase “assistance to the poor” names this historical form and item family. The training estimates include both documented English variants. A separate forecast for a literal “to” wording effect would need a different design. [2000 questionnaire, V2](https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/2000%20GSS%20v2.pdf) · [2004 questionnaire, V5](https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/2004%20GSS%20V5.pdf) · [Historical instrument index](https://gss.norc.org/get-documentation/questionnaires.html) · [2024 Spanish ballot B](https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/GSS2024_Ballot2_Spanish.pdf)

This pre-outcome clarification on 2026-11-21 interprets the compatibility check for the named forms. Record the actual 2028 wording on standard ballots, cite the instrument, and verify the corresponding question stem, response categories, context, translations, and assignment across modes. Either attested English phrase on form Y is compatible with the historical family, conditional on the same form contrast. Form X must retain “Welfare.” A different phrase or a substantively changed sequence fails the check. Unclear evidence stays pending. The published release label alone cannot establish comparability. [Review template](gss-review-template.json) · [Frozen protocol](gss-frozen.json)

## Historical evidence and uncertainty

Use the regular cross-section and ballots A/B/C. Exclude 1,518 ballot D respondents in 2006 from the point estimate. Retain all sampled PSUs when computing domain variance. The history begins in 2000, the start of the assignment audit's modern review period. The five omitted waves are 2002, 2010, 2012, 2016 and 2018. NORC documented an assignment error linking questionnaire versions to demographics in those years. Current weights correct those imbalances, but excluding these waves avoids relying on that correction to support the known one-half randomization probability. NORC also reports some 2024 sex imbalances within particular ballot-form and household-size combinations. [NORC assignment audit](https://gss.norc.org/content/dam/gss/get-documentation/pdf/reports/methodological-reports/MR134%20-%20Ballot%20and%20Form.pdf) · [NORC codebook, appendix B](https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook.pdf)

Each arm's denominator is **all eligible cross-sectional respondents assigned that form**, including don't know, no answer and skipped web responses. These count as zero recorded endorsements. An inapplicable code is required on the alternate form's item. The substantive-answer sensitivity conditions on respondents who gave one of the three spending answers, a group that form assignment can change. WTSSPS applies to completed interviews. Selective survey completion could bias the arm comparison if form assignment changed participation. Applying the result to all sampled addresses or the adult population requires assumptions about nonresponse. Language coverage and modes have also changed across survey years. [NORC instrument and mode documentation](https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook.pdf)

| Wave | Eligible X / Y | Form effect (pp) | Design SE (pp) | Among substantive answers (pp) | Cross-fitted effect (pp) |
|---|---:|---:|---:|---:|---:|
| 2000 | 1,409 / 1,408 | 42.05 | 1.54 | 42.92 | 42.11 |
| 2004 | 1,401 / 1,411 | 46.30 | 1.71 | 46.79 | 46.52 |
| 2006 | 1,484 / 1,508 | 44.12 | 2.06 | 44.97 | 43.97 |
| 2008 | 1,004 / 1,019 | 44.28 | 2.18 | 44.80 | 44.43 |
| 2014 | 1,274 / 1,264 | 42.78 | 1.85 | 43.72 | 42.51 |
| 2021 | 1,987 / 2,045 | 39.04 | 1.78 | 39.03 | 39.39 |
| 2022 | 1,780 / 1,764 | 42.90 | 2.26 | 42.47 | 43.38 |
| 2024 | 1,683 / 1,626 | 36.82 | 2.13 | 37.07 | 37.23 |

An independent check comes from NORC’s spending report. Its published substantive-answer rates imply a 2008 contrast of 70.2 − 25.4 = **44.8 pp** and a 2014 contrast of 63.6 − 19.8 = **43.8 pp**. The current weighted file gives 44.804 and 43.715 pp on that denominator. The replay allows 0.15 pp for one-decimal published arm rates and differences across data and weight vintages. Including all assigned respondents as required by the frozen endpoint gives 44.283 and 42.780 pp. [NORC Social Change Report 63, tables K1 and K2](https://gss.norc.org/content/dam/gss/get-documentation/pdf/reports/social-change-reports/SC63%20GSS_Trends%20in%20Spending_1973-2018.pdf)

The design-based estimate subtracts the WTSSPS-weighted share in X from the weighted share in Y. Standard errors use Taylor linearization of both arm ratios with VSTRAT and VPSU, and 95% intervals use survey design degrees of freedom. The 2024 interval was **[32.57, 41.07] pp**. WTSSPS is available in every included wave. NORC recommends the additional nonresponse-adjusted WTSSNRPS wherever available. On the seven waves where both weights exist, substituting WTSSNRPS moves the historical point estimate by at most 0.21 pp. The 2021 design has nine public variance strata and 3,563 within-stratum PSU identifiers, so it should not be assigned the 2024 design degrees of freedom. [Wave estimates, counts, both weights and intervals](gss-history.csv) · [NORC weights and variance estimation](https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook.pdf)

The unadjusted forecast gives every historical wave equal weight. DML gives the same weight to each wave's augmented inverse-propensity-weighted estimate. Each wave uses the known assignment probability of one half. Five-fold cross-fitting holds whole variance PSUs out of the ridge fits for arm-specific endorsement. Predictors are age, age squared, sex, region, ballot and missingness indicators. Sample weights enter the arm fits and the final average. Cross-fitting uses only that historical wave's respondents. The separate forward test issues one population-effect forecast before 2028 assignments. It does not score predictions of individual respondents' answers. [Adapter](../external/forward/prospective_gss.py)

A chronological development replay predicts each of the seven waves after 2000 from earlier eligible waves alone. The root-mean-squared errors were **3.44 pp** for full history, **3.47 pp** for DML, and **20.79 pp** for half-history. The last case is instructive. Ahead of 2024, earlier waves implied 43.07 pp under full history and 43.19 pp under DML. The observed estimate fell to 36.82 pp. Both full-effect forecasts missed the decline. These are retrospective comparisons using one 2026 data vintage. The seven target estimates contain survey noise and expanding histories overlap. [Development predictions by wave](gss-history.csv)

Under the predeclared normal design calculation, distinguishing DML from half-history with 80% probability when either forecast is true requires a later SE of **3.80 pp** or smaller. The eight historical SEs were 1.54–2.26 pp. For DML against full history the corresponding SE is **0.028 pp**. These benchmarks depend on the unknown 2028 effect and design. If the effect lies close to a midpoint, substantially greater precision is necessary. After release, the scorer also reports whether 90% design intervals for individual forecast errors fit within the exploratory ±2, ±5 and ±10 pp margins. No operational accuracy threshold has been selected. [Scoring rule](scoring.md)

## Data, reproduction and delayed scoring

Davern, Michael, René Bautista, Jeremy Freese, Pamela Herd and Stephen L. Morgan (2026). *General Social Survey 1972–2024*, Release 3a, July 2026. Chicago, NORC at the University of Chicago. We downloaded the public cross-sectional Stata archive and used its member `GSS_stata/gss7224_r3a.dta`. The archive is 47,562,980 bytes, SHA-256 `d4fd4e5057d53d6023e55d5114685430321c3ab00ef2bc11099c2358efc65dfe`. NORC's download URL serves the current cumulative archive and may change. A later revision fails digest verification. Store the pinned original in the local download cache as `gss-1972-2024-3a.zip`. This repository contains aggregate results. [NORC download](https://gss.norc.org/get-the-data/stata.html) · [Data access and attribution terms](https://gss.norc.org/terms-and-conditions.html)

The SHA-256 of the frozen four-row record is `163640289ef6a76d3537e9240fd8389435df846fb500e11fbba30994ad5b831c`. Both the adapter and issuance receipt pin it. The timestamp is a local issuance record. From the repository root, replay the pinned training and print all scheduled rows with the following commands. [Issuance receipt](gss-issuance.json)

```bash
GSS_ADAPTER=docs/audits/2026-11-19_time_series_dml/external/forward/prospective_gss.py
.venv/bin/python "$GSS_ADAPTER" --cache-dir ~/.cache/dml-ts/external
```

Before looking at 2028 responses, fill a copy of the review template from official design and questionnaire documents. Record a reviewer, the exact English strings, language and form context, checks, official fieldwork and release dates, data release citation, archive SHA-256, and its Stata member if using a ZIP. The wave label must be 2028 and all fieldwork must follow issuance. Fieldwork may begin in the preceding calendar year. A documented failed check returns four unscored rows with the reason. A verified compatible assignment and positive sampling weights can supply a point estimate even when public variance codes are unavailable. Record the documented compatible design and withheld variance fields in the review. Missing variance strata/PSU identifiers or a singleton stratum disable interval-based rankings. Only a compatible released source with matching bytes and valid mutually exclusive item routing supplies the four scores. Keep the later report separately and retain the original issuance. [Review template](gss-review-template.json) · [Scorer](../external/forward/prospective_gss.py)

```bash
.venv/bin/python "$GSS_ADAPTER" --cache-dir ~/.cache/dml-ts/external \
  --review /path/to/completed-review.json --future-data /path/to/released-2028.zip
```
