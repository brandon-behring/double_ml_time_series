# Forecasting the effect of a future randomized assignment

**International cricket and the planned 2028 General Social Survey offer different tests of whether an earlier causal effect carries forward.** In cricket, the DML and half-history forecasts differ by just 0.073 percentage points. In the survey, they differ by 21.30 points, but DML differs from full history by only 0.15 points. A survey result could distinguish persistence from substantial shrinkage while leaving the value of covariate adjustment unresolved. Both future effects are pending.

## Forecasts issued before the assignments

All entries are additive effects in percentage points. The historical forecast uses the earlier assignment design. Half-history is exactly half of its stored value. DML uses cross-fitted outcome models and the known assignment probability. Each method receives the same later design-based estimate.

| Future assignment and outcome | Zero | Full history | Half-history | DML | Movement from full history |
|---|---:|---:|---:|---:|---|
| Eligible senior T20 international matches, December 2026 through November 2027. Effect of giving the focal team the choice of innings order on its final win credit. | 0 | +1.31669 | +0.65834 | +0.73160 | Toward zero |
| Conditional 2028 GSS questionnaire experiment. Effect of assigning the assistance form versus the welfare form on recording “too little” spending. | 0 | +42.28721 | +21.14361 | +42.44153 | Away from zero |

The cricket forecasts use 5,848 historical matches with an eligible toss. The frozen population excludes Afghanistan matches because of archive coverage. The first descriptive quarter ends February 28, 2027. The primary comparison uses the entire year ending November 30, 2027. The scorer waits for the last local match date to finish and for reconciliation against an independent fixture roster. The four quarterly estimates and annual estimate share matches. [Cricket design, coverage and scoring](cricket.md)

The survey forecasts give equal weight to eight eligible historical waves and use the survey weights within each wave. Other spending questions change with the assigned form, so the contrast includes questionnaire context. The 2000 English instrument prints “Assistance to the poor.” The seven later instruments print “Assistance for the poor.” A pre-outcome clarification records both observed variants and requires review of the 2028 questionnaire and assignment before opening responses. The denominator includes every eligible respondent assigned a form, with item nonresponse reported separately. [GSS design and scoring](gss.md) · [2000 questionnaire](https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/2000%20GSS%20v2.pdf) · [2024 questionnaire](https://gss.norc.org/content/dam/gss/get-documentation/pdf/quex/GSS2024_Ballot2_English.pdf)

## What each test can resolve

| Comparison | Effect at equal forecast loss | Future SE needed for 80% resolution if either forecast is correct | Historical benchmark SE |
|---|---:|---:|---:|
| Cricket, DML versus half-history | +0.695 pp | 0.0131 pp | 3.02 pp in 2025 |
| GSS, DML versus half-history | +31.793 pp | 3.801 pp | 1.54–2.26 pp across eight waves |
| GSS, DML versus full history | +42.364 pp | 0.0275 pp | 1.54–2.26 pp across eight waves |

The randomized assignment identifies the later effect under each study's design assumptions. Time ordering answers a different question. How closely does an earlier, fixed forecast predict that independently estimated effect? The primary comparison transforms the later effect's 95% interval into the difference in squared loss between DML and half-history. For example, if the entire 2028 survey interval lies above 31.793 points, it favors DML over half-history. If it crosses that midpoint, the ranking stays unresolved. Historical standard errors describe earlier experiments. Actual future precision and participation may differ. [Scoring rule and power calculation](scoring.md)

Each track has one fixed issuance. Inside the historical sample, cricket holds out entire series and the survey holds out sampling units within each wave when fitting outcome models. Those folds can use earlier and later records that were already available at issuance. The forward test starts when the forecast is frozen and evaluates the effect in new assignments. Cricket's quarters describe changes under the same forecast. The reminder replay separately uses `temporalcv` to refit at successive origins with a 98-day gap for outcome maturity. Evidence about whether DML can anticipate changing effects will require repeated prospective issuances with predictions that can change. [Rolling reminder design](../rolling/reminders.md)

## Evidence from studies already seen

As a check on the survey decoder, NORC's published 2008 and 2014 tables imply form differences of 44.8 and 43.8 points among respondents who gave substantive answers. Our reconstruction gives 44.80 and 43.72 points. Keeping item nonresponse in the denominator, as specified for 2028, yields 44.28 and 42.78 points. [NORC spending tables K1–K2](https://gss.norc.org/content/dam/gss/get-documentation/pdf/reports/social-change-reports/SC63%20GSS_Trends%20in%20Spending_1973-2018.pdf) · [Wave estimates and denominators](gss-history.csv)

Predicting each of seven later survey waves using only its preceding waves gave root mean squared errors of **3.44 points for full history**, **3.47 for DML** and **20.79 for half-history**. Both full history and DML predicted about 43 points for 2024, when the measured effect fell to 36.82. This replay uses one retrospective release. [Forecasts by wave](gss-history.csv)

In four evaluable reminder quarters, zero had the smallest mean squared error (0.001771), followed by half-history (0.003498), DML (0.005072) and full history (0.005968). The paired DML gain over half-history was −0.001574 log units squared, with a joint refit 95% interval of −0.008428 to +0.000748. In the corrected primary job-training comparison, half-history forecast +$65.67, DML forecast +$138.45, and the later randomized earnings difference was −$14.29 (SE $61.77). These studies informed the choice of comparator. They provide development evidence. [Reminder and job-training results](development.md)

The source search found candidates for follow-up in physical activity and employment. It identified no verified public, newly assigned cohort with the same endpoint and observation clock as either historical study. [Source ledger and eligibility decisions](source-ledger.md)

## Issuance and reproduction

| Record | Issued in UTC | SHA-256 |
|---|---|---|
| [Cricket annual forecast](cricket-annual-frozen.json) | 2026-11-21 02:17:47 | `e9c2acf2ef96df10c741fc440f18cf87fede061266e28a7dda76f51c29314ddc` |
| [Cricket registry, including the four quarter digests](cricket-issuance.json) | 2026-11-21 02:17:47 | `079a1bb1c38fcb31f4d2b50e2e381a99485b4702913d2abf8c051d6e30f22885` |
| [GSS forecast](gss-frozen.json) | 2026-11-21 02:18:33 | `163640289ef6a76d3537e9240fd8389435df846fb500e11fbba30994ad5b831c` |

These are local issuance records. The SHA-256 values identify the exact forecast files. They provide a reference for later verification. The timestamps use the issuing computer's UTC clock.

From the repository root, the following command reproduces both issuances and the development replay. Three study workers can run concurrently, with one numerical-library thread per worker. Copies of the verified cricket and GSS source ZIPs are also retained on the issuing workstation in the runner's default cache, `~/.cache/dml-ts/external/`. The providers update their download URLs. Exact raw-data replay requires the versions identified by the source digests.

```bash
audit=docs/audits/2026-11-19_time_series_dml
cache=/tmp/dml-external-bench-cache
uv run --locked python "$audit/external/run.py" --suite prospective --cache-dir "$cache" --jobs 3
```

All **119 source-inclusive and regression tests** passed with the locked temporalcv 2.2.0 environment and the adjacent 2.3.0 checkout. The 12 original forward study adapters and all three new adapters also passed in each environment. The prospective runner reproduced the saved numerical records. To repeat the test selection in Bash, use the following command. Prefix the Python invocation with `PYTHONPATH=../temporalcv/src` for the adjacent checkout.

```bash
export DML_TS_EXTERNAL_REPLAY=1 DML_TS_BENCHMARK_CACHE="$cache"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m pytest -q \
  test/integration/test_forward*.py \
  test/integration/test_{open_bandit,bracero,teacher_pay,weatherization}_benchmark.py \
  test/integration/test_{rolling_controls,rolling_reminders,jtpa_holdout,shrinkage_replay,prospective_cricket,prospective_gss}.py \
  test/validation/test_{effect_forecast,rolling_calibration,rolling_bootstrap_calibration}.py
```
