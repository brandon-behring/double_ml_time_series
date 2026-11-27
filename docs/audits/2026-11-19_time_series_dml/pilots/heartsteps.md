# HeartSteps V1 · repeated randomized prompts

**The public CSV reproduces the published main effect.** For an available participant, sending an activity suggestion increased average log-transformed 30-minute step count by 0.1312, with a 95% interval of −0.0058 to 0.2682. The point estimate, standard error, interval and p-value agree with the authors’ reported results to their published precision. The first week matters for temporal validation. The corresponding benchmark is much smaller on the decisions that remain after a one-week warm-up. [Trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC6401341/) · [Primary analysis supplement, Table 1](https://people.seas.harvard.edu/~samurphy/papers/MRTSupplement.pdf)

## Design and reconstruction

Participants could receive a notification at five scheduled times per day for six weeks. Conditional on availability, probabilities were 0.4 for no suggestion, 0.3 for a walking suggestion and 0.3 for an anti-sedentary suggestion. The primary contrast pools both kinds of suggestion and averages over eligible occasions. Availability depends on current activity, snoozing and connectivity, and can change with treatment history. The outcome was log(Jawbone steps in the next 30 minutes + 0.5), adjusted for the same transformation of steps in the preceding 30 minutes. The paper assigned zero steps to minutes without a Jawbone record. The reconstructed sample has 552 available decisions with missing raw post-window step counts. [Release and CC BY 4.0 license](https://github.com/klasnja/HeartStepsV1) · [Data dictionary](https://github.com/klasnja/HeartStepsV1/wiki/Documentation-for-suggestions.csv) · [Missing-data sensitivity analysis](https://people.seas.harvard.edu/~samurphy/papers/MRTSupplement.pdf)

The release includes 37 analyzed participants from 44 enrolled, with 8,274 decision records and 6,590 raw available records. To handle enrollment days that started after slot one, the runner reconstructs study day from each person’s first scheduled slot and the travel-adjusted decision index. It excludes 390 travel decisions, 340 after study day 41, three unavailable decisions with a message sent and one sent message with missing type. This leaves 7,540 decisions and **6,061 available decisions**. The authors’ model uses recorded delivery (`send`). Six retained decisions show delivery despite a negative recorded draw (`is.randomized`). The release establishes this discrepancy and leaves the reason unresolved. [Authors’ preprocessing and model](https://github.com/tqian/paper_mrt_PsychMethods/blob/main/analysis.R) · [Study team’s main effects code](https://github.com/StatisticalReinforcementLearningLab/HeartstepsV1Code/blob/master/Main%20Effects%20Analysis.R)

## Results

| Analysis | Eligible decisions scored | Log effect | SE | 95% interval |
|---|---:|---:|---:|---:|
| Published centered regression, recorded delivery | 6,061 | 0.13120 | 0.06739 | [−0.00576, 0.26815] |
| Same regression, recorded assignment draw | 6,061 | 0.13713 | 0.06745 | [0.00006, 0.27420] |
| Known-propensity orthogonal score, five participant folds | 6,061 | 0.13768 | 0.06560 | [0.00465, 0.27072] |
| Centered regression, recorded draw, days 7–41 | 5,101 | 0.03805 | 0.07185 | [−0.10796, 0.18406] |
| Known-propensity orthogonal score, expanding study-week folds | 5,101 | 0.02398 | 0.07212 | [−0.12228, 0.17024] |

The centered regressions use the Mancl–DeRouen participant-level correction and the authors’ 34 degrees of freedom. We ran their covariance and testing code with the public-data regression and also calculated the correction independently. Both reproduce the published p-value of 0.0599. The authors’ original end-to-end script expects a team-only R workspace. [Original inference implementation](https://github.com/tqian/paper_mrt_PsychMethods/blob/main/xgeepack.R) · [Public release provenance](https://raw.githubusercontent.com/klasnja/HeartStepsV1/main/README.md)

For the two orthogonal-score rows, the runner uses the study probability 0.6 and regresses outcome residuals on the centered assignment draw. The nuisance model uses a quadratic in log pre-decision steps, slot and study day. The first split holds out entire participants. The second trains on earlier study days and tests consecutive seven-day blocks beginning on day 7, with a one-day gap. Enrollment dates differ, so its time axis is days since enrollment. Both report participant-clustered score intervals with a finite-cluster multiplier and a t critical value on 36 degrees of freedom. These pilot-specific scores use base R so they can honor the trial design. They target a proximal excursion effect among available participants. The repository’s recursive estimator uses a terminal outcome and period-specific treatment effects.

The strongest comparison is between the final two rows, which use exactly the same decisions. Much of the apparent change from 0.13768 to 0.02398 occurs when the first 960 available decisions leave the sample. The original trial also reports a declining effect over time. A comparison that simultaneously changes the scored dates and the estimator cannot tell how much of the shift the estimator caused. Given 37 participants, the six delivery discrepancies and the intervals near zero, this pilot supports a replication and a split-sensitivity result. A single trial cannot establish frequentist CI coverage. [Original trial results](https://pmc.ncbi.nlm.nih.gov/articles/PMC6401341/)

## Reproduce

From the repository root, use R with the `sandwich` package installed for the original-code cross-check. The reconstruction and both orthogonal-score analyses require only base R if the second input is omitted. The verified run used R 4.5.2. [Runner](heartsteps.R)

```bash
curl -fL 'https://raw.githubusercontent.com/klasnja/HeartStepsV1/main/data_files/suggestions.csv' -o /tmp/heartsteps_audit_suggestions.csv
curl -fL 'https://raw.githubusercontent.com/tqian/paper_mrt_PsychMethods/main/xgeepack.R' -o /tmp/heartsteps_audit_xgeepack.R
sha256sum /tmp/heartsteps_audit_suggestions.csv /tmp/heartsteps_audit_xgeepack.R
Rscript docs/audits/2026-11-19_time_series_dml/pilots/heartsteps.R /tmp/heartsteps_audit_suggestions.csv /tmp/heartsteps_audit_xgeepack.R
```

| Input | SHA-256 verified on November 19, 2026 |
|---|---|
| Public suggestions CSV | `60ee7896183d7084a084f9a6e8ef2d0afea2e7fecdb41d6b15382ce1745c5ddb` |
| Authors’ R inference source | `301f7bc5a2799c470274c0730b8220d42a8479064a2bc20668da8b2ef71d5d5b` |
