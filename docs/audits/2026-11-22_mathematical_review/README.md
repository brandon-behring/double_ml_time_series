# Is the mathematics of the time-series DML sound?

**Two parts give wrong answers in settings that they are meant for.** Series-mode g-estimation
returned direct effects where the blips include the path through the state: its nominal 95%
intervals covered the lag-1 and lag-2 blips in **0 of 200** replications. The default forest, on a
linear design with an AR 0.98 covariate, had bias **+0.746** on a true effect of 2 and covered it
in **5.8%** of 120 replications. Most other findings are assumptions that are unstated, arguments
that do not support their conclusion, and checks that could not have detected either failure.

The scalar orthogonal score is not at fault. At n = 500, with ridge on these linear designs at unit scale, its
bias was below a tenth of its SD in every design whose conditioning set identified the effect. What
fails is what surrounds the score: the conditioning set, the learner, the folds, the variance and
the tests.

| ID | Layer | Finding | Severity | Issue |
|---|---|---|---|---|
| F6 | Estimand | One impact coefficient, with horizon, timing and future treatments unstated | P2 | [#55](https://github.com/brandon-behring/double_ml_time_series/issues/55) |
| F1 | Identification | Series g-estimation conditions on a post-treatment state | P1 | [#50](https://github.com/brandon-behring/double_ml_time_series/issues/50) |
| F2 | Identification, score | Neither the complete-dynamics regime nor the incomplete one is stated | P2 | [#52](https://github.com/brandon-behring/double_ml_time_series/issues/52) |
| F5 | Nuisance learning | Untuned, scale-dependent learners that cannot follow persistent covariates | P1 | [#51](https://github.com/brandon-behring/double_ml_time_series/issues/51) |
| F3 | Cross-fitting | Forward folds are defended by a forecasting argument | P2 | [#53](https://github.com/brandon-behring/double_ml_time_series/issues/53) |
| F4 | Inference | The default HAR interval under-covers on persistent scores | P2 | [#54](https://github.com/brandon-behring/double_ml_time_series/issues/54) |
| F7 | Inference | The variance theorem has no conditions, proof or citation | P3 | [#57](https://github.com/brandon-behring/double_ml_time_series/issues/57) |
| F8 | Validation | The gates cannot detect F1 to F5 | P2 | [#56](https://github.com/brandon-behring/double_ml_time_series/issues/56) |
| F9 | Validation | The sensitivity "Gamma" is a relabelled z-statistic | comment | [#44](https://github.com/brandon-behring/double_ml_time_series/issues/44#issuecomment-7082331244) |
| F10 | All | Smaller verified defects | P3 | [#58](https://github.com/brandon-behring/double_ml_time_series/issues/58) |

Conventions used below:

- Line numbers refer to `main` at `9a3aac4` unless marked "working tree". The studies ran on a
  working tree with uncommitted changes to `dml_ts`. A second run on a `git archive` of `9a3aac4`
  gave byte-identical draws and summary, so every simulated number below is also true of `main`.
- "Lag-0" is the interval from the variance without autocovariances. "Default" is the package's
  Bartlett interval with floor(n^(1/3)) lags and normal critical values.
- A coverage near 95% has a Monte Carlo SE of about 0.7 points at 1,000 replications, 0.5 at 2,000
  and 1.1 at 384. Every cell's own SE is in [summary.csv](summary.csv).
- What each cited theorem assumes, and the passage checked, is in [literature.md](literature.md).

## 1. Estimand (F6)

**`TemporalPLRDML` estimates one number, and the manuscript does not say which causal number it
is.**

- Chapter 5 defines the effect by an outcome equation (`chapters/chapter_05.tex:35-37`). Chapters
  5, 6 and 8 never mention potential outcomes. Chapter 6 does define panel blips and sequential
  exchangeability given history (`chapter_06.tex:444-446`), without a potential-outcome path.
- The manuscript has no match for "impulse", "local projection" or "cumulative effect". The
  estimator has no horizon.
- The timing of `X_t`, predetermined or contemporaneous, is an identifying assumption for an
  effect at horizon 0. It is not stated. Carlson and Shephard state it as triangularity.
- Four objects go unnamed: the response at horizon h with later treatments evolving, the blip with
  later treatments at baseline, the direct effect given a mediator, and the cumulative effect.
  Ballinari and Wehrli note that the first includes the treatments that the first one causes.
- The slope is a conditional-variance-weighted average over the scored dates (Lal and Chou,
  Proposition 1, for i.i.d. data). Dropping the first fold changes that population. The working
  tree's [HeartSteps pilot](../2026-11-19_time_series_dml/pilots/heartsteps.md) saw its estimate
  move from 0.138 to 0.024, much of it when the first 960 decisions left the sample.

**Scope.** This is a defect of statement. No simulation here bears on it.

**Design change.** Name the estimand with its horizon and its treatment of later assignments,
state the timing of `X_t`, and report the scored population. A `horizon=h` option (outcome
`Y_{t+h}`, gap at least h) follows from that statement.

## 2. Identification

### F1. Series g-estimation is not g-estimation

**Series mode residualizes every lagged treatment on the current state, which is post-treatment
for all of them.**

- The design is `Y_t` on `T_t, ..., T_{t-m+1}` given `X_t` (`dml_ts/dml/dynamic_g_estimation.py:493-510`).
- Chapter 6 warns that conditioning on a post-treatment state blocks part of the effect
  (`chapter_06.tex:441`), and then presents this mode's coefficients as blips (`:483`).
- Lewis and Syrgkanis's Algorithm 2 residualizes each treatment on the state at its own date, on
  disjoint blocks, with progressive fitting. The docstrings attribute the series mode to them
  (`dynamic_g_estimation.py:32-34`, `dml_ts/validation/dynamic_dgp.py:16-19`).
- The series DGP declares the direct coefficients to be the truth even under feedback
  (`dynamic_dgp.py:358`). Panel mode uses total blips (`:254-266`). Every series test turns
  feedback off, so the tests agree with the defect.

One series of 20,000 dates, three lags, linear, treatment-to-state feedback on:

| | Lag 0 | Lag 1 | Lag 2 | Cumulative |
|---|---:|---:|---:|---:|
| True blips | 1.00 | 1.00 | 0.50 | 2.50 |
| Truth declared by the DGP | 1.00 | 0.50 | 0.25 | 1.75 |
| Series mode | 0.990 | 0.489 | 0.250 | 1.729 |
| Recursion on each own-date state, in-sample | 0.985 | 0.999 | 0.499 | 2.484 |

3,000 dates, 200 replications, coverage of the true blips:

| Estimator | Lag 0 | Lag 1 | Lag 2 | SD by lag |
|---|---:|---:|---:|---|
| Series mode | 94.5% | **0.0%** | **0.0%** | 0.020 / 0.022 / 0.015 |
| Own-date, disjoint blocks, progressive | 93.5% | 95.5% | 93.5% | 0.056 / 0.095 / 0.117 |
| Own-date, overlapping blocks, HAC | 92.0% | 94.0% | 94.0% | 0.028 / 0.044 / 0.044 |
| Overlapping, with treatment history | 94.5% | 93.0% | 96.0% | 0.021 / 0.044 / 0.044 |

**Scope.** Without feedback, direct and total effects coincide, and series mode recovered them
(0.990, 0.494, 0.252). The
prototype in [checks.py](checks.py) follows Algorithm 2's state and recursion but differs from it
in two ways: it refits the nuisances in five progressive steps instead of at every block, and it
uses a joint sandwich instead of the block-diagonal variance. It shows that the target is
recoverable. It is not a validated implementation.

**Design change.** Implement the own-date recursion as `mode="series"`, sharing the panel
recursion, keep today's behaviour under a distributed-lag name, and make the DGP expose both
truths.

### F2. Dynamic completeness is the unstated assumption

Let `W_t` be the conditioning set, `e_t` the outcome error and `v_t` the treatment residual.
*Dynamic completeness* means `E[e_t | T_t, W_t, past] = 0` and `E[v_t | W_t, past] = 0`.

1. It is sufficient for a causal reading of the slope, given that the outcome equation is
   structural. Without it the slope is biased when treatment responds to past outcomes and errors
   persist.
2. It makes the scores `v_t e_t` a martingale difference sequence, so the lag-0 variance suffices.
   The condition on `e_t` alone is enough for this, as is the condition on `v_t` strengthened to
   hold given `e_t`. Without either, inference needs a long-run variance.
3. It justifies forward folds with no gap, by Lewis and Syrgkanis's martingale argument.
   This point is theory only.

The manuscript states neither regime. Chapter 5 gives no exogeneity, stationarity or mixing
condition, and `ASSUMPTIONS.md:24` cites "sequential exogeneity (chapter 5)" for text that is not
there. The defaults straddle the regimes: lags of `T` only (`temporal_plr_dml.py:205-215`), so the
lagged outcome is left to the user, with HAC as insurance. HAC can insure the variance. It cannot
insure against bias.

Ridge, n = 500, forward folds, 1,000 replications:

| Design | Controls | Bias | SD | Lag-0 | Default | Score lag-1 ACF |
|---|---|---:|---:|---:|---:|---:|
| Treatment responds to 0.5 Y(t-1), error AR 0.8 | current | **+0.451** | 0.059 | 0.0% | 0.0% | 0.42 |
| | + lag of Y | −0.004 | 0.065 | 94.8% | 94.9% | 0.01 |
| | + lag of Y, T, X | −0.002 | 0.050 | 94.5% | 94.0% | 0.00 |
| Same feedback, white errors | current | +0.002 | 0.032 | 94.2% | 93.8% | 0.00 |
| | + lag of Y, T, X | +0.001 | 0.050 | 94.9% | 94.0% | 0.00 |
| Both residuals AR 0.8 | current | −0.002 | 0.108 | 61.7% | 89.0% | 0.62 |
| | + lag of T (default controls) | −0.002 | 0.102 | 80.3% | 90.7% | 0.20 |
| | + lag of Y, T, X | −0.001 | 0.054 | 92.6% | 92.0% | 0.00 |
| Treatment residual AR 0.8 only | current | −0.001 | 0.030 | 92.7% | 92.5% | 0.00 |
| | + lag of Y, T, X | 0.000 | 0.050 | 94.5% | 93.7% | 0.00 |
| Outcome error AR 0.8 only | current | +0.005 | 0.088 | 93.6% | 93.5% | 0.00 |
| | + lag of Y, T, X | +0.001 | 0.052 | 93.4% | 92.6% | 0.00 |
| ARMA errors, both sides | current | +0.001 | 0.075 | 77.8% | 89.6% | 0.19 |
| | + lag of Y, T, X | 0.000 | 0.054 | 92.7% | 93.2% | 0.01 |
| Latent AR 0.8 confounder | current | **+0.419** | 0.054 | 0.0% | 0.0% | 0.08 |
| | + four lags of Y, T, X | **+0.284** | 0.050 | 0.0% | 0.0% | 0.00 |

**Scope.**

- Completing the lags is an identification device. Its effect on precision has either sign. It
  raised the SD by 66% when only the treatment residual persisted, because the lags absorb valid
  treatment variation. It lowered it by 41% when only the outcome error persisted, and by 50% when
  both did. The SD is proportional to sd(e)/sd(v), which predicts the first two signs.
- Scores are nearly uncorrelated when one residual is unpredictable given the past and the other
  residual, so the lag-0 variance can be right with only one side complete.
- **Uncorrelated scores do not show that confounding is absent.** The latent confounder with four
  lags had score autocorrelation 0.00 and bias +0.28.
- Lags did not remove a persistent latent confounder.

**Design change.** State both regimes. Let the user lag `Y`, `T` and `X`. Report lag-0 and HAR
errors side by side, with score autocorrelation labelled as a diagnostic of the dynamics only.

## 3. Score

**The partialled-out score behaves as the theory says. Orthogonality alone does not protect the
estimate from a poor learner.**

- With ridge on these linear designs at unit scale and an identifying conditioning set, |bias| was
  at most 0.087 SD at n = 500.
- Centring the final stage left the default forest's bias at +0.715, against +0.746. That bias is
  residual confounding. It is not a shift in the residual means.
- Orthogonality removes first-order nuisance error. The product of the two errors remains, and
  the rate condition has to hold for that product. The working tree's evaluation audit measured
  the product remainder at +0.067 against an oracle term of −0.004 in its tree/tree cell.
- The weights on marginal effects are convex when the treatment regression is the true
  `E[T|W]`. With linear controls and a nonlinear `E[T|W]` they can be negative (Kolesár and
  Plagborg-Møller, §3.3 and Proposition 7). This is where flexible learning earns its place, if it
  estimates that regression consistently.

**Design change.** None to the score. Sections 2, 4, 5 and 6 change its inputs and its variance.

## 4. Nuisance learning (F5)

**The flexibility that DML permits is not used correctly.**

- `dml_ts/dml` tunes nothing and scales nothing (`_utils.py:54-68`). Ridge (penalty 1) and lasso
  (penalty 0.1) act on raw units. The forest is fixed at depth 5. Series and panel g-estimation use
  an unlimited-depth forest under the same name (`dynamic_g_estimation.py:196-199`).
- Design `T = 0.5x + noise`, `Y = 2T + x + noise`, n = 500, 120 replications:

| | x AR 0.98: bias | Default coverage | x AR 0.5: bias | Default coverage |
|---|---:|---:|---:|---:|
| Default forest, forward | **+0.746** | **5.8%** | +0.018 | 90.0% |
| Same, centred final stage | +0.715 | 4.2% | +0.017 | 90.0% |
| Forest, two-sided, gap 5 | +0.369 | 35.0% | +0.011 | 95.0% |
| Ridge, forward | +0.000 | 93.3% | +0.000 | 95.0% |

- The package's R² warning fired in none of the 240 default fits.
- Units matter. With the same data and `x` multiplied by 0.01, ridge went from bias 0.000 and 95.0%
  coverage to **+0.477 and 0.0%**, and lasso from −0.009 and 95.0% to **+0.500 and 0.0%**.
- Chapter 5's listing reaches a treatment R² of 0.081 where 0.272 is attainable, and an outcome R²
  of 0.211 where 0.411 is attainable ([run.json](run.json)).

Candidate repairs, forward folds, n = 500, 60 replications, bias with default coverage in
parentheses:

| Design | Ridge, standardized | Default forest | Stack of ridge and forest | Ridge, then forest on residuals |
|---|---:|---:|---:|---:|
| A. Persistent linear covariate | −0.001 (98%) | +0.654 (8%) | −0.016 (83%) | **−0.005 (97%)** |
| B. Stationary, threshold interaction | +0.296 (0%) | +0.142 (37%) | +0.152 (28%) | **+0.066 (75%)** |
| C. Both | +0.295 (0%) | +0.779 (0%) | +0.297 (3%) | **+0.066 (73%)** |

The additive learner was the only one tested that kept the bias small in all three designs. The
stack did not help in C, where neither member captures both parts.

**Scope.** Coverage in this table has a Monte Carlo SE of up to 6 points. The additive learner
still under-covers in B and C, because the forest's remaining bias is 1.3 to 1.4 SD at n = 500. It
was not tuned, and these are three designs. With a persistent covariate a held-out block is a
covariate-shifted sample, which is why two-sided folds only halved the bias.

**Design change.** Standardize inside each fold, tune inside training folds, default to an
additive learner, and report out-of-fold loss per fold against a naive baseline.

## 5. Cross-fitting (F3)

**Chapter 5 defends forward folds with a forecasting argument. DML needs a different one.**

- The chapter appeals to data "that would be unavailable in a real forecasting scenario"
  (`chapter_05.tex:77`) and calls two-sided training "itself lookahead" (`:150-157`). The
  requirement is that nuisance error be nearly independent of the scores it enters, or that a
  martingale structure holds.
- Published schemes: progressive (Lewis and Syrgkanis), two-sided with neighbours left out or a
  buffer (Semenova et al., Ballinari and Wehrli, Gonçalves et al.), reverse (Ciganovic et al.), and
  none, under learner stability (Cao and Leung). `TemporalPLRDML` offers only forward schemes. Its
  default scores nothing in the first n/(K+1) rows and trains its first fold on that many.
- In the working tree, chapter 9 uses two-sided buffered folds, which contradicts chapter 5.

Lag-completed design, both residuals AR 0.8, ridge, 1,000 replications:

| n | Forward: lag-0 | SE/SD | Two-sided, gap 2: lag-0 | SE/SD | Change in SD |
|---:|---:|---:|---:|---:|---:|
| 120 | 82.5% | 0.72 | 92.4% | 0.94 | −35% |
| 200 | 87.7% | 0.80 | 93.5% | 0.94 | −24% |
| 500 | 93.6% | 0.92 | 95.5% | 1.00 | −16% |

The attempt to refute this used eight ridge designs at three sizes and seven tree cells:

| | Ridge, 24 cells | Trees, 7 cells |
|---|---|---|
| Two-sided SD below forward SD | 23 cells | 7 cells |
| Change in SD | +2.2% to −63.2% | −9.2% to −26.6% |
| Two-sided RMSE below forward RMSE | 23 cells | 5 cells |
| Change in lag-0 coverage | +34.8 to −4.9 points | 0.0 to −9.9 points |
| Second-half forward SD above two-sided | 22% to 44% | 16% to 42% |

- **Holds:** two-sided folds showed no look-ahead bias, including under outcome-to-treatment
  feedback: with ridge their |bias| exceeded the forward folds' by at most 0.003. With trees their bias matched that of forward folds scoring the second half, which
  train on at least n/2 rows. The two differed by at most 0.016.
- **Narrower:** the coverage gain is confined to the three lag-completed designs with AR 0.8 or
  0.95 (up to 34.8 points). In the other five designs coverage changed by +0.5 to −4.9 points. The gain mostly
  repairs the expanding window's small first folds: forward folds that score only the second half
  calibrate about as well (91.3% against 90.0% at n = 120), at a higher SD.
- **Reverses with untuned trees:** two-sided folds shrink the SD but not the learner's bias.
  Boosting at n = 400 covered 43.2% against 49.2% forward.
- No systematic benefit of a gap appeared, and at small n with persistence a gap cost coverage:
  forward gap 5 against gap 0 gave 59.9% against 73.6% at AR 0.95, n = 120.
- With K = 5, leaving neighbours out was the two-sided choice with the largest SD. With K = 10 it
  behaved like a two-row gap except at AR 0.95 (84.7% against 89.7% in the hardest cell at
  n = 200).

**Scope.** The gain depends on the design, and "no bias appeared" is not a theorem. Two-sided
training needs a mixing or stability condition that the martingale argument for forward training
does not use.

**Design change.** Fix the learner first. Then offer two-sided folds through the existing
`temporalcv.PurgedKFold(purge_gap=g, embargo_pct=0.0)`, keep forward as the default, give forward
schemes a minimum training size, and replace the forecasting argument.

## 6. Inference

### F4. The default interval under-covers

The default is Bartlett with floor(n^(1/3)) lags and normal critical values
(`temporal_plr_dml.py:553-570`). It is a variant of what Lazarus, Lewis, Stock and Watson call
"textbook NW". Coverage with current controls, ridge, forward folds, 2,000 replications:

| AR of v, e | n | Default | Prewhitened | Cosine, t | NW 1.3√n, fixed-b |
|---|---:|---:|---:|---:|---:|
| 0, 0 | 100 | 92.1% | 91.8% | 94.2% | 94.3% |
| | 500 | 94.2% | 94.2% | 94.5% | 94.7% |
| 0.95, 0 | 100 | 87.6% | 87.3% | 90.0% | 90.1% |
| | 500 | 91.4% | 91.4% | 92.4% | 92.7% |
| 0.5, 0.5 | 100 | 88.9% | 89.8% | 94.2% | 93.0% |
| | 500 | 92.5% | 93.7% | 94.0% | 94.2% |
| 0.8, 0.8 | 100 | 77.5% | 85.2% | 89.0% | 86.9% |
| | 500 | 87.3% | 92.3% | 93.0% | 92.8% |
| 0.95, 0.95 | 100 | 56.0% | 72.7% | 75.4% | 71.7% |
| | 500 | 64.1% | 85.0% | 82.0% | 83.0% |

- **Holds:** the cosine interval beat the default in all 15 cells with fitted nuisances and never exceeded nominal
  coverage (maximum 94.55%).
- **Cost:** on white scores it is 6% to 22% wider than the default. At AR 0.8 it is 21% to 44%
  wider.
- **Narrower:** at AR 0.95 it helps and does not rescue, even with oracle nuisances (73.8%, 78.8%
  and 84.3% at n = 100, 200 and 500). Completing the lags was the only tested remedy there, and it
  needs n: lag-0 coverage with forward folds was 73.6% at n = 120 and 88.0% at n = 500, and 95.1%
  with two-sided folds at n = 500.
- Prewhitening is already upstream as `newey_west_se(prewhiten=True)`, beat the cosine interval at
  AR 0.95 with n = 500, and is not exposed by `TemporalPLRDML`.

**Scope.** The fixed-b cubic was recalled from Kiefer and Vogelsang. Simulated critical values
agree with it to within 2% ([run.json](run.json)). The fixed-b column of the first-pass
study uses one more lag (B lags, b = (B+1)/n) than that of the refutation study above (S − 1 lags,
b = S/n).

**Design change.** Expose prewhitening, add the cosine estimator upstream, and report lag-0 and
HAR together. Switch the default once the book's numbers are regenerated.

### F7. The theorem as stated

- `chapter_05.tex:196-207` asserts root-n normality with a long-run variance "under regularity
  conditions". It gives no condition, proof or citation.
- `main.tex:63` promises "complete proofs". Chapter 5 has no proof environment.
- Line 194 ties score autocorrelation to autocorrelated outcomes. The scores can be white with
  autocorrelated outcomes, as F2 shows. The working tree has already corrected that sentence.
- Theorems now exist (Lewis and Syrgkanis Theorem 5, Ballinari and Wehrli, Ciganovic et al.,
  Gonçalves et al.). Of these only `lewisSyrgkanis2021` is in `bibliography.bib` (line 326).

**Design change.** State one theorem per regime, each with its conditions and its citation.

## 7. Validation

### F8. The gates cannot detect F1 to F5

- The only pass/fail coverage test of `TemporalPLRDML` uses i.i.d. data, 20 replications and a
  floor of 80% (`test/dml/test_temporal_plr_dml.py:979-1025`). A method with 70% true coverage
  passes it **23.8%** of the time. It is tier 3, outside the tier 1 and 2 gate.
- The series coverage test has no feedback, so it certifies F1
  (`test/dml/test_dynamic_g_estimation.py:196-221`).
- A search found no DGP with outcome-to-treatment feedback, no coverage test above AR 0.8, and
  no repeated-split study.
- Working tree only: `scripts/validate_falsification.py:207-226` (`check_controls`) accepts 86.4%
  HAC coverage, because it tests the gain over lag-0 and never the level.

**Design change.** Test coverage on dependent data with feedback and AR up to 0.95, at least 500
replications in the nightly lane, with a binomial acceptance band and SE/SD in [0.9, 1.1]. The
band is a proposed target and has not been calibrated.

### F9. The sensitivity statistic

`compute_sensitivity_for_dml` returns a critical Gamma equal to exp(|z| − 1.96), up to its grid
step of 0.1 and its cap of 3. It does not depend on the sample size
(`dml_ts/sensitivity/rosenbaum.py:277-335, 458-514`):

| z | Gamma, n = 200 | Gamma, n = 20,000 | exp(z − 1.96) |
|---:|---:|---:|---:|
| 2.2 | 1.3 | 1.3 | 1.271 |
| 2.5 | 1.8 | 1.8 | 1.716 |
| 2.8 | 2.4 | 2.4 | 2.316 |

`main` attributes the formula to Rosenbaum (2002), Eq. 4.14. The working tree already calls it
illustrative.

**Design change.** Use the omitted-variable-bias bound for the partially linear model,
bias² ≤ S²·C_Y²·C_D² (Chernozhukov et al., Theorem 1 and Corollary 1), benchmarked against
observed covariates.

## Smaller verified defects (F10)

- `chapter_07.tex:469` and its MDX mirror pass `hac_bandwidth="auto"`, label it "Andrews'
  optimal" and say it adapts to persistence. The constructor rejects the value
  (`temporal_plr_dml.py:338-345`), and the default rule depends on n alone. Chapter 5's parameter
  table also lists the default as `auto` (`chapter_05.tex:535`).
- `ASSUMPTIONS.md:30` calls that rule "data-driven".
- `blocked_cv` does not honour `gap` (`temporal_plr_dml.py:384`). Rows actually left out:

  | Requested gap | 0 to 19 | 20 | 50 |
  |---|---:|---:|---:|
  | n = 300 | 25 | 46 | 90 |
  | n = 600 | 50 | 92 | 185 |

- temporalcv truncates the quadratic-spectral kernel at the bandwidth. Its own docstring
  acknowledges that the result can fail to be positive semi-definite.
- `CLAUDE.md` says `venv/`. The locked `uv` environment is `.venv/`.

## Corrections to the review's own first findings

Rerunning the studies reproduced all 35 rows of the first pass and all 336 rows of the refutation
pass. It also showed that several summaries written from those rows were wrong. The text above
uses the corrected statements.

| First statement | What the rows show |
|---|---|
| Persistence on one side only: lag completion raises SD by 55–66% | +66% for the treatment residual, −41% for the outcome error. The +55% is feedback with white errors. |
| Gaps of 0 and 5 were indistinguishable in every design | At small n with persistence gap 5 was worse. |
| "Repo default lag-0 coverage 80.3%" | That is ridge with the default controls and a lag-0 interval. The default interval covered 90.7% there, and the default learner is a forest. |
| Two-sided SD 5–30% lower in all 12 designs | Lower in 23 of 24 ridge cells (+2.2% to −63.2%) and in all 7 tree cells. |
| Second-half forward costs 25–40% in SD | 22–44% with ridge, 16–42% with trees. |
| Cosine intervals are 4–15% wider | That compared them with 1.96 × the true SD. Against the default they are 6–22% wider on white scores. |
| Every serial design in the archives has SE/SD 0.73–0.88 | False. In the working tree's paired study the serial cells span 0.23 to 0.95 across learners. Only the tree learners stay within about 0.74 to 0.88. |
| None of the new theory is in `bibliography.bib` | Lewis and Syrgkanis is. |
| The repair table's forest used the default settings | It used 50 trees, as did its additive learner (bias −0.006, +0.062, +0.063; coverage 97%, 78%, 78%). The table above uses the default 100 for both. Both are archived. |
| Two-sided bias matched forward bias at equal training size | Only approximately: the random forest had −0.064 two-sided, +0.006 forward and −0.066 second-half forward. |

## Not verified

- `PanelDML` and `RollingWindowDML` were not audited beyond the earlier audits.
- `production/dml_pipeline.py` and the estimators in `validation/` are outside this review. Their
  learners differ from those of section 4: several are tuned by cross-validation. On `main` the
  pipeline's continuous-treatment SE omits the residual treatment, and with forward folds it scores
  rows that have no out-of-fold prediction. The
  [2026-11-19 audit](../2026-11-19_time_series_dml/02-code-and-methodology.md) found both defects,
  and the working tree fixes them. [checks.py](checks.py) does not test them.
- No proof in the cited papers was checked. The fixed-b cubic was not compared with its source.
- The own-date recursion is a prototype, tested on one linear design and one block length.
- The additive learner was neither tuned nor tested beyond three designs and one sample size.
- Every design here is linear-Gaussian or a small departure from it, with a homogeneous effect. No
  real series was re-analysed, and the book's own tables were not regenerated.
- The searches behind "no DGP" and "no match" statements are text searches.
- Numbers quoted from other audits in this working tree were read from their reports. They were
  not rerun, and those reports are not on `main`.
- Acceptance bands proposed for later work are targets.

## Upstream

- HAR estimators: [temporalcv#56](https://github.com/brandon-behring/temporalcv/issues/56)
- The contract for two-sided cross-fitting: [temporalcv#57](https://github.com/brandon-behring/temporalcv/issues/57)
- The `CrossFitCV` docstring: [temporalcv#58](https://github.com/brandon-behring/temporalcv/issues/58)

## Reproduce

From the repository root. A run takes about 10 to 15 minutes on 36 workers. The draws do not
depend on the number of workers. [Versions, hashes and deterministic checks](run.json) ·
[Every estimate](draws.csv.gz) · [Every cell](summary.csv) · [Driver](checks.py)

```bash
study=docs/audits/2026-11-22_mathematical_review
out=$(mktemp -d)
uv run --locked python "$study/checks.py" --workers 36 --output "$out"
cmp "$out/draws.csv.gz" "$study/draws.csv.gz"
cmp "$out/summary.csv" "$study/summary.csv"
.venv/bin/python -m pytest test/validation/test_mathematical_review_checks.py --no-cov -q
```
