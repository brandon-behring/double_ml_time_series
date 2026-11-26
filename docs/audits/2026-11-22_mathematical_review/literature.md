# Literature behind the mathematical review

**Each cross-fitting theorem read here (L1 to L5, L14) rests on a condition that the manuscript
does not state:** a martingale structure, weak dependence with a buffer or left-out neighbours, or
learner stability. The other works bear on the estimand, the learners, inference and sensitivity.

The table lists the 18 works, the version read and the key in
[`bib_ledger.yml`](../../research/bib_ledger.yml). The first 16 hexadecimal digits of each PDF's
SHA-256 identify the copy that was checked. "Checked" below means that the statement was read in
that copy's text. Proofs were not checked.

| # | Work | Version read | Ledger key | PDF SHA-256 |
|---|---|---|---|---|
| L1 | Lewis and Syrgkanis, dynamic g-estimation | [arXiv 2002.07285v4](https://arxiv.org/abs/2002.07285v4) (v5 exists) | `lewis2020dynamic` | `54c288368a3c77fd` |
| L2 | Semenova, Goldman, Chernozhukov and Taddy, dynamic panels | [arXiv 1712.09988v6](https://arxiv.org/abs/1712.09988v6) | `semenova2018orthoml` | `a1f9fd244450bef9` |
| L3 | Ballinari and Wehrli, DML impulse responses | [arXiv 2411.10009v2](https://arxiv.org/abs/2411.10009v2) | `ballinari2024dmlirf` | `96a1bb48c28a8b7d` |
| L4 | Ciganovic, D'Amario and Tancioni, reverse cross-fitting | [arXiv 2603.10999v2](https://arxiv.org/abs/2603.10999v2) | `ciganovic2026dmltimeseries` | `71ced91c9d7ca9d2` |
| L5 | Cao and Leung, neighbourhood stability | [arXiv 2511.10995v1](https://arxiv.org/abs/2511.10995v1) | `cao2025neighborhoodstability` | `d0fac9893e077ac4` |
| L6 | Kolesár and Plagborg-Møller, nonlinear world | [arXiv 2411.10415v4](https://arxiv.org/abs/2411.10415v4) | `kolesar2025nonlinearworld` | `ffe715f59795b5d4` |
| L7 | Chernozhukov et al., Long Story Short | [arXiv 2112.13398v5](https://arxiv.org/abs/2112.13398v5) (v6 exists) | `chernozhukov2026longstory` | `30ce0c6efcd7f67c` |
| L8 | Bach et al., tuning for DML | [arXiv 2402.04674v1](https://arxiv.org/abs/2402.04674v1) | `bach2024tuning` | `a33a633605d86409` |
| L9 | Rambachan and Shephard, common estimands | [arXiv 1903.01637v4](https://arxiv.org/abs/1903.01637v4) | `rambachan2025commonestimands` | `b7d9c404864caf79` |
| L10 | Carlson and Shephard, potential system | [arXiv 2603.20394v1](https://arxiv.org/abs/2603.20394v1) | `carlson2026potentialsystem` | `f493656c3be1292d` |
| L11 | Montiel Olea and Plagborg-Møller, lag augmentation | [arXiv 2007.13888v4](https://arxiv.org/abs/2007.13888v4) | `montielolea2021lpinference` | `90adc1d027a41c49` |
| L12 | Adamek, Smeekes and Wilms, high-dimensional projections | [arXiv 2209.03218v3](https://arxiv.org/abs/2209.03218v3) | `adamek2024hdlp` | `0ff860273e01f1ff` |
| L13 | Lazarus, Lewis, Stock and Watson, HAR inference | [authors' copy, JBES 2018](https://scholar.harvard.edu/files/stock/files/lazarus_lewis_stock_watson_har_inference_recommendations_for_practice_jbes_2018.pdf) | `lazarus2018har` | `8175998586eb7873` |
| L14 | Gonçalves et al., semiparametric local projections | [Dallas Fed WP 2616](https://doi.org/10.24149/wp2616) | `goncalves2026semiparametriclp` | `7013c9d8cb55fca1` |
| L15 | Ahrens, Hansen, Schaffer and Wiemann, stacking | [arXiv 2401.01645v2](https://arxiv.org/abs/2401.01645v2) | `ahrens2025modelaveraging` | `d1378ce94c2028b0` |
| L16 | Lal and Chou, representative effects | [arXiv 2506.07462v4](https://arxiv.org/abs/2506.07462v4) | `lal2026representative` | `9fa31494c34df648` |
| L17 | Schäper, residualized treatment intensity | [arXiv 2502.10301v1](https://arxiv.org/abs/2502.10301v1) | `schaper2025rols` | `b963e8b5dfd9c7ee` |
| L18 | Montiel Olea et al., double robustness | [arXiv 2405.09509v4](https://arxiv.org/abs/2405.09509v4) | `montielolea2026varithmetic` | `7ec95da773f9d912` |

## Dynamic effects and cross-fitting

### L1. Lewis and Syrgkanis

- **Assumes (Theorem 5):** the paper's partially linear Markov model (linear state and outcome
  equations, cited in the theorem as Equation 13), applied to one series under stationarity,
  additive treatment noise
  with covariance bounded away from zero, bounded variables and nuisances, nuisance-rate conditions
  evaluated on earlier blocks, and a bound (kappa) on how treatment noise propagates across the
  periods of a block.
- **Checked:** Algorithm 2, Theorem 5, and the paragraph that introduces them. The series is cut
  into disjoint blocks of m periods. Outcome and treatments are residualized on the state at the
  treated date. Nuisances for a block use earlier blocks only, and only the second half of the
  blocks is scored. Σ is block diagonal. The reason given for fitting forward is that future
  samples "would ruin the martingale structure".
- **Bears on:** F1, F2, F3. The repository's series mode residualizes every lag on the current
  state, so it is not this algorithm. The argument for forward folds is the martingale structure,
  which is a different argument from forecast availability.

### L2. Semenova, Goldman, Chernozhukov and Taddy

- **Assumes (Assumptions 4.1 to 4.5):** weak dependence over time with exponential mixing, an
  identification condition, sub-Gaussian tails and rates for the residual learners. The first-stage
  example (FS-TL) takes the treatment residual to be a martingale difference sequence.
- **Checked:** Definition 3.1 and its discussion, the statement of FS-TL, and the headings and
  opening lines of the Section 4 assumptions, which were not read in full. Neighbours-left-out folds train on the blocks that do not touch the test
  block. K ≥ 10 is recommended, so that at least 70% of the data trains the nuisances. Independence
  is approximated through Strassen's coupling.
- **Bears on:** F3. Two-sided training is a published scheme with a stated justification.

### L3. Ballinari and Wehrli

- **Assumes:** a binary treatment, conditional independence of treatment and potential outcomes
  with overlap (Assumption 2), residuals mean independent of the past (Assumption 5, for their
  first procedure), and dependence and rate conditions for one series (Assumptions 6 and 7).
- **Checked:** Assumptions 2, 5 and 6, Procedure 2, Theorem 4's construction, Section 4.1, and the
  paragraph on critical values. Procedure 2 removes k_T rows on each side of the test block. K = 10
  or 20 is recommended, with k_T at least the horizon and the lag depth, and floor(T/K) as a first
  choice. Fixed-bandwidth critical values are advised. The discussion of Assumption 2 says that
  when treatment causes later treatments, the identified effect includes those later treatments.
- **Bears on:** F2, F3, F4, F6.

### L4. Ciganovic, D'Amario and Tancioni

- **Assumes (Assumptions 2.1 to 2.5):** a finite long-run variance, a functional CLT for the oracle
  score, smoothness and orthogonality, nuisance error o(T^-1/4) with conditional stability given
  the training blocks, and a block construction. The construction of auxiliary samples uses time
  reversibility.
- **Checked:** Assumption 2.4, Remark 2.1 and Theorem 2.1. Remark 2.1 says the baseline deletes no
  neighbours, and that Assumption 2.4 is what permits this.
- **Bears on:** F3, F7. Two-sided folds without a gap are valid under a stability assumption. That
  assumption is imposed on the learner and has to be argued for each one.

### L5. Cao and Leung

- **Assumes (Assumptions 1 to 4):** regular moments, neighbourhood stability of the learner,
  learner consistency, and beta-mixing that decays fast relative to the neighbourhood radius.
- **Checked:** Assumptions 2 to 4 and Theorem 1. The theorem proves the stochastic equicontinuity
  step. It does not prove a CLT for the oracle score.
- **Bears on:** F3, F7. It is a route to validity without sample splitting, for stable learners.

## Estimands

### L6. Kolesár and Plagborg-Møller

- **Assumes (Proposition 7):** finite second moments, residual treatment variation, regularity of
  the outcome function, and either a correct treatment regression or a condition that restricts
  the outcome regression.
- **Checked:** the closing discussion of Section 3.3 and the statement of Proposition 7. With
  linear controls the weights on marginal effects can be negative when E[X|W] is nonlinear. They
  are positive when the treatment regression is the true conditional mean.
- **Bears on:** F5, F6. Flexible fitting of the treatment equation protects the interpretation of
  the slope, if it succeeds in estimating that conditional mean.

### L9. Rambachan and Shephard

- **Assumes:** a direct potential-outcome system. Causal readings need restrictions on how the
  assignment depends on the past, on other contemporaneous assignments, on future assignments and
  on potential outcomes.
- **Checked:** Theorems 1 and 2 and the discussion of Theorem 5. An impulse response equals an
  average effect plus a selection term. A sufficient condition for that term to vanish includes
  independence from future assignments.
- **Bears on:** F6. The treatment of future assignments is part of the estimand and has to be
  stated.

### L10. Carlson and Shephard

- **Assumes:** the potential system: a sequential assignment mechanism (DGP.1, DGP.2),
  non-anticipation and triangularity (CP.1, CP.2), and a link to observed data (LP).
- **Checked:** Definition 1 with DGP.1, DGP.2 and CP.1, and the opening statements of CP.2 and LP.
  Triangularity says that the features at
  date t do not respond to the assignment at date t.
- **Bears on:** F6. This is the timing condition on `X_t` that chapter 5 does not state.

### L16. Lal and Chou

- **Assumes (Proposition 1):** i.i.d. data, conditional ignorability, a smooth dose response, a
  finite third moment and consistent nuisances.
- **Checked:** Proposition 1. Residual-on-residual regression equals the average causal derivative
  for an affine response. Otherwise it differs by a curvature term and a conditional-variance
  weighting term.
- **Bears on:** F6.

### L17. Schäper

- **Assumes:** i.i.d. data, an outcome polynomial in treatment with arbitrary coefficients in the
  controls, treatment noise independent of the controls (Assumption 1), and moment restrictions on
  that noise (Assumption 2, met by Gaussian noise).
- **Checked:** Assumptions 1 and 2.
- **Bears on:** F6. An average partial effect needs conditions beyond those of the partially
  linear slope.

## Local projections and inference

### L11. Montiel Olea and Plagborg-Møller

- **Assumes (Assumptions 1 to 3):** a VAR with known lag length, innovations mean independent of
  past and future innovations, and all p lags of all variables as controls.
- **Checked:** the introduction's statement of the conditions and Assumption 1. Assumptions 2
  and 3 were not read in full. Under the stated conditions
  heteroskedasticity-robust errors suffice. The paper says that strengthening the martingale
  difference assumption is what avoids HAC inference.
- **Bears on:** F2. Lag completion buys the lag-0 variance only under a condition stronger than
  "enough lags".

### L12. Adamek, Smeekes and Wilms

- **Assumes (Assumptions A.1 to A.5):** near-epoch dependence, moments, weak sparsity, eigenvalue
  bounds and a penalty rate.
- **Checked:** the assumptions, A.5 only in part, and Theorem 2.1. Serially correlated errors are allowed and
  inference uses a long-run variance. The target coefficient is left unpenalized.
- **Bears on:** F2, F4, F6.

### L13. Lazarus, Lewis, Stock and Watson

- **Assumes:** second-order stationary regressor and error. The recommendations minimize a loss in size distortion and power,
  derived from Edgeworth expansions and fixed-b asymptotics.
- **Checked:** the rules and the rounding convention. "Textbook NW" is S = 0.75 T^(1/3) with normal
  critical values. The proposals are Newey-West with S = 1.3 T^(1/2) and fixed-b critical values
  (S rounded up), or the equal-weighted cosine estimator with nu = 0.4 T^(2/3) (rounded down) and
  t critical values.
- **Bears on:** F4. The repository's floor(n^(1/3)) rule is a variant of textbook NW with a
  different constant.

### L14. Gonçalves, Herrera, Kilian, Pesavento and Holban

- **Assumes (Assumptions 1 to 3):** stationary geometric beta-mixing, a bounded Riesz representer,
  moment bounds, consistency in L_q and a product rate o(T^-1/2).
- **Checked:** Assumptions 1 to 3, Theorem 5.1 and the description of cross-fitting. The scheme is
  neighbours-left-out with K ≥ 4. The limit variance is a long-run variance. The authors note that
  a bounded Riesz representer rules out Gaussian-tailed distributions.
- **Bears on:** F3, F6, F7.

### L18. Montiel Olea, Plagborg-Møller, Qian and Wolf

- **Assumes (Assumption 2.1):** a stationary model local to a structural VAR, with i.i.d. shocks in
  the main text (relaxed in Appendix C.1) and recursive identification.
- **Checked:** Assumption 2.1 and Proposition 3.1. Local projections are robust to a degree of
  dynamic misspecification that biases VARs.
- **Bears on:** F6.

## Learners and sensitivity

### L7. Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis

- **Assumes (Theorem 1):** square-integrable outcome and treatment, and positive residual treatment
  variance given observed and latent confounders.
- **Checked:** Theorem 1 and Corollary 1. bias² = ρ²B² ≤ B² = S²·C_Y²·C_D². S² is estimable.
  C_Y² is the share of residual outcome variation that the confounder explains, and C_D² is
  (1 − R²)/R² for the two Riesz representers, where 1 − R² is the share of residual treatment
  variation that the confounder explains.
- **Bears on:** F9. The numbering is that of v5. Version 6, published in *The Review of Economics
  and Statistics*, was not compared.

### L8. Bach, Schacht, Chernozhukov, Klaassen and Spindler

- **Assumes:** no theorem. It is a cross-sectional simulation study.
- **Checked:** the conclusion. Default learner settings can bias the effect estimate. Tuning on
  the full sample or on the folds beats splitting off a tuning sample. The authors ask that
  nuisance losses be reported.
- **Bears on:** F5.

### L15. Ahrens, Hansen, Schaffer and Wiemann

- **Assumes:** i.i.d. data.
- **Checked:** the abstract, which defines short-stacking and pooled stacking, and the opening of
  Section 3, which states the sampling assumption.
- **Bears on:** F5. Stacking over learners is established for i.i.d. data. In this review's design
  C it did not repair the bias, because neither member captured both the persistent linear part
  and the interaction.

## Not read

- Proofs, supplements (except where named above), and the empirical sections.
- Kiefer and Vogelsang's fixed-b cubic. Its coefficients were recalled, then compared with
  simulated critical values in [`run.json`](run.json).
- Newer versions of L1 (v5) and L7 (v6). The theorem numbers above refer to the versions in the
  table.
