# Implementation Strategy: Time Series Double Machine Learning

**Date**: January 8, 2026
**Author**: Gemini (Acting as Lead Architect)
**Status**: Adopted & In Progress
**Revision**: 5.0 (Integrated & Refined)

---

## 1. Executive Summary

This report outlines the corrected implementation strategy for the **Double Machine Learning (DML) for Time Series** project. Following a critical audit which identified a significant gap—the reliance on i.i.d. estimators for time series problems—we have pivoted to a **Dynamic DML** framework based on the work of Lewis & Syrgkanis (2021).

The core engine has been successfully ported from the `causal_inference_mastery` repository, providing us with:
1.  **Sequential g-estimation** for dynamic treatment effects.
2.  **Time-series aware cross-fitting** (Blocked and Rolling Origin).
3.  **HAC-robust inference** (Newey-West standard errors), aligned with our internal `TemporalValidation.jl` standards.

We confirm that the project's goal is **theoretically sound and practically achievable**. However, this feasibility relies on strict adherence to time-series specific assumptions.

---

## 2. Methodological Framework

### 2.1 The Core Problem: Dynamic Confounding
Standard DML (Chernozhukov et al., 2018) assumes observations are independent and identically distributed (i.i.d.). In time series, this fails because:
*   **Autocorrelation**: $X_t$ depends on $X_{t-1}$.
*   **Feedback Loops**: Past outcomes $Y_{t-1}$ can affect future treatments $T_t$.
*   **Dynamic Effects**: Treatment $T_t$ affects not just $Y_t$ (contemporaneous), but also $Y_{t+1}, Y_{t+2}, \dots$ (lagged effects).

### 2.2 The Solution: Dynamic DML via g-Estimation
We adopt the **Dynamic DML** estimator proposed by Lewis & Syrgkanis (2021). This approach combines the "peeling" logic of sequential g-estimation with the bias-reduction of DML.

#### Algorithm: Sequential g-Estimation
To estimate the effect of treatment $T$ on outcome $Y$ at lag $h$, we proceed sequentially from the largest lag $H$ down to 0:

1.  **Peel Future Effects**: Adjusted outcome $\tilde{Y}_t = Y_t - \sum_{k=h+1}^H \hat{\theta}_k T_{t-k}$.
2.  **Nuisance Estimation**:
    *   Outcome model: $q(X_t) = \mathbb{E}[\tilde{Y}_t \mid X_t]$
    *   Propensity model: $p(X_t) = \mathbb{E}[T_t \mid X_t]$
3.  **Orthogonal Estimation**: Solve the Neyman-orthogonal moment condition:
    $$ \mathbb{E}[(\tilde{Y}_t - \theta_h T_t - q(X_t))(T_t - p(X_t))] = 0 $$
    This yields the closed-form estimator:
    $$ \hat{\theta}_h = \frac{\frac{1}{n} \sum (\tilde{Y}_t - \hat{q}(X_t))(T_t - \hat{p}(X_t))}{\frac{1}{n} \sum (T_t - \hat{p}(X_t))^2} $$

### 2.3 Valid Inference
Standard errors must account for serial correlation. We implement **HAC (Heteroskedasticity and Autocorrelation Consistent)** covariance estimation using the **Newey-West (1987)** estimator with a Bartlett kernel.

*Internal Verification*: Our HAC implementation uses the same Bartlett kernel logic found in `TemporalValidation/src/statistical_tests/hac.jl`, ensuring consistency across our Python and Julia ecosystems.

### 2.4 Systematic Review of Critical Assumptions
*This section makes explicit the conditions required for validity.*

#### A. Stationarity (Critical)
**The Assumption**: The data generating process (DGP) is covariance stationary.
**The Risk**: Unit roots invalidate the asymptotic normality of the DML estimator.
**Mitigation**: **Pre-estimation Differencing**. We must apply ADF tests. If data is I(1), we operate on differences $\Delta Y_t, \Delta T_t$. *Note: We explicitly reject 2024-2025 "Adaptive Forecasting" methods for this task, as they violate the stable parameter assumption required for causal inference.*

#### B. Sequential Exogeneity (No Anticipation)
**The Assumption**: Agents do not act on information about *future* treatments that is not contained in current controls $X_t$.
**The Risk**: "Anticipatory pricing" (competitors lowering prices *before* a known shock) breaks identification.
**Mitigation**: **Forward-Looking Controls**. Include futures prices, expert forecasts, or sentiment indices in $X_t$ to "block" the back-door path of anticipation.

#### C. Time Series Overlap (Positivity)
**The Assumption**: $0 < P(T_t=1 \mid X_t) < 1$.
**The Risk**: Deterministic policy rules (e.g., algorithmic pricing) lead to zero overlap.
**Mitigation**: **Continuous Treatment DML**. Move from binary $T \in \{0,1\}$ to continuous $T \in \mathbb{R}$ (prices). The overlap assumption relaxes to the existence of residual variance in $T$ after controlling for $X$.

#### D. Lag Truncation Bias
**The Assumption**: Effects beyond lag $H$ are zero.
**The Risk**: Omitted variable bias from long-memory effects.
**Mitigation**: **Sensitivity Analysis** and **Causal Discovery** (see Phase 2).

#### E. Effective Sample Size
**The Assumption**: Sufficient information content in $N$ observations.
**The Risk**: High autocorrelation ($\rho \approx 1$) implies $N_{eff} \ll N$.
**Mitigation**: **Ridge/ElasticNet Nuisance Models**. For small $N_{eff}$, avoid complex Random Forests; prefer linear models with high regularization.

### 2.5 Advanced Mitigation Strategies
*Based on further research into the "frontier" of Time Series DML.*

#### 1. Handling Non-Stationarity without Over-Differencing
While differencing solves unit roots, it removes long-run information. An advanced option is **Fractional Differentiation** (finding $d \in (0, 1)$ such that $(1-L)^d Y_t$ is stationary). This preserves more signal than integer differencing ($d=1$).
*   *Action*: Add `fracdiff` utility to `src/data` for advanced preprocessing.

#### 2. Instrumental Variable DML (IV-DML) for Anticipation
If "No Anticipation" is strictly violated, we can pivot to **IV-DML**.
*   *Method*: Identify a shock $Z_t$ (e.g., regulatory announcement dates, weather shocks) that affects $T_t$ but is unconnected to anticipation.
*   *Implementation*: Use `DoubleMLIV` structure but with time-series cross-fitting.
*   *Action*: Reserve this as a fallback if the standard Dynamic DML fails placebo tests.

#### 3. Continuous Treatment via "DoubleMLPLR"
For the pricing application, treating price as binary ("High" vs "Low") is a crude approximation that wastes information.
*   *Method*: Use **Partially Linear Regression (PLR)** DML.
    $$ Y_t = \theta T_t + g(X_t) + \epsilon_t $$
    $$ T_t = m(X_t) + \eta_t $$
    $$ \hat{\theta} = (\tilde{T}'\tilde{T})^{-1} \tilde{T}'\tilde{Y} $$
*   *Advantage*: Solves the "Overlap" issue by only requiring that price *variance* is not zero given $X_t$.
*   *Action*: Implement `ContinuousDynamicDML` in Phase 3.

### 2.6 Validation Guardrails (Added 2026-01-08)
*Inspired by `TemporalValidation/src/gates/suspicious.jl`.*

To prevent silent failures (where the code runs but the result is garbage), we enforce the following **Failure Gates**:
1.  **The "Too Good To Be True" Gate**: If the Propensity Score model ($p(X_t)$) has an accuracy/AUC > 0.95 (or beats a persistence baseline by >20%), we **HALT**.
    *   *Reason*: This indicates a violation of **Overlap**. The treatment is deterministic given history.
2.  **The "Persistence" Gate**: If the treatment $T_t$ has extremely low variance ($\text{Var}(\Delta T_t) \approx 0$), we **WARN**.
    *   *Reason*: DML requires "identifying variation". If prices never change, we can't estimate elasticity.

### 2.7 Sensitivity Analysis
*Adapted from `causal_inference_mastery/src/causal_inference/sensitivity/rosenbaum.py`.*

Dynamic DML assumes **Sequential Exogeneity**. We quantify robustness to violations using **Rosenbaum Bounds** (or E-values for continuous outcomes).
*   **Metric**: $\Gamma_{critical}$. The factor by which unobserved confounding would need to shift treatment odds to explain away the result.
*   **Threshold**: If $\Gamma_{critical} < 1.5$, the result is "Sensitive" and should be treated with caution.

---

## 3. Implementation Plan

### Phase 1: Foundation & Validation (Current Status: 90% Complete)
*   **Goal**: Establish a trustworthy Dynamic DML engine.
*   **Actions Taken**:
    *   ✅ **Port Engine**: `src/dml/dynamic_dml.py` implemented.
    *   ✅ **Integrate Splitter**: `src/dml/cross_fitting.py` now wraps the internal `temporalcv` library (PurgedWalkForward), replacing manual ports.
    *   ✅ **Fix DGP**: `src/validation/dgp_generator.py` updated to `TimeSeriesDGP`.
    *   ✅ **Update Validator**: `BiasValidation` refactored to support Dynamic DML.
*   **Remaining Steps**:
    *   Run the "7-Method Validation Suite".
    *   Add Stationarity Tests (ADF/KPSS) to the pipeline.

### Phase 2: Advanced Time Series (Q1 2026)
*   **Goal**: Robustness & Real-World Constraints.
*   **Key Tasks**:
    *   **Continuous DML**: Implement `ContinuousDynamicDML` for pricing.
    *   **Stationarity Handler**: Build an auto-differencing pipeline.
    *   **Reference Benchmark**: Align `src/dml` against the official implementation `syrgkanislab/dynamicdml`.
    *   **Causal Discovery Integration**: Integrate **PCMCI** logic (from `causal_inference_mastery`) to discover the optimal lag structure $H$ before estimation.

### Phase 3: Applications (Q2 2026)
*   **Goal**: Production-ready pricing models.
*   **Key Tasks**:
    *   **FRED Integration**: Build data loaders for macro indicators.
    *   **Competitor Pricing Model**: Estimate price elasticity using Continuous Dynamic DML.
    *   **Data Strategy**: See `docs/DATA_STRATEGY_REPORT.md` for the tiered approach (Synthetic -> OJ Benchmark -> MyGA Production).

---

## 4. References & Benchmarks

### 4.1 Academic Literature
1.  **Lewis, G., & Syrgkanis, V. (2021)**. Double/Debiased Machine Learning for Dynamic Treatment Effects via g-Estimation. *arXiv preprint arXiv:2002.07285*.
2.  **Chernozhukov, V., et al. (2018)**. Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.
3.  **Newey, W. K., & West, K. D. (1987)**. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*.
4.  **Bergmeir, C., & Benítez, J. M. (2012)**. On the use of cross-validation for time series predictor evaluation. *Information Sciences*.

### 4.2 Open Source Benchmarks
1.  **Official Implementation**: `syrgkanislab/dynamicdml` (GitHub)
    *   *Role*: Ground truth for `blip.py` (blip functions) and `snmm.py`.
2.  **DoubleML**: `DoubleML` (Python/R)
    *   *Role*: Benchmark for standard PLR/IVM models (Continuous Treatment).
3.  **EconML**: `microsoft/EconML`
    *   *Role*: Benchmark for Orthogonal Random Forests and CATE estimation.

---

## 5. Conclusion

We have successfully navigated the theoretical "valley of death" by adopting the correct framework (Dynamic DML) and explicitly addressing its assumptions. By moving to **Continuous Treatment DML**, enforcing **Stationarity**, and integrating **Causal Discovery**, we ensure the project's academic rigor and practical utility.

## Appendix A: Why Dynamic DML? (Method Selection Defense)

We chose **Dynamic DML (Lewis & Syrgkanis, 2021)** over competitors for three specific reasons critical to the Pricing use case:

1.  **The "Peeling" Property**:
    *   *Competitor*: Standard DML estimates $\theta_0$ by treating lagged treatments as fixed controls.
    *   *Dynamic DML*: Explicitly models the path dependency. It recognizes that $Y_t$ affects $T_{t+1}$ (feedback). By solving for $\theta_H$ first and peeling backward, it correctly isolates the "surprise" innovation at each step.
    *   *Benefit*: Correctly disentangles "Stockpiling" (short-term positive, medium-term negative) from "Brand Damage" (long-term negative).

2.  **High-Dimensional Controls**:
    *   *Competitor*: VAR/VECM requires $N > P^2$.
    *   *Dynamic DML*: Handles $P \gg N$ via Lasso/Random Forest nuisance models. This is essential when controlling for hundreds of macro indicators (FRED) and competitor prices.

3.  **Inference Validity**:
    *   *Competitor*: Deep Learning (LSTM/Transformer) provides superior forecasts but lacks valid confidence intervals.
    *   *Dynamic DML*: Provides $\sqrt{N}$-consistent, asymptotically normal estimators, allowing for rigorous hypothesis testing ($H_0: \text{Elasticity} = -1$).

