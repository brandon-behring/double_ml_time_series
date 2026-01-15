# Response to Skeptical Critique: Remediation Strategy

**Date**: January 8, 2026
**Author**: Gemini (Lead Architect)
**Status**: Remediation Plan
**Reference**: `docs/SKEPTICAL_CRITIQUE_2026-01-08.md`

---

## 1. Executive Summary

We welcome the skeptical audit and acknowledge the significant risks identified, particularly regarding **Stationarity** (Unit Roots) and **Linearity** (Pricing Convexity).

Our remediation strategy focuses on **Robustness over Complexity**. Instead of inventing new "Cointegration DML" methods (which are unproven), we will enforce strict data engineering standards to ensure the data fits the valid domain of Dynamic DML.

---

## 2. Remediation of Core Risks

### 2.1 The "Stationarity" Remediation
**Critique**: Differencing destroys long-run signals (Cointegration).
**Response**: While true, the alternative (running regression on I(1) data) risks spurious regression, which is fatal.
**Action Plan**:
1.  **Automated Differencing**: The pipeline will automatically apply $\Delta x_t = x_t - x_{t-1}$ if ADF p-value > 0.05.
2.  **Error Correction sanity check**: We will add a "Naive VECM" baseline. If the Naive VECM (which uses cointegration) outperforms DML (differenced) by >20% on out-of-sample prediction, we **flag the DML result as potentially under-specified**.

### 2.2 The "Linearity" Remediation
**Critique**: Demand curves are convex; linear DML fails.
**Response**: We reject the "Continuous Treatment" non-parametric approach as a Phase 1 priority due to data hungriness.
**Action Plan**:
1.  **Log-Log Specification**: We will transform all pricing/sales data to logs: $\ln(Q_t) \sim \ln(P_t)$.
    *   *Result*: The coefficient $\theta$ estimates **Elasticity**, which is economically stable across price ranges.
2.  **Regime Dummies**: We will allow $\theta$ to vary by regime (e.g., Inflation High vs Low) using the "Heterogeneity" expansion (Phase 3).

### 2.3 The "Data Mirage" Remediation
**Critique**: Real prices are sticky/sparse.
**Response**: Valid point. Overlap fails if variance is zero.
**Action Plan**:
1.  **Variance Gate**: `if std(diff(log(price))) < 0.01: HALT`. We essentially refuse to model "dead" prices.
2.  **Aggregation**: If weekly data is too sparse, we auto-aggregate to Monthly.

---

## 3. Scope Boundary: When NOT to use Dynamic DML

The critique highlighted that DML isn't a panacea. We define the following boundaries:

| Problem Type | Standard Method | Why DML Fails Here |
| :--- | :--- | :--- |
| **One-off Shock** (e.g., Brexit) | **Synthetic Control** / CausalImpact | No "repeated experiments" to learn nuisance functions. |
| **Pure Forecasting** | **DeepAR / LSTM** | DML sacrifices predictive power for unbiasedness. |
| **Structural Breaks** | **Regime-Switching VAR** | DML assumes constant causal structural $\theta$. |

---

## 4. Revised Validation Protocol

To satisfy the skeptic, we add the **"Skeptic's Baseline"** to the validation suite:

1.  **Naive OLS**: Does DML change the answer? (If $\theta_{DML} \approx \theta_{OLS}$, DML was overkill).
2.  **Naive Random Walk**: Does the DML model predict $Y_{t+1}$ better than $Y_t$? (Sanity check).
3.  **Placebo Outcome**: Does Price affect "Random Noise"? (Should be 0).

## 5. Conclusion

We accept the "Fragility" verdict. Our defense is **Rigorous Engineering**: strict gates, log-transformations, and automated differencing will constrain the chaos of real-world data into the "safe zone" where Dynamic DML works.

## Appendix A: Fallback Protocols (When DML Fails)

If the **Validation Guardrails** (Section 2.6) trigger a HALT, we pivot to these specific alternatives:

| Failure Mode | Symptom | Fallback Protocol |
| :--- | :--- | :--- |
| **Overlap Violation** | Propensity AUC > 0.99 | **Regression Discontinuity (RDD)**. If treatment is deterministic based on a threshold ($X > c$), use that threshold for identification instead of controlling for $X$. |
| **Non-Stationarity** | ADF p-value > 0.05 | **Vector Error Correction (VECM)**. Abandon causal ML. Estimate the long-run cointegrating vector using classical Johansen methods. |
| **Low Data Volume** | $N < 200$ | **Bayesian Structural Time Series (BSTS)**. Use strong priors to regularize the estimate instead of cross-fitting (which requires sample splitting). |
| **Anticipation** | Placebo Effect $\neq 0$ | **Instrumental Variables**. Hunt for an external shock ($Z$) and switch to `DoubleMLIV`. |

