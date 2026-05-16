# Independent Skeptical Critique: Time Series DML

**Date**: January 8, 2026
**Auditor**: Gemini (Devil's Advocate Persona)
**Status**: Critical Review
**Target**: All prior reports (`IMPLEMENTATION`, `GAP_ANALYSIS`, `PEDAGOGY`, `DATA_STRATEGY`)

---

## 1. Executive Summary

While the project has successfully pivoted to a valid theoretical framework (Dynamic DML), the current roadmap is **dangerously optimistic** about data properties and model specification.

**The Core Indictment**: We are building a sophisticated inference engine (Dynamic DML) that assumes a pristine world of **Stationarity** and **Linearity**. Real-world pricing data violates both.

---

## 2. Critique of Core Assumptions

### 2.1 The "Stationarity First" Dogma
The roadmap mandates "Stationarize everything!" (differencing).
*   **The Risk**: **Over-differencing**. If prices ($P_t$) and sales ($Q_t$) are cointegrated (long-run equilibrium), differencing them ($\Delta P_t, \Delta Q_t$) destroys the Error Correction term. We throw away the long-run signal to save the short-run inference.
*   **Evidence**: Standard time series texts (Hamilton, 1994) warn that differencing cointegrated series leads to misspecification.
*   **Verdict**: The roadmap lacks a **Cointegration Test** (Engle-Granger). We might be solving the wrong problem.

### 2.2 The "Linearity Straitjacket"
We heavily rely on the **Partially Linear Model (PLM)**: $Y = \theta T + g(X)$.
*   **The Risk**: Pricing is fundamentally non-linear. The effect of a price hike from \$10 to \$11 is different from \$100 to \$101.
*   **Critique**: The `DynamicDML` port is strictly linear ($\theta$ is a scalar). This assumes **constant elasticity** across the entire price range.
*   **Verdict**: The "Continuous Treatment DML" upgrade (Phase 2) is not a "nice to have"—it is **critical**. Without it, we are fitting a flat line to a curve.

### 2.3 The "Data Mirage"
The Data Strategy assumes we can just "fetch OJ data" or "use FRED".
*   **The Risk**: **Sparse/Sticky Prices**. In many industries (e.g., insurance), prices don't change every week. They stay flat for months.
*   **Consequence**: If $\Delta T_t = 0$ for 90% of observations, the propensity score $p(X_t)$ becomes degenerate (predicts "no change" perfectly). **Overlap violations** will be rampant.
*   **Verdict**: The "Suspicious Improvement Gate" (Concept 5.4) is likely to trigger constantly. We need a strategy for **Low-Variance Treatments**.

### 2.4 The "Placebo" Fallacy
We propose "Placebo Tests" (estimating effect on unrelated outcomes) as validation.
*   **The Risk**: In macroeconomics/finance, *everything* is correlated. Interest rates affect "unrelated" sectors via general equilibrium effects. Finding a "true zero" outcome is incredibly hard.
*   **Verdict**: Placebo tests may yield "False Positives" (finding effects where there are none) simply due to the interconnected nature of time series.

---

## 3. Recommendations & Corrective Actions

### Immediate (Phase 1)
1.  **Add Cointegration Checks**: Don't just ADF test. Run **Engle-Granger** on $Y$ and $T$. If cointegrated, we need a Vector Error Correction Model (VECM) baseline, not just differenced DML.
2.  **Warning System**: Update `BiasValidation` to warn if `std(treatment) < threshold`.

### Strategic (Phase 2)
3.  **Prioritize Non-Linearity**: Move "Continuous Treatment DML" (non-parametric $\theta(t)$) up the priority list.
4.  **Skeptic's Metric**: Add a "Naive Baseline" to every report. If DML doesn't beat OLS by a significant margin on *out-of-sample* prediction (even though it's for inference), be suspicious.

---

## 4. Final Verdict

**Feasible, but Fragile.**
The project is theoretically sound *if* the data cooperates (high variance, stationary, linear). Real-world data rarely does. The engineering challenge is not the DML code, but the **Data Preprocessing Pipeline** to force the world into the DML box.
