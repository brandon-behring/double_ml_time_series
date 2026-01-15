# Expansion Opportunities: Beyond the Core Engine

**Date**: January 8, 2026
**Purpose**: Exploring advanced capabilities to extend the Time Series DML framework.
**Status**: Research & Planning

---

## 1. Heterogeneous Dynamic Effects (Phase 3)
*Goal: Personalization in Time.*

The current `DynamicDML` estimates the *Average* Dynamic Effect $\theta_h$. However, the effect of a price cut likely varies by customer segment or economic regime.

### Methodology
We can extend the core estimator by interacting the treatment $T_{t-h}$ with a state variable $S_t$:
$$ Y_t = \sum_{h=0}^H \theta_h(S_t) T_{t-h} + g(X_t) + \epsilon_t $$

*   **Implementation**:
    *   Use the existing `sequential_g_estimation`.
    *   Instead of a simple mean of residuals, run a **Weighted Regression** of $\tilde{Y}$ on $\tilde{T}$ interacting with $S$.
    *   *Reference*: `experiment_hetero.py` in the `syrgkanislab/dynamicdml` repo demonstrates this.

### Application
*   **Recession vs. Expansion**: Does marketing work better in a boom?
*   **Customer Segmentation**: Do loyal customers react faster to price changes?

---

## 2. Local Projections (The "Model-Free" Benchmark)
*Goal: Robustness check against structural assumptions.*

Dynamic DML imposes a specific structure (peeling off lags). **Local Projections (Jordà, 2005)** estimate the effect at horizon $h$ using a direct regression:
$$ Y_{t+h} = \beta_h T_t + \text{Controls} + u_{t+h} $$

### Synergy
*   **Validation**: If `DynamicDML` (structured) and `LocalProjections` (unstructured) agree on the Impulse Response Function (IRF), the result is robust.
*   **Code Re-use**: We found `local_projections.py` in `causal_inference_mastery`. It already uses Newey-West inference, making it fully compatible with our pipeline.

---

## 3. Dynamic Policy Optimization
*Goal: From "What happens?" to "What should we do?"*

Once we have the dynamic effect $\theta_h$, we can optimize the treatment sequence $T_1, \dots, T_H$.

### The Challenge
Pricing is a dynamic control problem. Lowering price today ($T_t \downarrow$) increases sales today ($Y_t \uparrow$) but might cannibalize sales tomorrow ($Y_{t+1} \downarrow$) due to stockpiling.

### Solution
*   **Off-Policy Evaluation (OPE)**: Use the estimated Dynamic DML model to score counterfactual pricing strategies.
*   **Bellman Equation**: If we assume a state space, we can solve for the optimal policy $\pi(X_t)$.

---

## 4. Panel Data Extensions
*Goal: Leveraging cross-sectional power.*

We currently have `PanelStratifiedSplit`. We can go further:
*   **Correlated Random Effects (CRE)**: Modeling the unit-specific unobserved heterogeneity. `dml_cre.py` in `causal_inference_mastery` is a starting point.
*   **Fixed Effects DML**: Explicitly de-meaning data within units before DML.

---

## 5. Summary Recommendation

1.  **Immediate**: Integrate **Local Projections** as a validator (Phase 1B).
2.  **Near-Term**: Implement **Heterogeneous Effects** (Phase 3).
3.  **Long-Term**: Build a **Policy Optimizer** on top of the DML engine.
