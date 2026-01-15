# Gap Analysis: Dynamic DML vs. Project Roadmap

**Date**: January 8, 2026
**Target**: Aligning `src/dml` capabilities with Phase 3 (Pricing Application) requirements.
**Reference**: Lewis & Syrgkanis (2021) "Double/Debiased Machine Learning for Dynamic Treatment Effects".

---

## 1. The Reference Implementation (`syrgkanislab/dynamicdml`)

The official implementation (and our port) focuses on **Linear Dynamic DML**.

### Core Logic
The estimator solves for scalar parameters $\theta_h$ in the model:
$$ Y_t = \sum_{h=0}^M \theta_h T_{t-h} + g(X_t) + \epsilon_t $$

It uses **Sequential g-estimation** to "peel off" these effects:
$$ \hat{\theta}_h = \frac{\text{Cov}(\tilde{Y}_t, \tilde{T}_t)}{\text{Var}(\tilde{T}_t)} $$

### Capability Map
| Feature | Status in Reference | Status in Our Port | Roadmap Requirement | Gap? |
| :--- | :--- | :--- | :--- | :--- |
| **Binary Treatment** | ✅ Native | ✅ Supported | Baseline | No |
| **Continuous Treatment** | ⚠️ Implicit (Linear) | ⚠️ Implicit (Linear) | **Critical** (Pricing) | **YES** |
| **Time Series CV** | ✅ Supported | ✅ Supported | Phase 1B | No |
| **HAC Inference** | ✅ Supported | ✅ Supported | Phase 1B | No |
| **Stationarity** | ❌ Assumed | ❌ Assumed | **Critical** (Phase 2) | **YES** |
| **Non-Linear Effects** | ❌ Not Implemented | ❌ Not Implemented | Phase 3 (Elasticity) | **YES** |

---

## 2. Identified Gaps & Difficulties

### Gap 1: The Linearity Restriction (Critical for Pricing)
**The Problem**: The current `sequential_g_estimation` calculates a single scalar $\theta_h$ for each lag. This implies a **constant marginal effect** of price on sales (linear demand curve).
*   *Reality*: Demand curves are convex. Elasticity varies with price.
*   *Workaround*: If we log-transform variables ($y=\log(\text{Sales})$, $T=\log(\text{Price})$), $\theta$ becomes **elasticity**. This is often constant enough for local analysis.
*   *Difficulty*: If we need *varying* elasticity (e.g., customers are more sensitive at high prices), the current linear estimator fails. We would need **Non-Parametric DML** (estimating $\theta(T)$), which is mathematically much harder (requires solving integral equations).

### Gap 2: Stationarity & Integration
**The Problem**: Neither the reference repo nor our port checks for unit roots.
*   *Scenario*: If Price ($T_t$) and Sales ($Y_t$) are random walks (I(1)), the covariance terms in g-estimation $\text{Cov}(\tilde{Y}, \tilde{T})$ do not converge to constants. The estimator diverges.
*   *Difficulty*: "Auto-differencing" is risky. Differencing removes low-frequency signals (trends) which might contain the bulk of the causal info (e.g., long-term price drift affecting market share).
*   *Proposed Solution*: **Fractional Differencing** or strictly checking cointegration residuals. This is "Research Grade" difficulty.

### Gap 3: Validation on Real Data
**The Problem**: The reference repo validates on *synthetic* data (`dgp.py`).
*   *Our Goal*: Validate on *real* data (or realistic proxies).
*   *Difficulty*: We don't know the ground truth $\theta$ for real data.
*   *Proposed Solution*: **Placebo Tests** and **Public Benchmarks**.
*   *Status*: Solved. See `docs/DATA_STRATEGY_REPORT.md` for the selection of Dominick's OJ (Public) and MyGA (Private) datasets.

### Gap 4: Structural Uncertainty (Causal Discovery)
**The Problem**: Dynamic DML assumes we *know* the causal graph (e.g., "Price affects Sales at lags 0-5").
*   *Reality*: The true lag structure is unknown. Does it last 3 weeks or 30?
*   *Solution*: **PCMCI Integration**. Use the PCMCI algorithm (Runge et al., 2019) from our `causal_inference_mastery` repo to *discover* the significant lags ($X_{t-\tau} \to Y_t$) before feeding them into DML.
*   *Status*: Missing in current `double_ml_time_series`.

### Gap 5: The Forecasting Divergence
**The Problem**: Recent 2024-2025 literature focuses on "Test-Time Adaptation" for non-stationary forecasting.
*   *Conflict*: DML requires stationarity for valid inference. We cannot use these modern "adaptive" forecasting tricks because they change the estimand $\theta$ over time.
*   *Decision*: We explicitly reject "Adaptive Forecasting" methods in favor of **strict stationarization** (Differencing/Fractional Differencing) to preserve the definition of a stable causal effect.

### Gap 6: Inference Robustness (Bootstrap vs. Asymptotic)
**The Problem**: Newey-West SEs rely on asymptotic normality. In finite samples with heavy tails (common in finance), this can understate risk.
*   *Alternative*: **Block Bootstrap**. Resample blocks of data and re-estimate.
*   *Trade-off*: Computational cost. Bootstrapping DML requires re-training ML models hundreds of times.
*   *Decision*: Stick to Newey-West for the primary engine (speed). Reserve Block Bootstrap for "Final Audit" reports where compute time is secondary to rigor.

---

## 3. Remediation Strategy

### Addressing Linearity (Phase 3)
Instead of building a complex Non-Parametric DML from scratch:
1.  **Log-Log Specification**: Stick to constant elasticity ($\\log Y$ on $\\log T$) as the MVP.
2.  **Binning**: Discretize continuous Price into "Low", "Medium", "High" bins and estimate separate dynamic effects for each (approximating the curve).

### Addressing Stationarity (Phase 2)
1.  **Pipeline Guardrails**: Add a `StationarityCheck` class in `src/validation`.
2.  **Strict Enforcement**: If ADF p-value > 0.05, raise `ValueError` before training. Force the user (or the pipeline) to difference the data.

### Addressing Validation (Phase 1B)
1.  **Synthetic Stress Test**: We must generate *non-linear* DGPs and see how badly the Linear DML fails. This "Failure Mode Analysis" is just as valuable as passing tests.

---

## 4. Conclusion

The current "Linear Dynamic DML" engine is sufficient for Phase 1 and 2, provided we stick to **constant elasticity** assumptions (Log-Log) for pricing. The major technical hurdle is **not** the estimator code (which we have), but the **data engineering** (Stationarity) and **model specification** (Log-Log vs Linear) required to make it work on real time series.

## Appendix A: The "Buy vs Build" Decision

We considered using off-the-shelf libraries (`DoubleML`, `EconML`) versus porting our own (`src/dml`).

| Feature | `DoubleML` (Python) | `EconML` (Microsoft) | Custom Port (`src/dml`) | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Dynamic DML** | ❌ Not native (requires manual lag setup) | ⚠️ Experimental / Complex API | ✅ **Native** (Lewis & Syrgkanis exact match) | **Build** |
| **Time Series CV** | ⚠️ Basic (TimeSeriesSplit) | ❌ Standard K-Fold defaults | ✅ **Advanced** (Blocked, Rolling, Panel) | **Build** |
| **HAC Inference** | ❌ Standard Cluster SE | ❌ Robust SE | ✅ **Newey-West** (Bartlett Kernel) | **Build** |
| **Maintenance** | High (External dep) | High (External dep) | Low (Self-contained) | **Build** |

**Conclusion**: Porting the `syrgkanislab` logic to `src/dml` was the correct choice. It gives us full control over the **HAC Inference** and **Cross-Fitting** details which are often glossed over in general-purpose libraries.

