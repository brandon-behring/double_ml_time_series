# Data Strategy & Application Demos

**Date**: January 8, 2026
**Purpose**: Defining the datasets required to validate, demonstrate, and productionize Time Series DML.

---

## 1. The Data Hierarchy

We categorize data into three tiers based on their role in the project lifecycle:

| Tier | Role | Type | Primary Dataset |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Validation** | Synthetic | `TimeSeriesDGP` (Internal) |
| **Tier 2** | **Demonstration** | Public Real-World | **Dominick's Orange Juice (OJ)** |
| **Tier 3** | **Production** | Private Real-World | **MyGA Elasticity** (Internal) |

---

## 2. Tier 1: Synthetic Validation Data
*Goal: Prove the math works (Coverage, Bias).*

*   **Source**: `src/validation/dgp_generator.py` (The `TimeSeriesDGP` class we just wrote).
*   **Key Features**:
    *   Known ground truth $\theta$.
    *   Controllable autocorrelation $\rho$.
    *   Controllable confounding strength.
*   **Action**: This is fully implemented. No external fetch needed.

---

## 3. Tier 2: Demonstration Data (The "OJ" Dataset)
*Goal: Create a relatable "Hello World" pricing notebook.*

*   **Dataset**: **Dominick's Finer Foods Orange Juice Sales**.
*   **Structure**: Weekly panel data (121 weeks $\times$ 83 stores).
    *   $Y$: `logmove` (Log Unit Sales).
    *   $T$: `log(price)` (Log Price).
    *   $X$: `feat` (Advertisement), `INCOME`, `AGE60` (Demographics).
*   **Why OJ?**:
    *   It is the standard benchmark in `EconML` and `DoubleML`.
    *   It has "dynamic" features (inventory effects, price stickiness).
    *   It allows comparing our *Dynamic* DML against standard *Static* DML to show the "value add" (e.g., estimating the lag effect of a promotion).
*   **Acquisition**:
    *   Available in `econml.data`.
    *   Can be fetched via `pandas` from standard URLs if `econml` is not installed.

---

## 4. Tier 3: Production Data (MyGA & Macro)
*Goal: Solve the user's actual business problems.*

### 4.1 MyGA Elasticity (Internal)
*   **Source**: `~/Claude/myga-elasticity/data`.
*   **Context**: Annuity sales vs. Credited Rates.
*   **Challenge**: Rates are highly autocorrelated (sticky). Competitor rates are confounders.
*   **Plan**: Use this as the "Final Exam" for the `DynamicDML` engine.

### 4.2 Macroeconomic Policy (FRED)
*   **Source**: Federal Reserve Economic Data (FRED).
*   **Context**: Effect of Interest Rates ($T$) on Inflation ($Y$).
*   **Challenge**: Extreme non-stationarity.
*   **Plan**: Use this to demo the **Stationarity/Differencing** pipeline.
*   **Acquisition**: `pandas-datareader`.

---

## 5. Proposed Demo Notebooks

### Demo 1: "Pricing Orange Juice with Dynamic DML"
*   **Objective**: Estimate Price Elasticity.
*   **Comparison**:
    1.  OLS (Naive): Elasticity $\approx -2.0$ (Biased).
    2.  Static DML: Elasticity $\approx -3.0$ (Better).
    3.  **Dynamic DML**: Show that a price cut today increases sales today (+3.0) but *decreases* sales next week (-0.5) due to "pantry loading" (stockpiling).
*   **Key Lesson**: Only Dynamic DML captures the "Pantry Loading" effect. Static methods overestimate the long-term lift.

### Demo 2: "Macro Shocks"
*   **Objective**: Estimate the impulse response of Inflation to Interest Rates.
*   **Key Lesson**: Stationarity is king. If you don't difference the data, DML explodes.

---

## 6. Action Items

1.  **Create Loader**: `src/data/oj_loader.py` to fetch/clean the Orange Juice dataset.
2.  **Create Loader**: `src/data/fred_loader.py` for Macro data.
3.  **Link**: Connect `myga-elasticity` data to the validation pipeline (Phase 3).
