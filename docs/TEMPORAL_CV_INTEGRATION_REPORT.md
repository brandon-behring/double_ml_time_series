# TemporalCV Integration Report: Robustifying Time Series Validation

**Date**: January 8, 2026
**Author**: Gemini (Lead Architect)
**Status**: Adopted
**Reference**: `temporalcv` (Internal Python Library), Lopez de Prado (2018).

---

## 1. Executive Summary

We have identified an existing, production-grade Python library `temporalcv` within the user's environment. This library already implements the advanced financial cross-validation strategies we previously planned to build.

**Core Decision**: Instead of rebuilding `src/dml/cross_fitting.py` from scratch, we will **integrate `temporalcv`** as a core dependency. This provides immediate access to "Purged" and "Embargoed" validation schemes critical for financial time series.

---

## 2. Key Capabilities of `temporalcv`

### 2.1 Purged Walk-Forward CV
*   **File**: `temporalcv/cv_financial.py`
*   **Class**: `PurgedWalkForward`
*   **Feature**: Implements **Purging** (removing training data overlapping with test labels) and **Embargo** (removing training data *after* test labels to kill serial correlation).
*   **Relevance**: This is the "Gold Standard" for financial DML. Standard `BlockedTimeSeriesSplit` leaks information if labels involve forward returns (e.g., 5-day sales change).

### 2.2 Validation Gates
*   **File**: `temporalcv/gates.py`
*   **Function**: `gate_suspicious_improvement`
*   **Logic**: HALT if Model Metric > Baseline Metric + 20%.
*   **Relevance**: Perfectly aligns with our "Too Good To Be True" guardrail.

### 2.3 Signal Verification
*   **File**: `temporalcv/gates.py`
*   **Function**: `gate_signal_verification`
*   **Logic**: Permutation test (Shuffled Target).
*   **Relevance**: Serves as our "Placebo Test". If DML finds an effect on shuffled targets, the model is broken.

---

## 3. Integration Plan

### Phase 1: Dependency Linking
1.  Add `temporalcv` to `pyproject.toml` or `requirements.txt` (local path dependency).
2.  Replace `src/dml/cross_fitting.py` with a wrapper that imports `PurgedWalkForward` from `temporalcv`.

### Phase 2: Updating Dynamic DML
*   **Target**: `src/dml/dynamic_dml.py`
*   **Change**: Update `dynamic_dml` to accept `cv_strategy="purged"` and pass `purge_gap` / `embargo_pct` to the splitter.

### Phase 3: Validation Suite Upgrade
*   **Target**: `src/validation/bias_validation.py`
*   **Change**:
    *   Call `gate_suspicious_improvement` after estimation.
    *   Call `gate_signal_verification` as a "Placebo" check.

---

## 4. Benefit Analysis

| Feature | Build from Scratch | Integrate `temporalcv` | Benefit |
| :--- | :--- | :--- | :--- |
| **Purged CV** | 3 days dev + test | **Immediate** | Time savings, tested code. |
| **Embargo Logic** | Complex to get right | **Standardized** | Avoids subtle leakage bugs. |
| **Maintenance** | High (our burden) | **Shared** | Leverages existing tooling. |

---

## 5. Conclusion

The `temporalcv` library is a hidden gem that solves the hardest engineering challenges of Time Series Validation. Integrating it elevates our project from "Academic Implementation" to "Financial Grade" immediately.