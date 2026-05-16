# Pedagogical Concept Guide: From Time Series to Dynamic DML

**Date**: January 8, 2026
**Purpose**: A structural roadmap for teaching the complexities of Time Series Causal Inference.
**Target Audience**: Data Scientists and Econometricians moving from static/prediction tasks to dynamic causal tasks.

---

## 1. The Educational Arc

To understand Dynamic Double Machine Learning, one must master three distinct domains that are usually taught separately. The pedagogical strategy is to introduce each domain, demonstrate why "naive" approaches fail, and then introduce the solution.

**The Progression**:
1.  **Time Series Difficulties**: Why $iid$ assumptions break everything.
2.  **Causal Inference Trickiness**: Why correlation is not causation (even with big data).
3.  **Double ML Complexity**: Why standard ML fails at inference (Regularization Bias).
4.  **Dynamic DML**: The synthesis, where feedback loops and dynamics create new traps.

---

## 2. Part 1: The Time Series Trap
*Goal: Shatter the assumption that data points are interchangeable.*

### Concept 1.1: Autocorrelation & Effective Sample Size
*   **The "Kinda Known"**: "My data points aren't independent."
*   **The Nuance**: $N=1000$ with correlation $ho=0.9$ isn't 1000 data points. It's effectively only ~50 independent samples. Your p-values are lies.
*   **Visual Intuition**: Imagine a line graph that moves slowly (like a snake). If you know point 100, you effectively know points 90-110. You haven't learned 20 new things; you've learned 1 thing 20 times.
*   **The "Oh S**t" Experiment**:
    1.  Generate two *uncorrelated* random walks.
    2.  Run a regression.
    3.  Watch the t-statistic say "Significant!" (p < 0.05) 80% of the time instead of 5%.

### Concept 1.2: The "Random Shuffle" Crime
*   **The "Kinda Known"**: "I shouldn't peek at the future."
*   **The Nuance**: Random shuffling in Cross-Validation destroys the *structure* of the data. A random forest can interpolate $Y_t$ perfectly using $Y_{t-1}$ and $Y_{t+1}$ (which are in the training set). It learns "curve fitting," not "forecasting."
*   **Visual Intuition**: Imagine a puzzle. Random CV is like taking every other piece out. It's easy to guess the missing piece from its neighbors. Block CV is removing the entire *left half* of the puzzle. Much harder.
*   **The "Oh S**t" Experiment**:
    1.  Train RF on time series with `Shuffle=True`. $R^2 = 0.99$.
    2.  Train RF with `Shuffle=False` (Block split). $R^2 = 0.40$.
    3.  Realize your model was just memorizing the autocorrelation.

### Concept 1.3: The "Embargo" (Purging Leaks)
*   **The "Kinda Known"**: "I split my data into blocks."
*   **The Nuance**: If your labels are "5-day forward returns", then $Y_{train\_end}$ depends on data from $T+1, T+2, \dots, T+5$. If your Test set starts at $T+1$, you have leakage!
*   **Visual Intuition**: The "Radiation Zone". You can't enter the Test set immediately after Training. You must wait for the "radiation" (label overlap) to clear.
*   **The "Oh S**t" Experiment**:
    1.  Create labels $Y_t = P_{t+5} / P_t - 1$.
    2.  Train on $t=1 \dots 100$. Test on $t=101$.
    3.  Realize $Y_{100}$ "saw" $P_{105}$. You are predicting the past.

---

## 3. Part 2: The Causal Inference Minefield
*Goal: Distinguish "Predicting Y" from "Changing Y".*

### Concept 2.1: The "Bad Control" Trap
*   **The "Kinda Known"**: "Control for everything!"
*   **The Nuance**: If you control for a variable that is *caused by* the treatment (a mediator), you delete the effect you're trying to find.
*   **Visual Intuition**: Treatment: **Ad Spend**. Outcome: **Sales**. Mediator: **Web Traffic**.
    *   Path: Ads $ightarrow$ Traffic $ightarrow$ Sales.
    *   If you control for Traffic (hold Traffic constant), increasing Ads *cannot* increase Sales (because you blocked the path!). You will find Effect = 0.
*   **The "Oh S**t" Experiment**:
    1.  Simulate $T ightarrow M ightarrow Y$.
    2.  Regress $Y acksim T$. Result: Positive Effect (Correct).
    3.  Regress $Y acksim T + M$. Result: Zero Effect (Wrong).

### Concept 2.2: The Overlap Paradox
*   **The "Kinda Known"**: "I need data for both treated and untreated groups."
*   **The Nuance**: In high dimensions (or time series), overlap often disappears. If a policy is deterministic ("Always raise prices when inventory < 100"), you *cannot* estimate the causal effect of raising prices because you *never* see the counterfactual (low price with low inventory).
*   **Visual Intuition**: Two clouds of points (Treated/Control). If they are completely separated by a line, you aren't doing "inference," you are doing "wild extrapolation."
*   **The "Oh S**t" Experiment**:
    1.  Create a deterministic policy $T = 1 ackiff X > 0$.
    2.  Try to estimate the effect of $T$ using any method (IPW, DML).
    3.  Watch the weights explode to infinity or estimates behave erratically.

---

## 4. Part 3: Double ML (The "Tricky" Part)
*Goal: Explain why we can't just "throw all confounders into XGBoost".*

### Concept 3.1: Regularization Bias
*   **The "Kinda Known"**: "Lasso shrinks coefficients to zero to prevent overfitting."
*   **The Nuance**: That shrinkage applies to the *confounders* too. If a confounder $X$ is weak, Lasso sets its coefficient to zero. Now you have Omitted Variable Bias again. ML optimizes for *prediction* (MSE), not *parameter recovery*.
*   **Visual Intuition**: Imagine estimating a treatment effect $	heta$ is like tuning a radio dial.
    *   Standard ML assumes the dial is at 0 (prior) and only moves it if there's huge evidence.
    *   DML splits the problem: "First, remove all the noise ($X$) from the room. THEN listen for the signal ($T$)."
*   **The "Oh S**t" Experiment**:
    1.  Simulate $Y = 	heta T + 0.5 X_1 + 	ext{...}$ where $X$ correlates with $T$.
    2.  Run Lasso($Y acksim T, X$).
    3.  Watch $\hat{\theta}$ be biased because Lasso ignored $X_1$ to save variance.
    4.  Run DML. Watch $\hat{\theta}$ snap to the truth.

### Concept 3.2: Why "Double"? (Residualization)
*   **The "Kinda Known"**: "It uses two models."
*   **The Nuance**:
    *   Step 1: Predict $Y$ from $X$ (get residual $\tilde{Y}$: "Sales surprise").
    *   Step 2: Predict $T$ from $X$ (get residual $\tilde{T}$: "Price surprise").
    *   Step 3: Regress $\tilde{Y}$ on $\tilde{T}$.
    *   *Why?* We are correlating the *surprises*. "When Price was surprisingly high (given the economy), were Sales surprisingly low?" This isolates the causal shock.

---

## 5. Part 4: Dynamic DML (The Final Boss)
*Goal: Handling feedback loops where Outcome affects future Treatment.*

### Concept 5.1: Feedback Loops (Endogeneity)
*   **The "Kinda Known"**: "Prices affect sales."
*   **The Nuance**: Sales *also* affect prices (Dynamic Pricing algorithms). If yesterday's sales were low, algorithms drop prices today.
*   **Visual Intuition**: A spiral.
    *   $T_t ightarrow Y_t$ (Causal Effect)
    *   $Y_t ightarrow T_{t+1}$ (Feedback Policy)
    *   Standard regression sees $T_{t+1}$ moving with $Y_t$ and thinks "Price cuts cause *past* low sales" (Reverse Causality) or mixes them up.
*   **The "Oh S**t" Experiment**:
    1.  Simulate a feedback loop.
    2.  Run static DML (treating rows as i.i.d.).
    3.  See biased estimates because the "controls" $X_t$ include past outcomes $Y_{t-1}$, which are "colliders" or "bad controls" in this temporal graph.

### Concept 5.2: Sequential Exogeneity (No Anticipation)
*   **The "Kinda Known"**: "Controls make it random."
*   **The Nuance**: You must assume agents don't act on *future* information that you (the analyst) don't have. If Walmart knows a hurricane is coming next week and raises prices *now*, and you don't control for "Hurricane Forecast," you will see High Price $ightarrow$ High Demand (stockpiling) and get the wrong elasticity.
*   **Visual Intuition**: The "Surprise" $\tilde{T}_t$ must be a true surprise. If the store manager knew it was coming, it's not a surprise, and it's confounded.

### Concept 5.3: Impulse Response Functions (IRF)
*   **The "Kinda Known"**: "The effect is -0.5."
*   **The Nuance**: The effect is a *curve* over time.
    *   Week 0: Sales drop -0.8 (Shock).
    *   Week 1: Sales drop -0.2 (Lingering memory).
    *   Week 2: Sales bounce back +0.1 (Inventory refilling).
    *   Dynamic DML estimates this entire curve, peeling off one lag at a time.

### Concept 5.4: The "Too Good To Be True" Trap
*   **The "Kinda Known"**: "High accuracy is good!"
*   **The Nuance**: In Causal Inference, if your Propensity Model ($X \to T$) has AUC=0.99, you have **failed**.
    *   *Why?* If you can predict treatment perfectly, then treated and control units are *fundamentally different* (Separation). There is no "Overlap" to compare them.
    *   *Analogy*: If you only give medicine to sick people and placebos to healthy people, you can't tell if the medicine works (because you can't compare "sick with medicine" to "sick without medicine").
*   **The "Oh S**t" Experiment**:
    1.  Create a deterministic policy $T_t = 1 \iff Y_{t-1} < 0$.
    2.  Train a classifier for $T_t$. Accuracy = 100%.
    3.  Try to run DML. Weights become infinite ($1/p \to \infty$).
    4.  **The Lesson**: We need "Randomness" (Surprise) to learn causality.

---

## 6. Implementation Roadmap for Pedagogy

### Phase 1: The Basics (Week 1)
*   Build **Notebook 1** (Time Series CV) and **Notebook 2** (Spurious).
*   *Objective*: Verify the user understands why `scikit-learn` defaults are dangerous here.

### Phase 2: The DML Core (Week 2)
*   Build **Notebook 4** (Regularization Bias).
*   *Objective*: Demonstrate the "Why" of DML. This is the hardest conceptual leap for pure data scientists.

### Phase 3: The Dynamic Frontier (Week 3)
*   Build **Notebook 5** (Feedback Loops).
*   *Objective*: Show the power of the `DynamicDML` class we just ported.

---

## 7. Recommended Reading Order
1.  **Chapter 1 (TeX)**: The Theoretical Foundation (Potential Outcomes).
2.  **Notebook 1 & 2**: The Time Series constraints.
3.  **Notebook 4**: The DML Mechanism.
4.  **Chapter 3 (Validation)**: Seeing the proofs in action.
5. **Dynamic DML Report**: The synthesis.

---

## 8. Further Reading & Context (2024-2025 Update)

### The Inference vs. Forecasting Divide
It is crucial to distinguish between two separate goals in time series:
*   **Forecasting (Predicting $Y_{t+1}$)**: The cutting edge (2024/2025) focuses on **Test-Time Adaptation** (e.g., *TAFAS*, *DynaConF*) to handle *non-stationarity* and distribution shifts. "The world is changing, how do I adapt?"
*   **Causal Inference (Estimating $\theta$)**: The standard (Lewis & Syrgkanis, 2021) requires **Stationarity**. We assume "The rules of physics haven't changed" to estimate the effect of gravity.
*   **The Lesson**: If you are doing DML, you *cannot* use the wild non-stationary techniques from forecasting competitions. You must stationarize your data first.

### Key References
*   **Lewis & Syrgkanis (2021)**: "Double/Debiased Machine Learning for Dynamic Treatment Effects". (The Bible for this project).
*   **Runge et al. (2019)**: "PCMCI" - For discovering the causal graph before you estimate effects.
*   **Chernozhukov et al. (2018)**: "Double Machine Learning". (The Foundation).

## Appendix A: The Causal Hierarchy (Granger -> SCM -> DML)

To avoid confusion, we must distinguish between three levels of "Causality" in time series:

### Level 1: Predictive Causality (Granger)
*   **Question**: "Does knowing $X_{t-1}$ help predict $Y_t$ better than just $Y_{t-1}$?"
*   **Tool**: Granger Causality Test.
*   **Limit**: Mere correlation. A barometer "Granger causes" rain, but breaking the barometer won't stop the storm.
*   **Role in Project**: **Exploratory Analysis** (Phase 2).

### Level 2: Structural Causality (SCM)
*   **Question**: "If I force $X_t$ to value $x$, what happens to $Y_t$?"
*   **Tool**: Structural Causal Models (Pearl).
*   **Limit**: Requires knowing the full graph.
*   **Role in Project**: **The Goal**. We assume an SCM exists and try to estimate its parameters.

### Level 3: Policy Evaluation (DML)
*   **Question**: "What is the specific numerical value of $\theta = \partial Y / \partial T$?"
*   **Tool**: Double Machine Learning.
*   **Limit**: Requires "Unconfoundedness" (all common causes are observed).
*   **Role in Project**: **The Engine**. We use DML to quantify the link found in Level 2.