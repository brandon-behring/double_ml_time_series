# What Chapter 9 can establish about heterogeneous effects and policies

The original heterogeneity test always assigned a slope of one to any nonconstant prediction. It regressed the prediction on a centered copy of itself. Changing every evaluation treatment and outcome left the result identical. Across 2,000 simulations with a constant treatment effect, the test rejected homogeneity 2,000 times. This was an error in the chapter's teaching code. The corrected example freezes a prediction from separate training people and scores it against randomized outcomes from evaluation people. [Before/after experiment](heterogeneity/verify.py) · [Measured rejection rates](heterogeneity/blp-calibration.json) · [Revised chapter](../../../chapters/chapter_09.tex) · [Web edition](../../../web/src/content/chapters/chapter09-heterogeneity.mdx)

## Which prediction is being checked?

| Quantity | What can score it | What a good result establishes |
|---|---|---|
| Outcome mean and treatment probability | Outcomes and assignments on unused people or future supported dates | Predictive accuracy or probability calibration for those variables |
| Conditional treatment effect, in simulation | Individual covariate profiles and their known conditional mean effects | CATE bias or RMSE for the simulated population |
| Effect ordering and calibration, in a trial | A score fixed in advance and an outcome contrast with known randomization probabilities | Alignment between predicted and actual **conditional mean** effect differences |
| Decision value | The frozen decision for each eligible observation, an identifiable payoff difference, cost and a named comparator | Expected incremental net benefit of deploying that particular rule in the evaluation population |

Under binary Bernoulli randomization with known probability $p(X)$ in $(0,1)$, the signed outcome $Z=(T-p(X))Y/[p(X)(1-p(X))]$ has conditional mean $E[Y(1)-Y(0)\mid X]$. Either potential outcome can be missing for a person. Regressing $Z$ on an intercept and a centered frozen prediction $S$ estimates the projection slope $\operatorname{Cov}(S,\tau(X))/\operatorname{Var}(S)$. A positive slope measures linear alignment. Perfect effect calibration also requires the appropriate intercept and slope one. A flat score has no identifiable slope. The example uses HC2 on independent evaluation people. A covariate-adjusted augmented score can improve precision if its outcome models are fitted separately. Assignment by village, dates with correlated shocks, or a policy fitted using evaluation outcomes requires a different uncertainty analysis. [Generic ML inference in randomized experiments](https://doi.org/10.3982/ECTA19303) · [R-learner](https://doi.org/10.1093/biomet/asaa076)

## How often does the test detect heterogeneity?

For each of 2,000 seeds, the experiment used 300 independent people for fitting and 300 more for evaluation. The learner could see only $X_0$. It fitted a linear predictor on the training signed outcomes. Arm allocation was one half. The three scenarios reused the same covariates, noise and assignments. Each two-sided test used a nominal 5% cutoff. The Monte Carlo standard errors below measure uncertainty from the finite number of replications. [Simulation design](heterogeneity/verify.py) · [Counts and Monte Carlo standard errors](heterogeneity/blp-calibration.json)

| True effect | Original rejection | Corrected rejection | Monte Carlo SE of corrected rate |
|---|---:|---:|---:|
| Constant at one | 2,000 / 2,000 | 127 / 2,000 (6.35%) | 0.55 percentage points |
| $1+0.5X_0$ | — | 1,785 / 2,000 (89.25%) | 0.69 points |
| $1+0.5X_1$, where $X_1$ is independent of $X_0$ | — | 125 / 2,000 (6.25%) | 0.54 points |

With separate pure noise predictions and outcomes at seed 7, the original slope was 1.000 (SE below $4\times10^{-18}$). The corrected slope was 0.00695 (SE 0.12157, $p=0.9544$). The residual finite-sample over-rejection under a constant effect warrants reporting. The third row demonstrates that a failure to reject is compatible with real heterogeneity along a covariate the learner did not use. A later cohort with less overlap, a narrower spread of frozen predictions or fewer independent assignment units can have still lower power. [Single-draw and repeated controls](heterogeneity/blp-calibration.json)

## Which estimator wins when the CATE is known?

All 11 code listings ran in sequence against EconML 0.16.0. In 1,200 training and 800 evaluation people from the chapter's randomized linear-effect simulation, the linear effect model attained CATE RMSE **0.1755**, versus **0.3357** for the causal forest. The forest's evaluation projection slope was **0.9434**, HC2 SE **0.1194**. It discriminated effect differences even though a simpler model predicted effect magnitudes more accurately. The fitted linear CATE at the zero covariate vector was **0.9730**. Averaging fitted CATEs across evaluation people gave **0.9348**. The level at the reference profile and the cohort mean require separate labels. [Replay and dependency versions](heterogeneity/chapter-replay.json) · [EconML CausalForestDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html)

The original partial dependence display averaged lower and upper pointwise confidence limits. To see the problem, consider two profiles whose slope errors are equal and opposite. Their mean slope error cancels, while the average of two positive pointwise half-widths remains positive. Uncertainty for an average needs the joint covariance across fitted profiles and a declared covariate target. The revised curve reports point predictions on a prespecified reference sample. Separate pointwise intervals refer to named profiles. A uniform band would also have to account for searching across the displayed grid. [Chapter example](../../../chapters/chapter_09.tex) · [Effect inference API](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html)

The synthetic insurance example now compares direct one-basis-point effects conditional on three previous rates over months 90–119 for the same 20 products. All training ends before month 90. Inner retrospective date blocks have a three-month buffer on both boundaries. On the moderate path the generator's effect SD is zero, yet fitted effects have SD **0.0214** and RMSE **0.2257**. On the full path the true and fitted SDs are **0.1971** and **0.1259**, with RMSE **0.3539**. A histogram alone could suggest heterogeneity in the homogeneous case. The window gives one realized path and provides no calibration of repeated-panel uncertainty. [Insurance code and fixes](../../../chapters/chapter_09.tex) · [Measured results](heterogeneity/chapter-replay.json)

## What is the value of the proposed decision?

A tree that describes predicted effects, a ranking of people, and a policy assigning treatment above a cost threshold answer different questions. In an equal-sized two-group control, true effects of −1 and +2 and a cost of 1 give a net gain of **+0.5** for a calibrated threshold rule versus treating nobody. Adding 10 to both predictions preserves every rank and changes the policy gain to **−0.5** by treating everyone. The regression test also checks a trial with different assignment probabilities by stratum. [Executable controls](../../../test/validation/test_heterogeneity_evaluation_audit.py) · [RATE for treatment prioritization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12002561/) · [Policy learning](https://doi.org/10.3982/ECTA15732)

The chapter's tree fixed its decisions using training data. With cost 0.2 per treated person it treated **65.75%** of the evaluation people. Its estimated gain versus treating nobody was **1.2638** (SE **0.1525**). The synthetic truth for precisely these people and decisions was **1.2630**. Treating everyone had estimated gain **0.8991**. The direct paired comparison of the tree against treating everyone gave **0.3647** (SE **0.0991**). Real deployment also requires that actions, costs, interference and assignment support match those used in scoring. For logged decisions the assignment probabilities and the number of units supporting high-weight actions matter. [Measured comparison](heterogeneity/chapter-replay.json) · [Policy tree API](https://www.pywhy.org/EconML/_autosummary/econml.cate_interpreter.SingleTreePolicyInterpreter.html)

## A public trial for an external ranking check

Yadlowsky and colleagues evaluated aspirin prioritization with the International Stroke Trial. The trial authors released data for 19,435 randomized people, baseline variables, early outcomes and six-month follow-up. A next test would first reproduce the paper's treatment, endpoint, direction of benefit and held-out ranking comparison on its original sample. A temporal extension would freeze a rule before later enrollment, admit only earlier six-month labels available at that cutoff, retain the factorial heparin assignment, and score the frozen rule against later randomized treatment contrasts. Enrollment dates, assignment probabilities, missing outcomes and contributing centers must be checked from the release before specifying folds. This is a proposed external evaluation. The simulations above supply the measured evidence in this report. [Investigators’ public dataset](https://datashare.ed.ac.uk/items/91709c88-d913-45d8-8019-380027133d54) · [Published RATE application](https://pmc.ncbi.nlm.nih.gov/articles/PMC12002561/)

## Reproduce

From the repository root, the first command uses project dependencies. The full chapter replay also needs the optional EconML package. The stored replay records the versions used. Both commands write only the requested JSON, and the full replay creates its figures in a temporary directory.

```bash
python docs/audits/2026-11-21_evaluation_methods/heterogeneity/verify.py --output /tmp/ch9-calibration.json
python docs/audits/2026-11-21_evaluation_methods/heterogeneity/verify.py --chapter --output /tmp/ch9-replay.json
pytest -q test/validation/test_heterogeneity_evaluation_audit.py test/docs/test_tex_mdx_drift.py
```
