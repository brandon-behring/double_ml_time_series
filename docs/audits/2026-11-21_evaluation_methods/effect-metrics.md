# Can a validation score choose a better causal estimator?

In a process with a change in effect, choosing both nuisance learners by the previous block’s outcome prediction error raised next-block causal RMSE from **0.338 to 0.532** on previously unused simulation paths. The selection rule was frozen before those paths were generated. Its penalty was much smaller under stationary serial dependence. A favorable metric comparison needs a specified task, population, period, treatment assignment and independent unit of uncertainty. [Frozen protocol](experiments/selection-protocol.json) · [All selectors and regimes](experiments/selectors.csv)

## A prospective test inside a simulation

The original rolling archive contains three learners that estimate both nuisance functions. On each of the final two evaluable origins we used the previous completed scoring block to choose a learner, then compared the effect estimated after the next block matured with its known structural effect. Recomputing the earlier 500 paths reproduced the retrospective RMSEs of **0.5183 versus 0.3294** under the break and **0.1311 versus 0.1272** under stationary serial dependence. The comparison is against using the original tree pair at every origin. [Historical rolling draws](../2026-11-19_time_series_dml/paired/rolling-draws.csv.gz)

Before generating the next 500 paths per regime we fixed the rules, candidate order, seeds, target blocks, endpoint and uncertainty calculation. The SHA-256 of the frozen protocol is recorded in the run receipt. Each fresh path contains 1,200 dates. For MSE-based selection, both losses on the immediately previous block are observable by the decision date. For R-loss, each candidate slope is frozen before that previous block, then scored using the **same** held-out outcome and treatment residuals from past-trained tree models. Allowing all nine pairs changes the size of the candidate menu. These are tests on new draws from a known generator, with identical fold memberships across paths. They establish reproducibility under that generator. [Run receipt](experiments/run.json) · [Selection, dates, losses and measured effects by path](experiments/selector-draws.csv.gz)

| Rule applied on fresh seeds | Stable serial, later effect RMSE | Break, later effect RMSE | Break, frozen forecast RMSE |
|---|---:|---:|---:|
| Use trees for both throughout | 0.129 | 0.338 | 1.341 |
| Choose both by outcome MSE | 0.133 | 0.532 | 1.572 |
| Choose both by treatment MSE | 0.152 | 0.360 | 1.363 |
| Choose each role by its own MSE | 0.142 | 0.360 | 1.381 |
| Common R-loss, three same-family pairs | 0.172 | 0.364 | 1.371 |
| Common R-loss, all nine pairs | 0.222 | 0.329 | 1.179 |

The first two numeric columns score slopes **re-estimated with later outcomes**. The last scores an effect forecast frozen before that block. Using the 500 independent paths for paired uncertainty, outcome-based selection raised later-block mean squared effect error by **0.1690** (MC SE **0.0041**) in the break regime. Admitting all nine pairs to R-loss lowered it by **0.0062** (MC SE **0.0047**), an inconclusive difference. The same nine-way rule increased later-block MSE by **0.0327** (MC SE **0.0023**) in the stationary serial regime. The two origins per path share histories. We average both paired losses within path before computing MC errors. These are pointwise comparisons over a declared menu. [Paired comparison and frozen-forecast errors](experiments/selectors.csv)

## Which population does residual loss measure?

A validation observation with outcome residual Ỹ and treatment residual D̃ gives loss (Ỹ − D̃τ̂)² for a frozen candidate effect τ̂. Under a conditionally linear causal response, valid conditional assignment and the true nuisance functions, candidate-specific excess population loss is E[Var(D|X)(τ̂(X) − τ(X))²]. A comparison with estimated nuisances also depends on those evaluators. Least residual loss can favor the model that is closest where the treatment varies most. The population ATE, subgroup calibration, treatment-prioritization ranking and policy value use other objectives. [Nie and Wager](https://arxiv.org/abs/1712.04912) · [Vansteelandt and Dukes](https://arxiv.org/abs/2006.08402) · [Curth and van der Schaar](https://proceedings.mlr.press/v202/curth23b.html)

Consider equally populated histories with randomized treatment probabilities 0.50 and 0.01, and true effects 0 and 4. Model A predicts 0 and 0. Model B predicts 1 and 4. The population ATE is 2. The calculations use the assignment probabilities themselves and require no fitted evaluator. [Executable example](experiments/counterexamples.py) · [Numerical outputs](experiments/counterexamples.json)

| Candidate | Excess oracle R-loss | Population CATE RMSE | Absolute population ATE error |
|---|---:|---:|---:|
| A | **0.0792** | 2.828 | 2.0 |
| B | 0.1250 | **0.707** | **0.5** |

The histories with the larger effect make up half the population but carry little residual-treatment weight. A wins for the R-loss objective and B wins for both population targets. Quoting one of those rankings therefore requires specifying the intended target. In the hidden-factor simulations even the conditional score evaluated using the true prediction functions has nonzero mean at the structural effect. A validation rule cannot observe the missing counterfactual in the actual field studies. [Score decomposition](nuisance-prediction.md)

## Choosing an evaluation for a field result

| Evidence available | Valid measurement | Main limit |
|---|---|---|
| Known simulated or planted response | Bias, absolute error, RMSE, interval width, empirical coverage, Monte Carlo uncertainty, failures | Scientific interpretation depends on how treatment, outcome, censoring and common shocks were generated. |
| Published estimate on the same sample | Sample and transformation checks, numerical coefficient/SE equality under matching conventions | The reference can share design or coding errors. Estimate agreement establishes reproduction of that calculation. |
| Randomized comparison in a later block | Error against the design-based contrast, its precision and joint uncertainty when observations overlap | Observed treatment-arm variation, outcome maturity, missingness and stable target weights determine whether the later block can evaluate the earlier effect. |
| Observational time series | Timing, negative controls, sensitivity to assignment and alternatives, held-out residual loss for an explicit objective | Prediction loss and recurrence across dates alone leave hidden confounding unresolved. |
| Individual heterogeneity or treatment prioritization | Independent randomized effect scores for subgroup calibration, held-out targeting contrasts for ranking, design-specific utility for policy value | Individual effects are unobserved in an ordinary parallel-group study. One criterion can improve while another deteriorates. |

For a local replay, run the following commands from the repository root. A smaller smoke test can use `--repetitions 2 --output /tmp/dml-learner-audit`. The complete crossed run uses the fixed 500 original and 500 fresh path IDs in the protocol. All compared nuisance predictions are fitted once per role, fold and family, then reused across pairings.

```sh
audit=docs/audits/2026-11-21_evaluation_methods/experiments
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python "$audit/learner_pairs.py" --workers 20
uv run --locked python "$audit/counterexamples.py"
uv run --locked python -m pytest -q test/validation/test_evaluation_learner_audit.py
```
