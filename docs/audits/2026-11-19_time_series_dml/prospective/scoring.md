# Can the next experiment distinguish two forecasts?

For each intervention we issue a zero forecast, a design-based historical forecast, exactly half of that historical forecast, and a cross-fitted DML forecast. Halving the historical effect is meaningful on these additive scales with a zero null. The multiplier 0.5 is fixed before scoring. The development archives already informed this choice. Prospective evidence starts with later assignments under one of the new protocols.

Let the frozen DML forecast be `d`, half-history be `s`, and the later design-based estimate be `Z`. Positive gain in squared accuracy favors DML. The primary calculation in outcome units squared is

```text
midpoint = (d + s) / 2
gain = (s − Z)² − (d − Z)² = 2(d − s)(Z − midpoint)
```

We transform both ends of the design-based 95% confidence interval for the later effect through the same expression and order the resulting endpoints. An interval wholly above zero favors DML. An interval wholly below zero favors half-history. An interval touching or crossing zero leaves the relative accuracy unresolved. Equal forecasts have identical loss for every possible outcome. Missing results retain a pending row. This comparison conditions on the numbers actually issued. Questions about refitting the learner across samples require the separate joint resampling calculation in the reminder exercise.

**How much precision is needed?** If either frozen forecast equals the true effect and the later estimate is approximately normal, obtaining an 80% probability that a two-sided 95% interval favors that correct forecast requires a benchmark standard error at most

```text
|d − s| / [2 × (1.960 + 0.842)] ≈ |d − s| / 5.603.
```

Different true effects imply different power. Near the midpoint, even a large experiment can leave the ranking unresolved. The sample size needed to detect an intervention's effect against zero answers a separate question. Report the achieved test-period standard error, the historical design estimate of achievable precision, and the standard error required to distinguish the forecasts. Avoid interpreting the smallest point loss as evidence of a winner. [Scorer and design calculation](../../../../dml_ts/validation/effect_forecast.py)

The reports identify whether DML moved the earlier estimate toward zero, away from zero, across zero, or left it unchanged. A frozen estimate of the historical marginal effect assumes persistence into a new population. Winning against half-history can establish better forecast accuracy on the scored episode. A claim that a model anticipates changes in treatment response needs predictions that vary across episodes, comparisons to shrinkage, and enough independent episodes to evaluate their direction.

The generic scoring interface accepts an optional `comparators` list in the protocol and a `ci95` pair on a later observation. It verifies the digest and target identity before calculating gains. Omit the optional list to reproduce historical outputs in their original format. Pre-existing frozen files retain their recorded digests. The full-precision digests published alongside the new reports let a reader check the exact forecast bytes independently of the working tree. Dates assigned retrospectively to an archived match or survey row do not establish when the source was available.
