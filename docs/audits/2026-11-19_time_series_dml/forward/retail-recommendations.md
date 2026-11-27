# Did a retail recommendation policy keep its click advantage later in the week?

In ZOZOTOWN’s men’s campaign, Thompson sampling increased clicks by **0.1638 percentage points** relative to a uniform recommender during November 24–26, 2019. Frozen at midnight UTC on November 28, that estimate predicted the direction and came within **0.0238 points** of the randomized difference over the next three days. The 95% interval for the prediction error was **−0.0527 to +0.1003 points**. The prediction of zero missed by **0.1401 points**. [Original study, section 3 and Table 1](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/33e75ff09dd601bbe69f351039152189-Paper-round2.pdf) · [Pinned data release](https://huggingface.co/datasets/zozonext/open-bandit/tree/90a1283aecd19026caf15a0840af1176a41f6a76)

## The frozen forecast

The platform randomized the recommendation policy for an impression and displayed three items. We measure clicks per displayed item under each policy. The seven-day replay counts 27,495 clicks on 4,077,727 Thompson rows (**0.6743%**) and 2,321 clicks on 452,949 uniform rows (**0.5124%**), reproducing the paper’s Table 1 at its reported precision. The policy contrast takes the difference of those arm rates over the specified dates. It targets the two deployed policies on the traffic observed in that period. The logs provide no stable person ID for evaluating whether an effect carried over to a new population of shoppers. [Study](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/33e75ff09dd601bbe69f351039152189-Paper-round2.pdf) · [Field definitions](https://research.zozo.com/data.html)

| Delay allowed for a click to reach the operator | Earlier dates in November | Predicted lift (pp) | Later lift (pp) | Forecast minus later lift (pp), 95% interval |
|---|---|---:|---:|---:|
| None | 24–27 | +0.1780 | +0.1401 | +0.0380 [−0.0388, +0.1147] |
| **24 hours** | **24–26** | **+0.1638** | **+0.1401** | **+0.0238 [−0.0527, +0.1003]** |
| 48 hours | 24–25 | +0.1509 | +0.1401 | +0.0109 [−0.0692, +0.0909] |

The later differences were **+0.1395**, **+0.2094** and **+0.0724 points** on November 28, 29 and 30. A day with heavier traffic contributes more to the pooled estimate. The primary historical window contains 1,756,059 Thompson and 194,057 uniform rows. The later comparison contains 1,723,359 and 192,714. The timestamps mark impressions. The primary 24-hour reporting delay is an assumption because the archive gives no separate click-arrival timestamps. The operator comparison is reconstructed retrospectively, and the public release came after the experiment. [Source archive and schema](https://huggingface.co/datasets/zozonext/open-bandit) · [Analysis](../external/forward/open_bandit.py)

All dates stay in UTC. The split uses the 168 distinct campaign hours, with every row in an hour assigned together. We sum the influence of both arms in each clock hour and estimate a Bartlett HAC interval through lag five, preserving the 24-hour gap between training and evaluation. For the primary error, 95% intervals using spans of one, twelve and twenty-four hours are [−0.0550, +0.1026], [−0.0517, +0.0992] and [−0.0336, +0.0812] points. These are approximate sampling intervals for the observed periods. Three evaluation days give limited information about calendar variation or correlations among repeat shoppers over longer periods. [Split and uncertainty calculation](../external/forward/open_bandit.py)

## Could the Thompson logs recover the concurrent randomized benchmark?

The **observed uniform-policy click rate was 0.5417%** over November 28–30. We withheld all those uniform-arm clicks from estimation and used them to score two estimates made from later Thompson traffic. This asks whether another policy’s value can be recovered during those same days. The forward forecast above answers a separate question with earlier outcomes.

| Method applied to the later Thompson logs | Inferred uniform click rate | Estimate minus observed uniform rate (pp), 95% interval |
|---|---:|---:|
| Inverse-probability weighting | 0.5284% | −0.0134 [−0.0940, +0.0673] |
| Doubly robust, earlier click model | 0.3609% | −0.1809 [−0.5028, +0.1411] |

The weight is the uniform probability (1/34) divided by the logged probability that the Thompson policy would display the item in that position. The randomized policy comparison uses the arm labels directly. The off-policy comparison assumes that, conditional on the item, position and relevant context, the other displayed items do not change its click response. The authors estimated the Thompson probabilities by Monte Carlo. We use their values without clipping. [Paper, sections 3 and 6](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/33e75ff09dd601bbe69f351039152189-Paper-round2.pdf) · [Logging fields](https://research.zozo.com/data.html)

The click model uses counts by item and position from November 24–26 Thompson traffic. We shrink each cell toward its earlier position mean by adding 100 observations at that mean. Averaging its fitted rates across all 34 candidate items gives **0.5331%** on later positions. Weighting the fitted rates for the items actually displayed gives **0.7006%**. The doubly robust correction adds the former and subtracts the latter from the inverse-weighted estimate. The intervals compare both policies in the same hours, conditional on the frozen model. [Implementation and tests](../../../../test/integration/test_open_bandit_benchmark.py)

All 34 items appeared in each of three positions, but a handful of impressions accounted for much of the estimated policy value. The mean weight was **1.273**, the largest was **29,412**, and the effective sample size, computed as squared total weight divided by total squared weight, was **362** out of **1,723,359** Thompson rows. Weights over ten supplied **48.8%** of total weight. Just **41 rows** with weights above 1,000 supplied **23.0%**. The hour beginning November 28 at 00:00 UTC carried **20.9%** of all weight. Omitting that single hour moves the doubly robust estimate from **0.3609%** to **0.5228%**, against **0.5406%** observed in the remaining uniform traffic. This last calculation was selected after inspecting the weights. The primary result includes every evaluation hour. One week cannot establish a ranking of estimators. [Calculations](../external/forward/open_bandit.py)

## Reproduce

The two CSVs total about 1.49 GB. The runner verifies their SHA-256 hashes and processes them in chunks. From the repository root, run

```bash
audit=docs/audits/2026-11-19_time_series_dml
uv run --locked python "$audit/external/run.py" retail-recommendations
```
