# What the external controls can detect

The four replays examine different reasons an estimate can change over time. The effort experiment repeats the assignment with new people. Sierra Leone follows one randomized rollout over two surveys. The nudge archive tests transfer across later trials. The retail archive lets us inspect assignment at both policy and item level. The scripts read checksum-pinned public inputs. [Computations](../external/forward/rolling_controls.py) · [Machine-readable estimates and diagnostics](external-controls.json)

## Stable rankings in a repeated effort experiment

DellaVigna and Pope randomized MTurk workers to variants of the same ten-minute button-pressing task in 2015 and 2018. For the 15 arms present in both experiments, the Spearman correlation computed from Table 1's paired arm means is **0.907**, which rounds to the published **0.91**. Subtracting a common control mean from every arm within a wave gives the same rankings. The authors estimated that sampling alone would yield a rank correlation of **0.93** under full stability using the Table 2 bootstrap. Our replay reconstructs the observed correlation from the published means and uses the published bootstrap reference. Rank stability still permits changes in average effect size. [Original paper, Tables 1–2](https://devingpope.com/assets/files/Website_Stability.pdf)

## A measurable decline in one randomized rollout

We reconstructed the indices from the 2009 and 2016 village records for the same 236 Sierra Leone communities. The development infrastructure and economic activity index retained a positive treatment effect in 2016. Its decline from 2009 is measurable on both the main and common-item definitions. Pairing villages preserves the covariance between the waves. The institutional index leaves a much wider range of changes plausible relative to the estimated decline. [Published study](https://edepot.wur.nl/638529) · [Versioned data, questionnaires and author code](https://zenodo.org/records/7411565)

| Outcome and construction | Components in 2009 / 2016 | 2009 effect | 2016 effect | Paired change (HC1 SE) |
|---|---:|---:|---:|---:|
| Development, Table 1 | 39 / 30 | +0.298 | +0.204 | −0.094 (0.036) |
| Development, common items | 29 / 29 | +0.352 | +0.208 | −0.144 (0.037) |
| Institutions, Appendix Table A1 | 44 / 57 | +0.085 | +0.065 | −0.020 (0.033) |
| Institutions, common items | 57 / 57 | +0.077 | +0.065 | −0.013 (0.031) |

The Appendix uses a different selection of institutional items in its 2009 index. The common-item sensitivity gives each distinct later-wave component equal weight in both years. Every column still uses the respective wave's control-group mean and standard deviation for each item, and the authors' arm-mean missing-item imputation. Changes therefore measure persistence in wave-specific units. The adjusted regressions include ward fixed effects, road distance and village size. Continued program activity also forms part of the observed assignment history. [Raw reconstruction](../external/forward/sierra_indices.py)

## Forecasts across annual portfolios of nudge trials

Across 241 arms in 126 trials, the arm-average effect on take-up is 1.390 percentage points, reproducing the study's rounded 1.4. For evaluation we first average arms within a trial, so a trial with multiple variants receives one vote. At each date the forecast is the mean of all previous trial effects. Each individual later trial supplies a noisy randomized benchmark. The archive gives trial year. The calculation assumes the preceding year's results were available at the start of the next year. [Study](https://doi.org/10.3982/ECTA18709) · [Data and programs](https://www.econometricsociety.org/publications/econometrica/2022/01/01/rcts-scale-comprehensive-evidence-two-nudge-units/supp/18709_Data_and_Programs.zip)

| Later year | Earlier / later trials | Historical forecast (pp) | Later mean (pp) | Zero MSE (pp²) | Historical MSE (pp²) | Advantage over zero (pp²) |
|---|---:|---:|---:|---:|---:|---:|
| 2016 | 23 / 26 | +3.070 | +2.532 | 25.642 | 19.517 | +6.124 |
| 2017 | 49 / 61 | +2.785 | +1.082 | 7.132 | 8.862 | −1.729 |
| 2018 | 110 / 16 | +1.840 | −0.355 | 16.136 | 20.831 | −4.695 |

The historical mean improved squared loss in 2016. Across all 103 later trials its trial-weighted advantage was **−0.207 pp²**. The interventions and participants differ across trials. Three annual transitions give limited evidence about performance in a future year. [Losses, including signed and absolute errors](external-controls.json)

## Retail assignment and support

In the men's campaign, the earlier randomized policy advantage forecast **+0.1638 pp** and the later direct randomized difference was **+0.1401 pp**. The forecast error was **+0.0238 pp**, with a 95% interval of **−0.0527 to +0.1003 pp** using the original six-hour Bartlett span. Separately, a concurrent doubly robust estimate of the uniform policy's click rate from Thompson traffic was **0.3609%**, versus **0.5417%** measured on uniform traffic. [Original analysis](../forward/retail-recommendations.md) · [Pinned logs](https://huggingface.co/datasets/zozonext/open-bandit/tree/90a1283aecd19026caf15a0840af1176a41f6a76)

| Consecutive records within a UTC hour, middle slot | Adjacent pairs | Repeated label | Expected after permuting within hour |
|---|---:|---:|---:|
| Item in uniform traffic | 150,840 | 38.879% | 3.620% |
| Item in Thompson traffic | 1,359,315 | 41.593% | 16.298% |
| Policy arm in combined traffic | 1,510,323 | 82.000% | 82.004% |

All three positions show excess item repetition. The expectation holds each hour's label counts fixed. Tied timestamps preserve file order within an arm, with uniform records first across arms. Dropping tied neighboring pairs from the middle-slot uniform calculation gives 38.866%. The policy-arm sequence agrees closely with the permutation expectation on this diagnostic. Item-level clustering can affect precision even when marginal logging probabilities are correct. [Ordering checks for all positions](external-controls.json)

The **41** later rows with weight above 1,000 supplied **23.00%** of the total weight and occurred in three item–slot–hour cells. One cell held 15 displays of item 22, each with recorded probability 0.000001, within **14.53 seconds**. The other two cells held 19 and seven displays. Across 1,723,359 later Thompson rows the importance-weight effective sample size was **362**. The public logs give selected-item probabilities and do not identify the underlying assignment refreshes. These diagnostics explain why the off-policy comparison is sensitive to a few short blocks. [Authors' logging documentation](https://github.com/st-tech/zr-obp/blob/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/docs/about.rst#L18-L36) · [Assignment diagnostic implementation](../external/forward/retail_assignments.py)

## Reproduce

Read the four source packages and print every estimate with the command below. The first run needs network access and Poppler's `pdftotext`. Subsequent runs check the cached file digests. The PDF, two ZIPs and retail CSVs remain in the external cache.

```bash
uv run --locked python docs/audits/2026-11-19_time_series_dml/external/forward/rolling_controls.py \
  --cache-dir /tmp/dml-external-bench-cache
```
