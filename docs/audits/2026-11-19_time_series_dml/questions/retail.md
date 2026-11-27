# Do orange-juice discounts borrow sales from the following week?

**The discounted brand sold more in the cut week, but the adjusted data do not resolve a next-week decline.** Relative to store–brand weeks without a cut of this size, the same-week adjusted difference was **+35,560 fluid ounces** (95% interval +25,344 to +45,776). The following-week difference was **+1,348 ounces** (−6,855 to +9,552). Sales of the other two brands fell during the cut week. A household's decision to stockpile cannot be distinguished here from substitution between brands or changes in store traffic. [Original research question](../../../archive/superseded_2026-05-02/DATA_STRATEGY_REPORT.md#5-proposed-demo-notebooks) · [Analysis](../external/questions/retail.py) · [Hendel and Nevo on the post-promotion dip](https://bpb-us-w2.wpmucdn.com/web.sas.upenn.edu/dist/3/499/files/2018/09/Dip_QME-1ac5n86.pdf)

## What the price cut and volume mean

A cut is a shelf price at least 10% below the previous calendar week's price for the same brand in the same store. This definition uses the information available in the cut week. It allows the discount to continue into the following week. Comparison observations include smaller reductions, unchanged prices and price increases. The analysis keeps complete sequences from three weeks before through two weeks after the cut, so each outcome uses the same origins. Missing calendar weeks stay missing.

The historical scanner study aggregates UPC movements in fluid ounces. We recover that scale as `exp(logmove)`. All 28,947 observations in the pinned teaching extract lie within 0.00222 ounces of a multiple of 64 after reversing the rounded logarithm. The same-week own-brand estimate is equivalent to about 556 64-ounce containers. Each store-week contributes one row per focal brand. “Rivals” sums movement of the other two observed brands. “Three-brand total” omits the other varieties in Montgomery's eleven-product study. These are changes in purchase volume per focal brand and store, with no shopper identities or household consumption records. [Montgomery’s data-construction appendix, p. 1](https://www.andrew.cmu.edu/user/alm3/papers/micromarketing%20appendix.pdf) · [Kilts Center scanner data](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) · [Pinned three-brand extract](https://github.com/gchoi/Dataset/blob/8e81d46872564218a78be95d53a9c6bd3b00b199/oj.csv)

## Sales on identical scored weeks

The estimates below are in **thousands of fluid ounces**. Each weekly outcome subtracts the same brand–store's sales in the week before the cut, or the corresponding rivals' or three-brand total. The placebo is the change from two weeks before to one week before. The cumulative outcome sums the three weekly changes from the common baseline. Its uncertainty comes from the joint three-week outcome. Parentheses show pointwise 95% intervals.

| Outcome window | Own brand, unadjusted | Own brand, adjusted | Rivals, adjusted | Three-brand total, adjusted |
|---|---:|---:|---:|---:|
| Prior change, placebo | −8.00 (−16.67, 0.66) | −5.95 (−16.93, 5.03) | +0.60 (−9.03, 10.24) | −5.34 (−16.18, 5.50) |
| Cut week | +39.04 (28.39, 49.69) | +35.56 (25.34, 45.78) | −12.41 (−20.29, −4.52) | +23.15 (11.25, 35.06) |
| One week later | +9.41 (5.25, 13.58) | +1.35 (−6.86, 9.55) | −7.14 (−16.75, 2.48) | −5.79 (−17.44, 5.86) |
| Two weeks later | +4.17 (−0.42, 8.77) | −1.42 (−9.25, 6.40) | −2.04 (−10.35, 6.27) | −3.46 (−14.13, 7.21) |
| Three-week cumulative | +52.63 (39.63, 65.63) | +35.48 (15.50, 55.47) | −21.58 (−42.45, −0.72) | +13.90 (−12.51, 40.31) |

Both columns for the focal brand score **20,436 store–brand–week origins from 83 stores in weeks 64–158**. There were 4,424 cuts and 16,012 comparison records. All three brands in a store–week share a fold. Five expanding folds hold out entire weeks and keep each training origin's last outcome strictly before the first held-out week. The earliest held-out week in each fold is 64, 83, 102, 121 and 140, with 954, 1,002, 633, 886 and 949 cuts respectively. Week-level influence scores sum over stores and brands, preserving dependence among observations from the same store–week and common promotions. Bartlett HAC with four calendar lags also allows serial dependence across nearby weeks. Longer-lived shared demand shocks could change the intervals. [Analysis](../external/questions/retail.py)

The unadjusted column is a difference of means and is independently checked against the repository’s residual regression. Both methods use the repository’s influence scores for inference. The adjusted column residualizes both the binary cut and each volume change using fold-specific standardized ridge regressions (penalty 10). Covariates are prior-week log price and featured-ad status, log sales three weeks earlier, annual sine and cosine, and store and brand indicators. Treatment prediction had out-of-fold R² of −0.109. The adjusted number is a partially linear projection. Even with adequate adjustment and exogenous timing, heterogeneous effects would be weighted by conditional treatment variance, placing more weight on stores and weeks where either price decision was plausible. Identical scored rows make the estimators comparable while leaving these weighting differences intact. Chain pricing, anticipated demand, current displays and inventory availability can affect both the decision and sales. [Chapter 4’s price interpretation](../../../../chapters/chapter_04.tex)

Of the 4,424 cuts, 1,733 coincided with a cut in at least one other brand at that store. Repeating the fit on origins with no simultaneous rival cut retains 12,483 rows and 2,691 cuts. The same fold boundaries cover 93 observed origin weeks across the same 95-week calendar span. The adjusted own-brand difference one week later becomes **−3.02 thousand ounces** (−16.72 to +10.69). The three-week total across brands becomes −29.27 thousand ounces (−88.57 to +30.04). Simultaneous promotions and the resulting comparison population matter enough that the data support a narrower claim about the immediate own-brand increase.

## Why an apparent next-week dip appears in completed discounts

A retrospective rule also requires next week's price to return to at least 98% of the pre-cut price. That rule identifies 2,441 episodes, of which **2,185** have every calendar week from two before through two after. The following figures average each episode's difference in log volume from its own week immediately before the cut. The full sample of completed episodes is used, without a comparison group.

| Week relative to cut | Own brand | Three-brand total |
|---|---:|---:|
| Two weeks before | +0.3004 | +0.2329 |
| Cut week | +1.5888 | +0.5218 |
| One week later | −0.3187 | −0.0210 |
| Two weeks later | +0.2126 | +0.1893 |

The brand-level dip is visible under this selection rule, and the dip nearly disappears when the three brands are summed. Sales also differed before the intervention window. Selecting episodes by next week's price, changing the outcome scale and omitting comparable unexposed weeks all change the question answered by this calculation. Neither result identifies a causal effect on household inventories.

## Reproduce

The CLI verifies SHA-256 `01b8c698a0511231cf09424cf946fe849fb902639e4584023d6040955701c522` against the pinned Chapter 4 extract. It emits estimates in unscaled fluid ounces, including both estimators for all outcomes, fold support and the retrospective log-volume comparison. Run from the repository root with the project Python environment. [Runnable analysis](../external/questions/retail.py) · [Regression and calendar tests](../../../../test/integration/test_questions_energy_retail.py)

```sh
python docs/audits/2026-11-19_time_series_dml/external/questions/retail.py --cache-dir /tmp/dml-external-data
python -m pytest -q -m 'not network' test/integration/test_questions_energy_retail.py
```
