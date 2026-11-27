# Can six microcredit experiments predict the effect in a seventh location?

An equal-weight forecast using six countries missed the held-out business-profit contrast by **18.71 international dollars per household per two weeks** on average, compared with **19.72** for a zero-effect forecast. These are absolute differences from noisy randomized estimates. The seven comparisons use overlapping training sets. Their one-dollar difference gives little basis for preferring the six-site rule for the next country. [Meager’s study](https://doi.org/10.1257/app.20170299) · [Author analysis workspace](https://bitbucket.org/rmeager/aggregate-average-impacts-microcredit/src/be1fe9098d5a27bd5a3a9cf84686087f501b0e3b/)

| Held-out location | Complete assignment and profit rows | Local published estimate | Mean of six other estimates | Forecast minus local estimate (95% interval) |
|---|---:|---:|---:|---:|
| Mexico | 16,560 | 0.00 | 22.89 | +22.89 [0.53, 45.24] |
| Mongolia | 961 | −0.36 | 22.95 | +23.31 [2.32, 44.29] |
| Bosnia | 1,195 | 28.89 | 18.07 | −10.82 [−60.44, 38.81] |
| India | 6,863 | 15.34 | 20.33 | +4.99 [−28.65, 38.62] |
| Morocco | 5,498 | 17.89 | 19.90 | +2.01 [−27.57, 31.60] |
| Philippines | 1,113 | 66.46 | 11.81 | −54.65 [−166.02, 56.72] |
| Ethiopia | 3,113 | 9.10 | 21.37 | +12.28 [−13.37, 37.92] |

The inputs are the original trial estimates and standard errors that Meager standardized to purchasing-power-adjusted **2009 US dollars per two weeks**. They retain the published trials’ individual or group designs and specifications. The executable also reads the 40,267-row source workspace and computes unadjusted differences as a cross-check. Those differ from some published specifications. In Mexico the complete-case difference is −4.55, while the standardized reported estimate is zero. [Standardization code and seven source studies](https://bitbucket.org/rmeager/aggregate-average-impacts-microcredit/src/be1fe9098d5a27bd5a3a9cf84686087f501b0e3b/import_organise_data_v7.R) · [Extraction](../external/forward/microcredit.R)

Each target is excluded from model construction. Signed error is the equal-country forecast minus the held-out estimate. For seven fixed, independently studied sites, its estimated sampling variance equals the target variance plus the sum of the other six variances divided by 36. The table uses pointwise normal intervals. Variation in programs, borrower selection, lending products, measurement horizons and settings still matters when considering a new country. The original paper used hierarchical models and investigated how omitting sites changed pooled estimates. This exercise predicts the omitted contrast itself. [Original appendix](https://www.aeaweb.org/content/file?id=8597) · [Holdout calculation](../external/forward/microcredit.py)

This benchmark is geographic. The retrospective analyst can access all seven contrasts in the author workspace. Holding out each country does not establish that a trial operator or a public analyst could have obtained the other six results at its start date. The results therefore measure geographic transfer only. [Replication deposit](https://doi.org/10.3886/E116357V1)
