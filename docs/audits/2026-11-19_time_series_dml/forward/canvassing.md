# Did New Haven's published canvassing effect predict turnout in six later elections?

The 1999 paper reported turnout of 46.88% among 4,509 voters assigned to a canvass attempt and 44.63% among 23,921 controls. A public analyst could therefore freeze a **+2.25 percentage-point** prediction for 2001. The later pooled randomized effect was **+2.10 points**. Its pooled forecast error of +0.15 points conceals city-specific errors from −2.39 to +2.18 points. The public forecast was closer than zero in five of the six cities. [1999 paper, Table 1](https://www.uvm.edu/~dguber/POLS234/articles/gerber.pdf) · [2001 experiments, Table 2](https://sites.temple.edu/nickerson/files/2017/07/Green_Gerber_Nickerson.JOP_.2003.pdf) · [Analysis](../external/forward/canvassing.py)

| November 6, 2001 election | Focal voters | Randomized turnout effect, pp | Public forecast minus observed, pp (95% interval) | Corrected-data forecast minus observed, pp |
|---|---:|---:|---:|---:|
| Bridgeport | 1,806 | +4.04 | −1.79 [−4.78, +1.20] | −1.21 |
| Columbus | 2,478 | +1.38 | +0.87 [−1.36, +3.10] | +1.45 |
| Detroit | 4,954 | +2.42 | −0.17 [−2.91, +2.57] | +0.41 |
| Minneapolis | 2,827 | +1.87 | +0.38 [−2.79, +3.55] | +0.96 |
| Raleigh | 4,660 | +0.07 | +2.18 [−0.59, +4.95] | +2.75 |
| St. Paul | 2,208 | +4.64 | −2.39 [−6.43, +1.64] | −1.81 |
| All sites, walk-list fixed effects | 18,933 | +2.10 | +0.15 [−1.11, +1.41] | +0.73 |

The contrast concerns **assignment to a canvass attempt**. The 1999 estimate of +6.33 points conditional on actual contact has a different denominator. The later archive contains one focal voter per household and randomizes within walk lists. The reproduction fits a common within-list slope in each city and across all sites, checking column 2 of Table 2 and its original homoskedastic SEs to printed precision. It also checks column 1 using turnout in the prior year's election, a negative-control outcome. The largest pre-assignment difference is +3.17 points in St. Paul. Two of 195 lists have no within-list treatment variation. The table's new error intervals use HC1 standard errors and condition on the numeric forecast already published. They quantify sampling uncertainty for these experiments. [Six-city data and design](https://doi.org/10.60600/YU/B9TZNZ)

**Operator and public clocks.** The New Haven operators had the voter files before the six later elections. Their pre-2001 public summary supplies the same directly documented forecast to an outside analyst. Subsequent reconciliation changed assignment and linkage records. The 2005 release gives a retrospective estimate of **+2.83 points**, household-clustered SE **0.83**, on 29,811 voters with observed turnout in 22,449 households. It controls for household size and the independently assigned mail, telephone and placebo groups. The unadjusted corrected-data contrast is +2.73 points. The public tabulation and corrected regression differ in analysis sample and specification as well as source vintage. The pooled corrected prediction minus the later contrast is +0.73 [−1.32, +2.78] allowing sampling variation in both experiments. [Authors' correction](https://isps.yale.edu/resource/correction-to-gerber-and-green-2000-replication-of-disputed-findings-and-reply-to-imai) · [Corrected data](https://doi.org/10.60600/YU/ODL44F)

The six cities voted on the same date. Calendar time, the type of election, local operations and eligibility change together between 1998 and 2001. The six comparisons measure how the estimate travels across those combined changes. Raleigh and St. Paul limited eligibility to households with multiple registered voters. Individual intervals are pointwise.

Run `uv run --locked python docs/audits/2026-11-19_time_series_dml/external/forward/canvassing.py --cache-dir /tmp/dml-external-bench-cache` from the repository root. The original PDF table extraction requires `pdftotext`.
