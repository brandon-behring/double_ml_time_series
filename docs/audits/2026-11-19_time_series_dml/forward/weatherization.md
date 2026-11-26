# What can public records verify about Michigan weatherization?

Randomly offering help with Michigan's Weatherization Assistance Program raised applications by **12.58 percentage points** and completed retrofits by **4.93 points**. The public files also recover the engineering model's average prediction of **$10,611** in discounted energy-bill savings. Measuring how much energy households actually saved requires monthly utility bills covered by the authors' data agreements. [Fowlie, Greenstone and Wolfram, published paper](https://epic.uchicago.edu/wp-content/uploads/sites/5/2019/07/Publication-1-1.pdf) · [Author data and code](https://doi.org/10.7910/DVN/OJFYSR)

## The randomized offer

Table III compares **7,549 encouraged households** and **21,339 controls**. Each effect is the difference between the groups' participation rates. All three differences and control rates match the published values at their printed precision. The last column uses an additional normal approximation with each arm's own household-level variance. [Original Stata analysis](https://dataverse.harvard.edu/api/access/datafile/3107603?format=original) · [Reproduction](../external/forward/weatherization.py)

| Outcome | Control rate | Encouraged rate | Difference, percentage points | Independent-arm 95% interval, points |
|---|---:|---:|---:|---:|
| Submitted application | 2.34% | 14.92% | +12.58 | [11.75, 13.41] |
| Received efficiency audit | 1.07% | 6.35% | +5.27 | [4.70, 5.84] |
| Completed weatherization | 0.84% | 5.78% | +4.93 | [4.39, 5.47] |

The original trial identified 34,161 presumptively eligible households. Of 30,515 rows in the released status file, 28,890 carry the experimental sample flag. Applying the authors' eligibility and billing-history filters leaves 28,888 distinct household identifiers. Two control records excluded by the filters share identifiers with retained households. Missing audit indicators are coded as zero in the authors' program. The ordinary least-squares standard errors used for their table are 0.300, 0.205 and 0.191 percentage points. [Household data](https://dataverse.harvard.edu/api/access/datafile/3107601?format=original) · [Table III and original trial population](https://doi.org/10.1093/qje/qjy005)

## Projected and realized energy savings

The Appendix Table A.1 program selects **1,638** weatherized records with an audit identifier. The number of observed values varies by measure. The mean discounted bill-savings projection is **$10,611.02** across **1,619** records using the authors' 3% discount rate. Projected annual heating savings average **50.43 million Btu (MMBtu)** across **1,636** records, and recorded installation costs average **$5,154.54** across **1,449**. Households enter these summaries after completing a retrofit. [Engineering records](https://dataverse.harvard.edu/api/access/datafile/3107601?format=original) · [Appendix and published paper](https://michaelgreenstone.com/research/do-energy-efficiency-investments-deliver-evidence-from-the-weatherization-assistance-program/)

For households induced to participate by the offer, the paper reports **56 MMBtu** in projected annual total energy savings versus **17 MMBtu** estimated from actual bills, about **31%** of the projection. This experimental comparison uses encouragement as an instrument for retrofitting and assumes, among other things, that encouragement affects energy use only through participation. It targets total energy in a different population from the heating-only summary above. The authors' Table IV program reads the restricted experimental billing panel `RED_est.dta`. Table V reads the restricted applicant billing panel `QUASI_cmb_est.dta`. The replay checks both program inputs and their status in the authors' access table against the archive manifest. The 17 MMBtu figure is a published estimate. [Published comparison, pp. 1623–24](https://epic.uchicago.edu/wp-content/uploads/sites/5/2019/07/Publication-1-1.pdf) · [Replication instructions and data permissions](https://dataverse.harvard.edu/api/access/datafile/3107604?format=original)

**Chronology.** The operator generated engineering predictions during audits before installation. The encouragement campaign ran from March through May 2011, enrollment assistance continued through February 2012, and the billing histories cover June 2008 through May 2014. The public status file combines households from one randomized campaign and provides final participation indicators. A public analyst can replay that historical comparison and the archived projections. A new forward test of realized savings would require dated forecasts, linked bills and a later intervention or cohort with a defensible comparison group. The working paper appeared in July 2015, and the version of the replication archive used here was released in July 2018. [Study design and billing period](https://doi.org/10.1093/qje/qjy005) · [Working-paper date](https://www.nber.org/papers/w21331) · [Archive version](https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId=doi:10.7910/DVN/OJFYSR)

Run from the repository root. The command verifies the SHA-256 of each downloaded input and reuses verified copies.

```bash
uv run --locked python docs/audits/2026-11-19_time_series_dml/external/forward/weatherization.py --cache-dir /tmp/dml-external-bench-cache
```
