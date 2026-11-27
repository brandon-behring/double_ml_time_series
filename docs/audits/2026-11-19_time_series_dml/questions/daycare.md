# Did a fine deter late day-care pickups?

**Late pickups increased after the fine was introduced.** Compared with the four centers without a fine, the six fined centers had **9.22 more late pickups per center-week** during the fine, after subtracting their baseline difference. The corresponding change was **+12.46** in the four reported weeks after repeal. For each phase, only the observed assignment among all 210 possible assignments of six centers produced an absolute contrast at least this large. The evidence describes the ten centers with complete records. [Gneezy and Rustichini, Table 1 and design footnotes](https://doi.org/10.1086/468061) · [Executable reanalysis](../external/questions/daycare.py)

## What changed at the centers

Parents at the six centers could be charged 10 shekels per child for a late pickup. The study observed all ten centers during four baseline weeks, twelve fine weeks and four weeks after the fine was removed. The outcome is a weekly **count**, which can include repeat pickups by the same parent. Two of twelve centers originally considered had incomplete records and were excluded from the published table. The analysis gives each retained center equal weight. [Study and original count table](https://raw.githubusercontent.com/OpenIntroStat/openintro/001a16cb7f599472c22b61331060bb1981d0cc0f/data-raw/daycare_fines/fine.pdf)

| Comparison in pickups per center-week | During the fine | After repeal |
|---|---:|---:|
| Fined centers, weekly mean | 16.44 | 18.71 |
| Other centers, weekly mean | 9.23 | 8.25 |
| Difference between arms in that phase | +7.22 | +10.46 |
| Difference after subtracting the baseline arm gap of −2.00 | **+9.22** | **+12.46** |
| 95% interval from exact tests of a common additive effect | [5.73, 12.90] | [7.38, 17.88] |
| Adjusted for each center's baseline mean using FWL | +9.66 | +13.31 |
| Model-based 95% interval using HC3 and seven residual degrees of freedom | [1.97, 17.35] | [2.11, 24.51] |

## What the framework adds

We recovered all 200 published count cells directly from the paper. Table 2 checks for center 1 give 7.25 pickups before the fine, 9.50 during the first four fine weeks and 15.25 after repeal. We calculated the change from baseline once per center. The primary test compares the observed difference in mean changes against complete reassignment of six labels among ten centers under the sharp null of no assignment effect. Each two-sided *p* is 1/210 (0.00476). The intervals invert that same test under a constant additive effect on the center's phase average. They are phase-specific. The regression uses the same centers and periods and a single covariate available before assignment. Independent OLS verifies the repository’s Frisch–Waugh–Lovell estimate. [Calculation and tests](../../../../test/integration/test_question_daycare.py)

Week 11 contained only four study days. The authors multiplied its count by 5/4 before publication. Omitting that week reduces the during-fine contrast from +9.22 to **+8.98**. A holiday after observed week 14 means the 20 reported weeks span 21 calendar weeks. Phase means use observed study weeks. A distributed-lag analysis would need to represent the holiday explicitly.

The ten assignment units support an exact comparison and one simple adjustment. Flexible nuisance fits would divide ten centers across training and evaluation, leaving little independent information for estimating a new response pattern. The persistent difference after repeal is a consequence associated with **previous assignment**. Every fined center stopped charging at the same time, so the experiment gives no same-period comparison with centers that continued charging. Recorded counts also cannot distinguish a change in social expectations from parents learning that late pickup was possible.

## Reproduce

The script uses Poppler’s `pdftotext` and the pinned PDF in the OpenIntro maintainers’ data archive. Its SHA-256 is `06b583380fc9bf1c876747acea4ba92952a57f6253add3bacd8e4a51cf2ef022`. The PDF and OpenIntro’s public transcription provided an independent 200-cell cross-check. [Data archive](https://github.com/OpenIntroStat/openintro/tree/001a16cb7f599472c22b61331060bb1981d0cc0f/data-raw/daycare_fines) · [OpenIntro transcription](https://www.openintro.org/data/index.php?data=daycare_fines)

```bash
uv run --locked python docs/audits/2026-11-19_time_series_dml/external/run.py daycare-fines
```
