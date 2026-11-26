# Low Carbon London electricity pricing

Households enrolled in dynamic pricing used an average **1.112 kWh** during a three-hour high-price event and **1.321 kWh** during a comparable low-price event. The high-minus-low difference was **−0.209 kWh per household per event**. Its approximate 95% interval was **[−0.397, −0.020]**, using variation across the 48 event dates. The final quarter had nine event dates, but no start time received both prices. Its same-hour contrast cannot be estimated from that quarter. [Observed effects](effects.csv) · [Event dates and meter counts](events.csv)

## Assignment and measurement

The energy supplier announced a tariff schedule a day ahead. Participants opted into dynamic pricing, and every enrolled household faced the same schedule. The trial designer reports randomly distributing supply-following schedules through the year subject to calendar constraints. We excluded the 21 constraint-management dates in Table 4.5, defined the trial day as starting at 05:00 local time, and selected days with exactly six consecutive high or six consecutive low half-hours. Each of eight local start times then had three dates of each price, matching the design in Table 4.4. High was 67.20p/kWh and low was 3.99p/kWh. The contrast measures total use during those three hours among enrolled households. It leaves daily load shifting, tariff enrollment and the response of other London residents open. [Trial design, sections 4.2–4.8](https://spiral.imperial.ac.uk/entities/publication/4ad3c39f-8bf7-4a66-b832-5a189111e0fa) · [Supplier data and prices](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)

I processed all **167,932,474 records** in the 168-file public release. Its meter IDs identify **1,123 dynamic** and **4,443 standard-tariff households**. The supplier's landing page lists 5,567 households overall. Within an event, a household contributed its sum only when all six readings were present once. The primary clock converts local event times to GMT and joins each half-hour to its ending meter timestamp. This convention follows the trial user guide and the documentation for the public flat-tariff extract. The raw London Datastore fields omit a timezone. [Release receipt](run.json) · [Trial user guide, section 3.3](https://doc.ukdataservice.ac.uk/doc/7857/mrdoc/pdf/7857_userguide.pdf) · [Meter time convention](https://research.tudelft.nl/en/datasets/low-carbon-london-smart-meter-data-refactored/)

For each start hour, I subtracted the mean of the three low-price date means from the mean of the three high-price date means. Each date and each start hour gets equal weight. The SE uses within-arm date variances in each hour, combined with Welch–Satterthwaite degrees of freedom. It approximates uncertainty under within-hour allocation across dates. The public schedule gives realized assignments, while the designer's full constrained allocation algorithm and calendar probabilities are unavailable here. The reported interval is therefore approximate. Concurrent shocks can move the estimate, and observing incomplete windows can change the mix of households. The flat-price comparison and complete-panel sensitivity quantify those concerns. [Event-level inputs](events.csv) · [Analysis](replay.py)

| Population and timing | Households per date | High minus low, kWh | Date SE | Approximate 95% interval |
|---|---:|---:|---:|---:|
| All available dynamic households, GMT end | 1,041–1,111 | −0.209 | 0.087 | [−0.397, −0.020] |
| Dynamic households with complete readings for all 48 events | 982 | −0.214 | 0.085 | [−0.398, −0.031] |
| Standard-tariff households on the same dates | 4,058–4,396 | +0.055 | 0.091 | [−0.141, +0.250] |
| Dynamic households, incorrectly aligning start to start in GMT | 1,041–1,111 | −0.176 | 0.086 | [−0.361, +0.009] |

Using the raw local tariff labels as meter timestamps gives −0.161 kWh, with interval [−0.339, +0.017]. The date-level contrast retains its sign under these timing shifts. Its interval endpoint crosses zero under a plausible timestamp processing error. The flat group supplies a check on background load differences across assigned dates. Its households selected a different tariff and therefore cannot supply a randomized comparison of enrollment. [Full clock and cohort sensitivities](effects.csv)

## What rolling windows can establish

| Quarter | High dates | Low dates | Start hours observed at both prices, local time |
|---|---:|---:|---|
| Q1 | 9 | 5 | 05, 14, 23 |
| Q2 | 7 | 4 | 02, 11, 14, 20 |
| Q3 | 6 | 8 | 02, 05, 08, 20 |
| Q4 | 2 | 7 | None |

All eight start times contribute to the annual target. Each quarter misses at least one comparison, and restricting quarters to available hours would change the estimand across windows. In Q4, high-price events began at 02:00 and 20:00. Low-price events began at 05:00, 08:00, 11:00, 17:00 and 23:00. A model can transport an earlier time-of-day response into Q4, but Q4 offers no empirical same-hour check of that prediction. [Per-quarter allocation counts](run.json)

## Calibration on real untreated histories

I sampled from the **3,845** flat-price households with complete outcomes on all 48 dates. In each of 500 seeds I randomized prices anew by date within each of eight start hours, assigning half of the dates to each price. Subsamples of 32 and 48 dates and 32, 256 and 1,024 households used paired random draws. The planted high-minus-low effect was either zero or −0.18 kWh. For the latter, a low price added 0.03 kWh in each of six half-hours to its observed flat-price baseline. This experimental assignment is known exactly. It preserves household histories, calendar shocks and the baseline dispersion observed on the selected dates. [Draws](draws.csv.gz) · [Calibration results and Monte Carlo SEs](calibration.csv)

| Households | Dates | Measured sampling SD, kWh | Date-based 95% coverage | Coverage treating household-days as independent |
|---:|---:|---:|---:|---:|
| 32 | 32 | 0.122 | 96.2% | 87.4% |
| 256 | 32 | 0.104 | 96.8% | 46.2% |
| 1,024 | 32 | 0.103 | 96.2% | 25.6% |
| 32 | 48 | 0.101 | 94.4% | 86.2% |
| 256 | 48 | 0.085 | 94.8% | 44.2% |
| 1,024 | 48 | 0.085 | 94.2% | 25.2% |

At 1,024 households and 48 dates, the mean estimated SE was 0.086 kWh using assignment dates and 0.014 kWh treating household-days as independent. The respective Monte Carlo SEs for coverage were 1.0 and 1.9 percentage points. Increasing the household sample from 256 to 1,024 barely moved the sampling SD at either date count. Adding 16 assignment dates reduced it by about 0.018 kWh at 1,024 homes. The constant planted shift gives identical errors and interval coverage at both effect sizes by construction. [Measured draws and uncertainty](calibration.csv)

## Reproduction

Run from the repository root. A full replay with one worker reproduced the six-worker run byte for byte for all five result files, including the compressed simulation draws. The source files remain in the external cache. The script validates their SHA-256 digests, reads ZIP members in parallel and publishes aggregate outputs plus simulation draws. Allow about 760 MiB for the compressed meter archive.

```bash
uv run --locked python docs/audits/2026-11-19_time_series_dml/field/london/replay.py \
  --workers 4 --output /tmp/london-replay
```
