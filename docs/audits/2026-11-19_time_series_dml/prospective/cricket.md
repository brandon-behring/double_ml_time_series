# Can a historical toss advantage predict the next year of international cricket?

**At 02:17:47 UTC on November 21, 2026, we issued a forecast of +0.732 percentage points for the effect of winning the right to choose innings order.** The half-history forecast is +0.658 points. A full year with the precision of 2025 would struggle to distinguish them. The primary cohort opens December 1, 2026 and runs through November 30, 2027, by local match date. All five outcomes are pending. [Frozen annual protocol](cricket-annual-frozen.json) · [Issuance record](cricket-issuance.json) · [Current score rows](cricket-pending.json)

## What does a toss identify?

Consider one senior international Twenty20 match with a contested fair toss. Fix the focal team as the alphabetically first of the two archived names before the toss. Let its win credit be one for winning, zero for losing and one-half for an unresolved tie or a post-toss no-result. The estimand averages, equally across eligible matches, how much the focal team's credit changes if it wins the right to choose innings order instead of its opponent winning that right. Under the fair-toss assumption the assignment probability is 0.5. How captains exercise the right is part of this effect. Identifying the causal effect of batting first itself would require additional assumptions about those strategic choices. Bhaskar studied the related choice in one-day cricket. T20 internationals provide a different population, period and prospective test. [MCC Laws of Cricket, Law 13](https://www.lords.org/getmedia/a4ef9f77-2a25-4f5b-a286-4601f08e6334/Laws-of-Cricket-2017-Code-4th-Edition-%282026%29_5.pdf) · [Bhaskar, *Rational Adversaries?*](https://academic.oup.com/ej/article-abstract/119/534/1/5089654)

The focal team's winner credit follows the recorded final winner, including an eliminator or bowl-out. An unresolved tie, draw or no-result after the toss gives each team 0.5. Pre-toss cancellations and uncontested tosses never created a randomized comparison. The protocol includes both genders, excludes youth teams and excludes fixtures involving Afghanistan to account for systematic source withholding. The archive README reports 166 withheld entries. In recent years ICC membership eligibility expanded and Cricsheet documents gaps, particularly in women's cricket. Both the set of future opponents and the representation of earlier matches can change the transported effect. [Cricsheet result and toss fields](https://cricsheet.org/format/json/) · [Cricsheet coverage](https://cricsheet.org/coverage/) · [ICC eligibility change](https://www.icc-cricket.com/news/all-t20-matches-between-icc-members-to-get-international-status)

## Forecasts and precision

| Forecast | Percentage points of win credit | Construction before issuance |
|---|---:|---|
| Zero | 0 | No average advantage |
| Historical | +1.31669 | Mean randomized contrast from 5,848 eligible archived matches |
| Half-history | +0.65834 | Exactly one-half the stored historical estimate |
| DML | +0.73160 | Cross-fitted augmented inverse-probability score with the known assignment probability |

For the historical estimate and the future benchmark, one match contributes +100 when the toss-winning team wins the match, −100 when it loses and 0 for equal credit. This equals `100 (2A − 1)(2Y − 1)` with focal assignment `A` and credit `Y`. The expected contribution for a fair toss equals the focal team's treatment effect in percentage points. Each completed target window estimates that effect anew. Predicting an individual match winner would answer a different question.

The historical DML model fits ridge regressions separately in the two toss arms. Features are gender-specific focal and opposing team names, venue, gender, calendar month and year. Five folds keep each event, season and gender together. One-hot encoding is fitted inside each training fold. Team strength can improve win-credit predictions even if the toss advantage remains uncertain. All fitting occurs within history. The forward test takes the same four issued numbers into each subsequent quarter without refitting. Randomization identifies a contemporaneous causal comparison, while forecasting its future magnitude also assumes sufficient stability across schedules and conditions. [Estimator implementation](../external/forward/cricket_data.py)

DML moved the historical estimate toward zero. The midpoint between DML and half-history is **+0.69497 points**. A later estimate above it has smaller squared error for DML, and one below it has smaller squared error for half-history. We transform the two endpoints of the later benchmark's 95% interval into the corresponding difference in squared errors. An interval spanning zero leaves the ranking unresolved. With either forecast equal to truth, 80% power for a two-sided 95% ranking would require SE **0.01307 points**. The 2025 historical design had **946 eligible matches in 158 series**, estimate +3.594 points and SE **3.020** (95% interval −2.371 to +9.559). Fixed 28-day blocks gave SE 2.634 across 14 blocks. The corresponding ordinary effect-versus-zero design calculation using the series SE is 8.46 points. These scales describe different questions. A year may expose a large change in advantage while giving little evidence about which of two near-identical forecasts is closer. [Precision calculations](cricket-precision.json) · [Ranking rule](scoring.md)

## When outcomes can be scored

| Window | Local dates, inclusive | Earliest scoring date, UTC | Role |
|---|---|---|---|
| Q1 | Dec 1, 2026 – Feb 28, 2027 | Mar 2, 2027 | Descriptive |
| Q2 | Mar 1 – May 31, 2027 | Jun 2, 2027 | Descriptive |
| Q3 | Jun 1 – Aug 31, 2027 | Sep 2, 2027 | Descriptive |
| Q4 | Sep 1 – Nov 30, 2027 | Dec 2, 2027 | Descriptive |
| Annual | Dec 1, 2026 – Nov 30, 2027 | Dec 2, 2027 | Primary |

The extra UTC day allows the last local match date to finish. The archive reader uses its dated README index to open only closed-window match payloads. Event/season/gender clusters supply the primary standard error, with a cluster-count degrees-of-freedom correction and Student t interval. Fixed calendar blocks of 28 days measured from January 1, 2000 supply a sensitivity interval. A single cluster yields only a point estimate. Opponents recur across series, and adjacent blocks can share shocks. The two intervals expose sensitivity to those choices. The overlapping annual and quarterly estimates cannot be counted as five independent replications.

## Fixture coverage and reconciliation

The ICC announced India–Sri Lanka men's T20Is on December 22 in Rajkot, December 24 in Cuttack and December 27 in Pune. The ECB announced England–Sri Lanka women's T20Is on February 1, 3 and 5 at Kandy. The six-row example ledger records the scheduled fixtures, source links and unresolved status at issuance. It is a starting list. Identifying coverage for a whole quarter also requires enumerating other members, including qualifiers and associate-team matches. [Example scheduled fixtures](cricket-fixture-examples.csv) · [ICC announcement](https://www.icc-cricket.com/news/india-announce-schedule-for-blockbuster-home-season) · [ECB announcement](https://www.ecb.co.uk/news/4596552/england-women-tour-to-sri-lanka-fixtures-announced)

A historical check shows why the fixture list matters. The pinned archive omits the February 21, 2026 New Zealand–Pakistan World Cup fixture. The ICC reports Pakistan elected to bat after the toss, then rain ended the match. Its observable contribution is zero and it belongs in the denominator. At scoring time an official source can supply such a match. The historical forecasts above retain the exact training extract that was issued. [ICC match report](https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/live-pakistan-take-on-new-zealand-in-super-eight-opener) · [ICC fixture 268112](https://www.icc-cricket.com/news/category/match-268112)

1. Enumerate fixtures for both genders in the ICC calendar and member-board schedules, including tie-breaks, washouts, rescheduling and cancellations. Check tournament fixture lists and official scorecards or match reports for unresolved entries. Use the actual local start date when teams played and the last scheduled local date for a cancellation. Preserve the original schedule and source evidence for date changes. [ICC fixtures and results](https://www.icc-cricket.com/fixtures-results)
2. Retain a stable `fixture_id` in the provider's namespace. If the announcement lacks an ID, combine its stable article ID/slug and the match number within the announced series. Enter the verified numeric Cricsheet ID as `match_id` after matching local date, teams, gender and, where needed, venue and match number. Keep the source URL. For a fixture absent from Cricsheet, use its namespaced fixture key in both columns, including cancellations that will never receive an archive ID. For example, ICC's 268117 corresponds to Cricsheet's 1512764 for New Zealand–Sri Lanka on February 25, 2026. [ICC fixture](https://www.icc-cricket.com/matches/268117/sri-lanka-vs-new-zealand) · [Historical crosswalk fields](cricket-history.csv.gz)
3. Supply CSV columns `fixture_id,match_id,date,gender,team1,team2,status,source`. Use ISO dates, `male` or `female`, and archive team names. Give a public HTTPS source for every row. Set `status` to `tossed`, `no_toss`, `cancelled_before_toss`, `uncontested`, `coverage_exclusion`, `youth` or `unresolved`. A missing toss field alone leaves eligibility unresolved. Only an independent account establishing that no contested toss occurred supports structural exclusion.
4. For an officially documented toss followed by a no-result, `tossed_no_result` also requires `toss_winner,event,season`. The event and season labels must match the tournament's Cricsheet grouping. The scorer recovers the observed zero contribution and counts it in its series and calendar block. For an independently documented played match, `tossed_official_result` requires those same fields plus `final_result`, the final winning team including after a tie-break, or `tie` or `draw` if unresolved. Enter each supplemental match under the same eligibility and evidence rules irrespective of who won. A tossed match with unverified results stays pending. The scorer reports unmatched IDs, population counts, the provisional covered-sample estimate and extreme bounds conditional on the roster. With conflicting matches or unmatched archive IDs it uses the full feasible range, −100 to +100 points. The bounds cannot account for fixtures absent from both sources.
5. Attest the last date through which the independent ledger covers every eligible fixture since December 1, 2026. Use an attestation only after reconciling the released schedules across the target population. The scorer checks each ID, date, gender and pair of teams and retains hashes of both inputs. A partial list without that attestation leaves the final benchmark pending, even if every listed match agrees with Cricsheet. Each annual observation again requires complete coverage across its whole period. A date attestation records the operator's claim, and its validity still depends on the supporting sources.

## Reproduction and provenance

The original ZIP contained 5,849 records through November 1, 2026, including one uncontested toss. The retained gzip CSV includes all 5,849 normalized records, starting February 17, 2005. It contains data derived from Cricsheet's JSON records, available under the Open Data Commons Attribution License 1.0. The archived match IDs, status, randomized assignment and derived outcome enable offline replay after the upstream ZIP updates. [Normalized match data](cricket-history.csv.gz) · [Cricsheet downloads](https://cricsheet.org/downloads/) · [Cricsheet attribution and license](https://cricsheet.org/licensing/) · [ODC-By 1.0](https://opendatacommons.org/licenses/by/1.0/)

| Artifact | SHA-256 |
|---|---|
| Raw historical ZIP, 23,330,741 bytes | `956f29df9d2b95dd6fe12de5bc70d4522f23693e5e7e2ac4c1a004e1b541ad8e` |
| Archived historical CSV gzip | `6ed9c19769f0a8430fbd3fd5d6762990e5ace58b66220809b4f811f6e20d876c` |
| Issuance registry covering five manifests | `079a1bb1c38fcb31f4d2b50e2e381a99485b4702913d2abf8c051d6e30f22885` |
| Annual frozen manifest | `e9c2acf2ef96df10c741fc440f18cf87fede061266e28a7dda76f51c29314ddc` |

From the repository root, `replay` refits the frozen forecasts in a temporary directory and checks the five resulting digests. To verify raw provenance, provide an original ZIP or point to a cache containing `cricsheet-t20s-<raw SHA-256>.zip`. The loader verifies the digest after fetching from the mutable upstream URL. If the source has advanced, replay from the archived CSV or supply a retained copy of the original ZIP. A digest checks exact bytes against a retained reference. Independently dating the issuance requires an externally retained timestamp or archive of that reference.

```sh
runner=docs/audits/2026-11-19_time_series_dml/external/forward/prospective_cricket.py
.venv/bin/python "$runner" --stage replay
.venv/bin/python "$runner" --stage verify-source --archive /path/to/original-t20s.zip
.venv/bin/python "$runner" --stage precision
.venv/bin/python "$runner" --stage score
```

Once the first quarter closes and its roster has been verified, the corresponding scoring invocation is

```sh
.venv/bin/python "$runner" --stage score --archive /path/to/results-snapshot.zip \
  --fixtures /path/to/complete-independent-ledger.csv --fixtures-complete-through 2027-02-28
```

Keep the dated result ZIP, fixture CSV, supporting schedules and emitted JSON together. Each score records both input digests and the presence or absence of a ranking interval. [Scoring adapter](../external/forward/prospective_cricket.py) · [Integration tests](../../../../test/integration/test_prospective_cricket.py)
