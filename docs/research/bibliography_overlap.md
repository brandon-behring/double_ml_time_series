# Bibliography Overlap Report
**Generated:** 2026-05-16.
Compares the manuscript's `bibliography.bib` (34 entries) against the dossier's `bib_ledger.yml` (159 entries). Produced by Step 11 of the master research-toolkit pipeline.
**Migration priority scoring** (three-factor, user-confirmed 2026-05-16):
- **Canonical-ness** (1-3): landmark / widely-cited / pre-known anchor paper?
- **URL-recoverability** (1-3): primary URL findable or fixable?
- **Sub-area fit** (1-3): clarity of fit to research_plan.md's taxonomy?
- **Sum** = 3-9. High-priority (≥7) entries seed Step 12 pass-2 gather AND are flagged for `bibliography.bib` migration. Lower-priority (3-6) entries are listed for manuscript migration but NOT seeded.
- **Out-of-scope** entries (textbooks, non-paper references) are listed separately and never seeded under the strict pass-2 rule.
---
## Summary
- Total `bibliography.bib` entries: 34
- Already covered in dossier (overlap): **23**
- Overlooked but in scope, **HIGH priority (seed pass-2)**: **4**
- Overlooked but in scope, low priority (manuscript-migrate only): **0**
- Out of scope (textbooks / data references): **7**
---
## Overlap table (entries already in dossier)
| `bibliography.bib` bibkey | dossier bibkey | sub-area | match basis |
|---|---|---|---|
| `athey2016recursive` | `athey2016recursive` | estimator_xs | exact bibkey match |
| `athey2019machine` | `athey2019machine` | survey | exact bibkey match |
| `bach2022doubleml` | `bach2022doubleml` | tooling | exact bibkey match |
| `bojinov2019time` | `bojinov2019time` | estimator_temporal | exact bibkey match |
| `chernozhukov2017double` | `chernozhukov2017double` | theory_orthogonality | exact bibkey match |
| `chernozhukov2018double` | `chernozhukov2018double` | theory_orthogonality | exact bibkey match |
| `chernozhukov2022locally` | `chernozhukov2022locally` | theory_orthogonality | exact bibkey match |
| `dePrado2018` | `deprado2018advances` | inference_ts | title match → deprado2018advances |
| `dehejia1999causal` | `dehejia1999causal` | replication_applied | exact bibkey match |
| `dowhy2022` | `dowhy2022` | tooling | exact bibkey match |
| `econml` | `econml` | tooling | exact bibkey match |
| `fred` | `fred` | replication_applied | exact bibkey match |
| `frisch1933partial` | `frisch1933partial` | theory_partialing | exact bibkey match |
| `imai2023should` | `imai2021should` | estimator_temporal | title match → imai2021should |
| `imbens2003sensitivity` | `imbens2003sensitivity` | replication_applied | exact bibkey match |
| `lalonde1986evaluating` | `lalonde1986evaluating` | replication_applied | exact bibkey match |
| `lovell1963seasonal` | `lovell1963seasonal` | theory_partialing | exact bibkey match |
| `newey1987` | `newey1987simple` | inference_ts | title match → newey1987simple |
| `neyman1959optimal` | `neyman1959optimal` | theory_orthogonality | exact bibkey match |
| `nie2021quasi` | `nie2021quasi` | estimator_xs | exact bibkey match |
| `rosenbaum2002observational` | `diamond2013genetic` | replication_applied | partial title match → diamond2013genetic |
| `van2000asymptotic` | `vandervaart2000asymptotic` | theory_orthogonality | title match → vandervaart2000asymptotic |
| `wager2018estimation` | `wager2018estimation` | estimator_xs | exact bibkey match |
---
## HIGH-PRIORITY overlooked entries (seed Step 12 pass-2)
These entries fit the research_plan taxonomy AND score ≥7 on the migration-priority sum. They will be seeded into Step 12's `/research-gather` pass-2 with `status: seed`.
| `bibliography.bib` bibkey | sub-area fit | canonical | URL-recov | fit | sum | rationale |
|---|---|---|---|---|---|---|
| `cinelli2020omitted` | inference_ts | 3 | 3 | 3 | **9** | Making Sense of Sensitivity: Extending Omitted Variable Bias |
| `rubin1974estimating` | theory_orthogonality | 3 | 2 | 3 | **8** | Estimating causal effects of treatments in randomized and nonrandomized studies |
| `holland1986statistics` | theory_orthogonality | 3 | 2 | 3 | **8** | Statistics and causal inference |
| `vanderweele2017evalue` | inference_ts | 2 | 3 | 2 | **7** | Sensitivity Analysis in Observational Research: Introducing the E-Value |
---
## Out-of-scope (textbooks / data references — never seeded)
Per the strict pass-2 rule, textbooks and non-paper references do NOT enter the dossier. They are listed here for completeness so the user can see what was excluded.
| `bibliography.bib` bibkey | type | rationale |
|---|---|---|
| `angrist2009mostly` | textbook / pedagogy | Mostly harmless econometrics: An empiricist's companion |
| `pearl2009causality` | textbook / pedagogy | Causality: Models, reasoning, and inference |
| `imbens2015causal` | textbook / pedagogy | Causal inference for statistics, social, and biomedical sciences: An introductio... |
| `hernan2020causal` | textbook / pedagogy | Causal inference: What if |
| `rosenbaum2010design` | textbook / pedagogy | Design of Observational Studies |
| `facure2022causal` | textbook / pedagogy | Causal inference for the brave and true |
| `huntington2021effect` | textbook / pedagogy | The effect: An introduction to research design and causality |
---
## Notes for pass-2 (Step 12)
The following bibkeys will be seeded as `status: seed` entries in `bib_ledger.yml` before re-running `/research-gather` in append mode: `cinelli2020omitted`, `rubin1974estimating`, `holland1986statistics`, `vanderweele2017evalue`.
Additionally, the 11 high-impact migration suggestions from `bibliography_audit.md` (Step 0) are already covered by pass-1 of the dossier — they appear in the overlap table above for confirmation. The strict pass-2 rule prevents textbook entries (Pearl 2009, Imbens-Rubin 2015, Hernan-Robins 2020, etc.) from entering the dossier; the user can still migrate URLs / DOIs in those entries during the manuscript revision.
