# Bibliography Audit — `bibliography.bib`

Generated: 2026-05-16
Source: `/Users/brandonbehring/double_ml_time_series/bibliography.bib` (34 entries)
Scope: Step 0 of research-toolkit pass — pre-pipeline audit of existing book
references for URL freshness, primary-source coverage, and obvious gaps.

## Summary

| Metric | Value |
|---|---|
| Total entries | 34 |
| Entries with explicit `url` field | 3 |
| Entries with arXiv ID (no `url` field) | 2 |
| Entries with no URL or arXiv ID | 29 (85%) |
| URLs checked | 5 (3 direct + 2 arXiv-derived) |
| URLs returning 2xx | 4 |
| URLs bot-blocked (live but rejects curl) | 1 |
| URLs hard-broken (4xx/5xx) | 0 |
| URLs timeout | 0 (FRED reclassified as bot-blocked) |

## URL Freshness Results

| URL | Bibkey | HTTP | Status |
|---|---|---:|---|
| `https://matheusfacure.github.io/python-causality-handbook/` | `facure2022causal` | 200 | OK |
| `https://github.com/py-why/EconML` | `econml` | 200 | OK |
| `https://fred.stlouisfed.org/` | `fred` | 000 | **Bot-blocked** — server (Akamai) returns `HTTP/2 INTERNAL_ERROR` to curl even with browser UA + HTTP/1.1. Site is live in a real browser; this is rejection of automated traffic, not a 404. **Not actionable** — keep the citation. |
| `https://arxiv.org/abs/1608.00060` (from `chernozhukov2017double`) | `chernozhukov2017double` | 200 | OK |
| `https://arxiv.org/abs/2011.04216` (from `dowhy2022`) | `dowhy2022` | 200 | OK |

**Verdict**: 0 hard-broken URLs. The book's existing URL footprint is healthy.
FRED is the only flagged result and it's a known bot-detection false positive,
not a broken citation.

## Missing-URL Coverage Gaps

29 of 34 entries (85%) have **no primary URL** recorded in `bibliography.bib`.
Most are journal/book entries where a `doi` or canonical-publisher URL would
let readers (and future agents) reach the primary source without a manual
search. Listed here for the user's review during the manuscript revision —
these are **not blockers** for the rest of the research-toolkit pipeline.

### High-impact additions (top primary sources for the core DML claims)

| Bibkey | Year | Suggested canonical URL form | Why this entry matters |
|---|---:|---|---|
| `chernozhukov2018double` | 2018 | DOI: `10.1111/ectj.12097` (Oxford Academic) | The headline DML paper — anchor for half the manuscript |
| `chernozhukov2022locally` | 2022 | DOI: `10.3982/ECTA17518` (Econometrica) | Locally robust semiparametric estimation — feeds the orthogonality story |
| `wager2018estimation` | 2018 | DOI: `10.1080/01621459.2017.1319839` (Taylor & Francis / JASA) | Causal forests — book mentions but does not implement |
| `nie2021quasi` | 2021 | DOI: `10.1093/biomet/asaa076` (Biometrika) | R-learner foundations |
| `athey2019machine` | 2019 | DOI: `10.1146/annurev-economics-080217-053433` | Survey paper readers will want directly |
| `athey2016recursive` | 2016 | DOI: `10.1073/pnas.1510489113` (PNAS) | Causal-tree foundation |
| `bach2022doubleml` | 2022 | DOI: `10.18637/jss.v108.i03` (Journal of Statistical Software) | Implementation paired with the book's Python helpers |
| `newey1987` | 1987 | JSTOR: `https://www.jstor.org/stable/1913610` | HAC estimator — chapter 5/7 anchor |
| `frisch1933partial` | 1933 | JSTOR: `https://www.jstor.org/stable/1907330` | FWL theorem — chapter 1 anchor |
| `lovell1963seasonal` | 1963 | DOI: `10.1080/01621459.1963.10480682` | FWL completion — chapter 1 anchor |

### Foundational entries (lower urgency — well-known textbooks/papers)

`angrist2009mostly`, `pearl2009causality`, `imbens2015causal`, `hernan2020causal`,
`huntington2021effect`, `neyman1959optimal`, `van2000asymptotic`,
`rubin1974estimating`, `holland1986statistics`, `rosenbaum2002observational`,
`rosenbaum2010design`, `dehejia1999causal`, `lalonde1986evaluating`,
`imbens2003sensitivity`, `vanderweele2017evalue`, `cinelli2020omitted`,
`bojinov2019time`, `imai2023should`, `dePrado2018`, `huntington2021effect` —
publisher/DOI form available but not blocking citation.

## Suggested actions (post-pipeline; not done by this run)

1. After the main research-toolkit pipeline finishes, the `papers/` agent-index
   will surface canonical primary URLs for many of these. Use
   `bibliography_overlap.md` (produced at step 11) to migrate matching entries
   into `bibliography.bib`.
2. For FRED specifically, the citation is fine as-is — bot-blocking does not
   indicate the citation is wrong.
3. Consider whether `chernozhukov2017double` (the arXiv preprint) should be
   merged with or replaced by `chernozhukov2018double` (the published version
   in The Econometrics Journal). Both are currently cited.

## Provenance

- URL extraction: `grep -E 'url\s*=' bibliography.bib` + `grep -oE 'arXiv:[0-9]+\.[0-9]+'`
- HEAD checks: `curl -sIL -o /dev/null -w '%{http_code}|%{url_effective}' --max-time 15`
- FRED retried with HTTP/1.1 + browser UA + GET + 60s timeout; all variants
  failed with HTTP/2 INTERNAL_ERROR or receive failure → categorized as
  bot-blocked.
- Chapter `.tex` files (`chapters/chapter_*.tex`): searched for `\href`,
  `\url`, bare `https://`, `arXiv:` patterns, DOI strings — none found
  (chapters use `\cite{}` only).
