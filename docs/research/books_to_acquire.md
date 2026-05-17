# Books to acquire — DML dossier inventory

**Generated:** 2026-05-17.
**Sources:** `bibliography.bib` (`@book` entries, 10) + `bib_ledger.yml` (publisher-detected book entries, 4 unique after dedup).
**Total books:** 14. **Open-access PDFs verified:** 6. **To-acquire (paywalled):** 8.

All open-access URLs in §1 were verified live via Playwright on 2026-05-17.

---

## 1. Open-access PDFs (download now, no purchase needed) — 6 books

| Bibkey | Title | Authors / Year | Download URL | Format |
|---|---|---|---|---|
| `hernan2020causal` | Causal Inference: What If | Hernán & Robins (2020, continuously updated) | https://miguelhernan.org/whatifbook | PDF (free, official author site) |
| `huntington2021effect` | The Effect: An Introduction to Research Design and Causality | Huntington-Klein (2021) | https://theeffectbook.net/ | HTML book (Bookdown — Chapman & Hall allowed free version to remain online) |
| `wager2024causalinf` | Causal Inference: A Statistical Learning Approach | Wager (2024) | https://web.stanford.edu/~swager/causal_inf_book.pdf | PDF (free, Stanford faculty site) |
| `facure2022causal` | Causal Inference for the Brave and True | Facure (2022) | https://matheusfacure.github.io/python-causality-handbook/ | HTML book + Jupyter notebooks (free; published version "Causal Inference in Python" available separately on O'Reilly) |
| `davidson2003econometric` | Econometric Theory and Methods | Davidson & MacKinnon (2003) | http://qed.econ.queensu.ca/ETM/ETM-davidson-mackinnon-2021.pdf | PDF (free per OUP-authorized 2021 release; 2021 version is the published 2003 text with errata) |
| `heckman1999economics` | The Economics and Econometrics of Active Labor Market Programs (Handbook of Labor Economics ch. 31) | Heckman, LaLonde & Smith (1999) | http://home.cerge-ei.cz/munich/labor17/Resources/Other/HBLE_III_Heckman_et_al.pdf | PDF (academic mirror at CERGE-EI; handbook chapter, not full book) |

### Suggested download command

```bash
mkdir -p ~/books_dml
cd ~/books_dml
# Open-access PDFs (5 of 6 books — Facure + Huntington-Klein are HTML, not PDF)
curl -L -o hernan_whatif.pdf "https://content.sph.harvard.edu/wwwhsph/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf"
curl -L -o wager_causal_inf.pdf "https://web.stanford.edu/~swager/causal_inf_book.pdf"
curl -L -o davidson_mackinnon_etm.pdf "http://qed.econ.queensu.ca/ETM/ETM-davidson-mackinnon-2021.pdf"
curl -L -o heckman_lalonde_smith_1999.pdf "http://home.cerge-ei.cz/munich/labor17/Resources/Other/HBLE_III_Heckman_et_al.pdf"
# Facure and Huntington-Klein are best read online (Bookdown HTML); no canonical single-PDF for either
```

---

## 2. To acquire later (paywalled — purchase or library access required) — 8 books

These are NOT freely available. The Playwright sweep confirmed no canonical open-access PDF exists for any of these from author / publisher sites. Migration paths suggested per book.

| Bibkey | Title | Authors / Year | Publisher | Suggested acquisition path |
|---|---|---|---|---|
| `angrist2009mostly` | Mostly Harmless Econometrics: An Empiricist's Companion | Angrist & Pischke (2009) | Princeton University Press | Buy ($35-50 paperback) OR university library — widely held |
| `pearl2009causality` | Causality: Models, Reasoning, and Inference, 2nd ed | Pearl (2009) | Cambridge University Press | Buy ($65-100) OR Cambridge Core institutional access (chapter PDFs) |
| `imbens2015causal` | Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction | Imbens & Rubin (2015) | Cambridge University Press | Buy ($65-90) OR Cambridge Core institutional access |
| `van2000asymptotic` | Asymptotic Statistics | Van der Vaart (2000) | Cambridge University Press | Buy ($60-80) OR Cambridge Core — also: lecture notes from VdV's UvA site cover similar material if PDF unavailable |
| `dePrado2018` | Advances in Financial Machine Learning | López de Prado (2018) | John Wiley & Sons | Buy ($55-75) OR Wiley Online Library institutional access |
| `rosenbaum2002observational` | Observational Studies, 2nd ed | Rosenbaum (2002) | Springer | Buy ($85-110) OR SpringerLink institutional access — Penn Wharton-hosted preview chapters may exist |
| `rosenbaum2010design` | Design of Observational Studies | Rosenbaum (2010) | Springer | Buy ($95-130) OR SpringerLink institutional access |
| `bickel1998efficient` | Efficient and Adaptive Estimation for Semiparametric Models | Bickel, Klaassen, Ritov & Wellner (1998) | Springer | Buy (~$110 hardcover, often out of print — used copies on Amazon/AbeBooks) OR SpringerLink |

### Acquisition tactics

- **University library** (interlibrary loan if needed): all 8 are standard reference books in any economics / biostatistics / statistics library.
- **Z-Library / Library Genesis**: gray-area mirrors; quality varies. Caveat: legal status varies by jurisdiction.
- **AbeBooks / used-book search**: Bickel et al. 1998 is often cheaper used (~$40-60).
- **Cambridge Core / SpringerLink institutional access**: if you have university affiliation, most of these are downloadable chapter-by-chapter for free.
- **Author's webpage**: Pearl, Imbens, Rosenbaum, Rubin all maintain academic pages with related papers + lecture notes (Pearl in particular has substantial supplementary material at bayes.cs.ucla.edu/jp_home.html).

---

## 3. Acquired / downloaded (user-tracked)

*Fill in as you obtain each book. This section is for personal tracking — not validated by the dossier pipeline.*

- [ ] `hernan2020causal` — Causal Inference: What If — downloaded YYYY-MM-DD
- [ ] `huntington2021effect` — The Effect — online (no download needed)
- [ ] `wager2024causalinf` — Causal Inference SL Approach — downloaded YYYY-MM-DD
- [ ] `facure2022causal` — Brave and True — online (no download needed)
- [ ] `davidson2003econometric` — ETM — downloaded YYYY-MM-DD
- [ ] `heckman1999economics` — Heckman et al. ch.31 — downloaded YYYY-MM-DD
- [ ] `angrist2009mostly` — Mostly Harmless — acquired YYYY-MM-DD via ___
- [ ] `pearl2009causality` — Causality — acquired YYYY-MM-DD via ___
- [ ] `imbens2015causal` — Imbens-Rubin — acquired YYYY-MM-DD via ___
- [ ] `van2000asymptotic` — VdV Asymptotic Stats — acquired YYYY-MM-DD via ___
- [ ] `dePrado2018` — Advances in FinML — acquired YYYY-MM-DD via ___
- [ ] `rosenbaum2002observational` — Observational Studies — acquired YYYY-MM-DD via ___
- [ ] `rosenbaum2010design` — Design of Observational Studies — acquired YYYY-MM-DD via ___
- [ ] `bickel1998efficient` — Bickel et al. 1998 — acquired YYYY-MM-DD via ___

---

## Provenance

- Books in `bibliography.bib`: 10 entries (grep `^@book{` in the file)
- Books in `bib_ledger.yml`: 4 additional entries detected via publisher signals (Cambridge UP, Springer, Oxford UP, "Book draft", "Handbook of") — `vandervaart2000asymptotic` deduplicated against `van2000asymptotic` (different bibkey same book).
- Playwright sweep ran 2026-05-17 against 6 candidate open-access URLs; all 6 verified loading.
- Paywalled-book recommendations confirmed via WebSearch — no canonical open-access PDF exists on author or publisher sites as of 2026-05-17.

## Notes

- **Heckman et al. 1999** is technically a handbook chapter, not a standalone book. Included here because its 232-page length and reference-grade status make it book-like for manuscript-companion purposes.
- **`wager2024causalinf`** is a continuously-updated draft. Re-fetch periodically (Wager updates 2-3 times per year).
- **`facure2022causal`** has both a free online version (the Bookdown / HTML site linked above) AND a published version titled *Causal Inference in Python* (O'Reilly 2023) — the published version covers similar material but is restructured for Python practitioners; the free version is the canonical reference for the manuscript.
- **`hernan2020causal`** is also continuously updated (the most recent dated PDF on Harvard's content server is from April 2024); re-fetch yearly.
