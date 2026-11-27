# California Proposition 99 and cigarette sales

**The authors' synthetic difference-in-differences estimate reproduces at −15.60 packs per person per year, averaged over 1989–2000.** Its placebo-based 95% interval runs from −32.01 to +0.80. Applying the repository's temporal PLR to California gives −34.50 and a narrower nominal interval. The PLR fold covering policy adoption had never seen a treated year in training. Its nonzero residual treatment variance and finite standard error did not flag that lack of support. The donor-panel comparison provides the more credible design for this question. [Authors' replication](https://synth-inference.github.io/synthdid/articles/synthdid.html) · [Temporal PLR diagnostic](california.py)

## Design and published benchmarks

The authors' released CSV contains 1,209 rows, one per state-year for 39 states over 1970–2000. We checked uniqueness, completeness, and the treatment rule. The fields are `State`, `Year`, `PacksPerCapita`, and `treated`. California passed its tobacco-control measure in 1988. The first treated year is 1989, leaving 19 pre-program years and 12 post-program years. The 38 comparison states supply counterfactual sales paths. The measured outcome is tax-paid cigarette sales divided by state population. Cross-border purchases and concurrent changes in tobacco policies can affect its interpretation as consumption. A donor design assumes the untreated donors remain informative about California's counterfactual and that the intervention has limited spillover into donor sales. [Data dictionary](https://synth-inference.github.io/synthdid/reference/california_prop99.html) · [Original study and data sources](https://economics.mit.edu/sites/default/files/publications/Synthetic%20Control%20Methods.pdf)

The original Abadie, Diamond and Hainmueller study estimates roughly 26 fewer packs **in 2000**. Its post/pre mean squared prediction error ratio ranked first among all 39 states, giving a placebo tail fraction of 1/39. Its matching design uses additional pretreatment predictors. Arkhangelsky and colleagues instead report estimators for the **average post-program effect** using the outcome panel. Our reproduction below follows the latter study's released data and code. [Original synthetic-control estimate](https://doi.org/10.1198/jasa.2009.ap08746) · [Authors' SDID Table 1 reproduction](https://synth-inference.github.io/synthdid/articles/paper-results.html)

## Donor-panel reproduction and falsification

Each method used the authors' original R estimation function and 200 draws of their donor-placebo standard-error procedure, with seed 12345 reset for each method. The same 12 post-program years enter each estimate. The 95% intervals use the estimate plus or minus 1.96 placebo SEs. With a single treated state, the authors caution that this placebo distribution may give a poor approximation of California's uncertainty. The authors' confidence interval and the original paper's placebo rank correspond to different estimators and statistics. [Executable comparison](california_reference.R) · [Authors' inference notes](https://synth-inference.github.io/synthdid/articles/synthdid.html)

| Recomputed method | Packs/person/year | Placebo SE | Approximate 95% interval |
| --- | ---: | ---: | ---: |
| Ordinary difference in differences | −27.35 | 17.74 | [−62.12, 7.42] |
| Synthetic control, outcome history | −19.62 | 9.92 | [−39.06, −0.18] |
| Synthetic difference in differences | −15.60 | 8.37 | [−32.01, 0.80] |

Leaving out each donor in turn gave SDID estimates from −17.06 (Nevada omitted) to −14.98 (Montana omitted). Using only the pre-program data and assigning a fictitious start in 1983 gave +0.32 packs averaged over 1983–1988. These results support the direction of the estimate against individual donor exclusions and this pre-period comparison. They still leave uncertainty about California's untreated post-1988 trajectory.

For a known-effect check we removed California, took each of the 38 donors in turn as a pseudo-treated state, and fit SDID using the other 37 donors. The mean estimated effect under a zero injection was +0.39 packs, RMSE 9.38, and the empirical 2.5th to 97.5th percentile range was [−16.46, +14.73]. Subtracting exactly 15 packs from each pseudo-treated state's post-period outcome shifted its estimate by −15 to within 7 × 10⁻¹⁵. This tests sensitivity to an additive post-program change. The dispersion of the unshifted estimates measures how poorly some states' untreated paths are reconstructed. All 38 states were enumerated and share donor information, so the percentile range describes this donor pool.

## Temporal DML support diagnostic

We sorted California by year, set the exposure to the 1989 policy indicator and used the contemporaneous mean sales of all 38 comparison states as the sole control. The repository's public `TemporalPLRDML.fit` API used ridge nuisances, no treatment lags, three forward folds and a one-year gap. The donor mean can proxy common shocks if other states are unaffected. This scalar before/after specification also needs credible intervention timing and residual treatment support. Only one California policy transition is available. [Diagnostic implementation](california.py) · [Repository estimator](../../../../dml_ts/dml/temporal_plr_dml.py)

| Training years | Test years | Treated years in training | Treated years in test |
| --- | --- | ---: | ---: |
| 1970–1978 | 1980–1986 | 0 | 0 |
| 1970–1985 | 1987–1993 | 0 | 5 |
| 1970–1992 | 1994–2000 | 4 | 7 |

Only 21 dates supplied scored residuals. The initial 10 were discarded. Treatment nuisance R² was −0.034, outcome R² was 0.275 and mean squared residual treatment was 0.253. All three fits returned −34.50 without a runtime warning. HAC bandwidths 0, 2 and 4 gave respective SEs of 2.92, 3.95 and 4.09 and nominal intervals [−40.23, −28.78], [−42.25, −26.75] and [−42.52, −26.49]. A bandwidth adjustment treats temporal covariance in the score. It supplies no experience of treatment in the two early training samples and no second treated state. Agreement in direction therefore carries little weight as a validation of the PLR policy interval.

## Reproduce

Run from the repository root using the locked Python project dependencies and base R. The audit used Python 3.13.17, temporalcv 2.2.0 and R 4.5.2. The R script sources the authors' four computational files directly from the pinned source tree. Its estimator, panel conversion and placebo routines require only base/recommended R functions in this application. The CSV bundled in that archive is byte-identical to their separately downloaded data. [Pinned author source and data](https://github.com/synth-inference/synthdid/tree/70c1ce3eac58e28c30b67435ca377bb48baa9b8a) · [R script](california_reference.R) · [Python script](california.py)

```sh
audit_src=/tmp/dml-audit-synthdid-src
audit_csv="$audit_src/data/california_prop99.csv"
mkdir -p "$audit_src"
curl -fLsS 'https://codeload.github.com/synth-inference/synthdid/tar.gz/70c1ce3eac58e28c30b67435ca377bb48baa9b8a' -o /tmp/dml-audit-synthdid-pinned.tar.gz
sha256sum /tmp/dml-audit-synthdid-pinned.tar.gz
tar -xzf /tmp/dml-audit-synthdid-pinned.tar.gz --directory "$audit_src" --strip-components=1
OPENBLAS_NUM_THREADS=1 Rscript --vanilla docs/audits/2026-11-19_time_series_dml/pilots/california_reference.R "$audit_csv" "$audit_src"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python docs/audits/2026-11-19_time_series_dml/pilots/california.py "$audit_csv"
```

Pinned source archive SHA-256 is `9bf048eb12e5972a2290726136acaacbf6f13471072428354eeeed0298607766`. CSV SHA-256 is `550d24c2f743507d097692d181bdbbf36c6168a39a0b6fc831cc89d98082f7de`. The replication repository identifies its software licenses. A separate license for redistributing the original cigarette-sales data was not established in this audit. The report links to the source distribution.
