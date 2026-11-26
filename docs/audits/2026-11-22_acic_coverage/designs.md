# What varies across the 32 ACIC designs

The organizers published the population average additive treatment effect (PATE) and a design ID for every contest file. The 32 low-dimensional IDs represent eight families of four variants built from five source datasets. Two families use right heart catheterization data and two use the same Portuguese student dataset. A design's name identifies the source of its covariates. The organizers simulated treatment and outcome, so the effects describe the simulation. [Challenge specification](https://sites.google.com/view/acic2019datachallenge/data-challenge) · [Organizer results archive](https://drive.google.com/open?id=1GnZMX9tQp8-n1DRGsIf4DGGtO_3DrDu5)

## Design crosswalk

Within each range the IDs correspond to variants 1 through 4 in order. The outcome type is published in `lowDim_plots.pdf`. We reconstructed the mapping to named generators using the checks below. [Organizer generator archive](https://drive.google.com/file/d/1Qqgmb3R9Vt9KTx6t8i_5IbFenylsPfrK/view)

| DGP IDs | Covariate source and generator | Outcome | Rows per file | Covariates across draws |
|---|---|---|---:|---|
| 17–20 | Cervical cancer, `low_cervicalcancer_bin_sim.R` | Binary | 668 | Fixed |
| 21–24 | Credit card defaults, `generate_simCreditCard.R` | Binary | 500 | Resampled |
| 29–32 | Right heart catheterization, `low_rhc_cont_sim.R` | Continuous | 5,735 | Fixed |
| 33–36 | Right heart catheterization, `low_rhc_bin_sim.R` | Binary | 5,735 | Fixed |
| 37–40 | Spam email, `generate_simSpam_cont.R` | Continuous | 500 | Resampled |
| 41–44 | Spam email, `generate_simSpam_binary.R` | Binary | 500 | Resampled |
| 57–60 | Student performance, `low_studentperf_cont_sim.R` | Continuous | 649 | Fixed |
| 61–64 | Student performance, `low_studentperf2_cont_sim.R` | Continuous | 649 | Fixed |

## What the variants test

* **Constant additive effects with complex assignment and response.** In 29, 30 and 31, adding treatment changes the conditional mean by 4, 10 and 2 respectively for every covariate profile. Design 30 uses nonlinear functions of age, lab measurements and risk factors in both assignment and outcome. Design 31 explicitly targets poor overlap. Design 32 instead lets the effect vary with age. Two designs can therefore challenge the same learner through quite different mechanisms.
* **A constant coefficient can yield heterogeneous risk differences.** Design 34 adds 0.25 to the log odds of the outcome under treatment. The increase in probability depends on baseline risk. The official PATE is a risk difference of 0.04963. Design 22 likewise combines nonlinear confounding functions with a constant treatment coefficient on the log-odds scale.
* **The number of a variant is insufficient to infer its features.** In the continuous spam family, variants 1 and 2 have constant additive effects of 4 and 2.5. The code for variant 3 (ID 39) includes a treatment-by-covariate interaction and a nonlinear propensity. Variant 4 (ID 40) also makes the effect a function of covariates. In the student families, designs 59 and 63 explicitly introduce poor overlap while preserving constant effects of −0.8 and −2. [Organizer generator archive](https://drive.google.com/file/d/1Qqgmb3R9Vt9KTx6t8i_5IbFenylsPfrK/view)

## What repeats in a draw

We compared the ordered covariate rows across all 3,200 files. Each fixed design has exactly one covariate matrix across its 100 draws. Its coverage describes repeating treatment assignment and outcome noise on those same profiles. Each resampled design has 100 distinct matrices. Its generator draws 500 source indices with replacement, generates treatment and outcome into arrays indexed by those indices, then writes the indexed rows. Two occurrences of the same index therefore have identical covariates, treatment and outcome. In the continuous spam designs the files have 456–488 distinct full rows out of 500. A row-wise random split can place an exact copy on both sides of a fold. The contest supplies neither source-unit IDs nor observation and label-release times for a rolling-origin analysis. [Generator archive](https://drive.google.com/file/d/1Qqgmb3R9Vt9KTx6t8i_5IbFenylsPfrK/view) · [Contest schema](https://sites.google.com/view/acic2019datachallenge/data-challenge)

## Reproducing the crosswalk

1. Join the observed CSV stem to `filename` in `lowDim_trueATE.csv` from the results ZIP. Its SHA-256 is `f5359ee8676fe96a97a440bf3b892de1836f0f45e83b1a47175710accc6ce028`. All 3,200 stems match exactly once. Group by `DGPid`. For example, designs 31, 61 and 62 all have PATE 2. The contest files can score point error and interval coverage. Computing person-level effect error requires the extra expected counterfactual outcomes released for the development files. [Organizer FAQ](https://sites.google.com/view/acic2019datachallenge/faq)
2. For the 20 fixed designs, `TS_Generate_Low_Dim/simall.R` assigns shuffled file numbers to its generators. The following reproduces all 2,000 assignments in R 4.5.2 using the historical sampling algorithm.

   ```r
   key <- read.csv("lowDim_trueATE.csv")
   suppressWarnings(RNGversion("3.5.0"))
   set.seed(10000)
   stems <- paste0("low", sample.int(2000))
   expected <- rep(c(29:32, 57:60, 61:64, 33:36, 17:20), 100)
   stopifnot(all(key$DGPid[match(stems, key$filename)] == expected))
   ```

3. The three scripts in `SG_Generate_Low_Dim/` name their outputs by variant. To connect these names to contest IDs, we reproduced the duplicate-index positions in the first draw for each seed. With the same R sampling algorithm, run `set.seed(seed)` followed by `b <- sample.int(n, 500, replace = TRUE)`. For each of the 1,200 resampled-family files, compare each full row at a position where `duplicated(b)` is true to the row at its first occurrence, `match(b, b)`. Each seed matched exactly one file. Looking up those stems in the published key gives the 12 IDs below. The generator ZIP has SHA-256 `985f2ba290863b4f8c5b3fe7451e28dd47ad281f4e36d34cd5c87a7bf616d551`. [Organizer generator archive](https://drive.google.com/file/d/1Qqgmb3R9Vt9KTx6t8i_5IbFenylsPfrK/view)

   | Family | Source rows `n` | Seeds for variants 1–4 | Matched file stems in seed order |
   |---|---:|---|---|
   | Credit, binary | 30,000 | 1–4 | low3054, low2951, low2709, low2773 |
   | Spam, continuous | 4,601 | 15–18 | low3134, low3118, low2142, low2109 |
   | Spam, binary | 4,601 | 10–13 | low2281, low2862, low2363, low2939 |
