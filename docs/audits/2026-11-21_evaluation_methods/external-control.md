# External known-truth control on the ACIC development data

On all **16 ACIC 2019 public development files**, the repo's ordinary PLR slope agreed with an independent reconstruction from held-out treatment probabilities and outcome predictions to within **7.2 × 10⁻¹⁵**. On one high-dimensional constant-effect file the true effect was −3, the unadjusted difference was −0.633, PLR was −3.016 and a cross-fitted augmented inverse-probability estimate was −3.002. This is an external check on implementation and on a favorable case for adjustment. Other files show meaningful error under the same procedure. [All observations summarized by file](additional/acic-results.csv) · [Official source and development-file documentation](https://sites.google.com/view/acic2019datachallenge/data-challenge)

| Organizer's development file | N / features | Population ATE | Mean of supplied conditional effects on these rows | Arm difference | Repo PLR | Cross-fitted AIPW (SE) | Conditional-effect RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low 1 | 500 / 22 | 0.191 | 0.189 | 0.042 | 0.139 | 0.925 (0.704) | 0.522 |
| Low 2 | 500 / 22 | 2.500 | 2.500 | 5.272 | 2.378 | 2.370 (0.171) | 0.595 |
| High 3 | 2,000 / 185 | −3.000 | −3.000 | −0.633 | −3.016 | −3.002 (0.089) | 0.720 |
| High 8 | 2,000 / 185 | 44.786 | 41.967 | 57.848 | 40.459 | 42.051 (2.744) | 32.701 |

These four examples show the range of outcomes. The CSV includes all remaining cases, held-out outcome MSE, propensity Brier score, residual treatment variation, treatment counts, clipping and individual conditional-effect RMSE in each file's own units. Pooling raw losses across files would give disproportionately more weight to effects measured on a large scale.

**Design.** The organizers released paired observed and counterfactual-mean files. Each observed file contains a binary treatment, outcome and pretreatment covariates. We used all numerical covariates as supplied and one frozen fivefold shuffled partition with seed 42. Outcome regression used ridge with penalty 1, and propensity estimation used logistic regression with inverse penalty 1. Every transformation was fitted on the current training rows. The AIPW calculation fits two additional arm-specific ridge models and bounds estimated probabilities to [0.01, 0.99]. It uses a cross-sectional influence-score SE. Unconstrained linear outcome predictions can exceed the admissible range for a binary outcome. All settings were fixed across the files. The supplied truth enters after fitting. [Driver with archive hashes](additional/acic.py)

**How to interpret disagreement.** The organizers' population ATE averages over their generating population. The sample mean of supplied conditional effects averages over this file's covariates. PLR weights conditional effects by residual treatment variation under appropriate nuisance functions. These generally differ under heterogeneous effects. The last file illustrates a 2.820-unit sample-versus-population difference before adding estimation error. The Low 1 AIPW calculation bounded seven probabilities and had a large estimated SE. That motivates comparing overlap and arm models against known truth across replicated draws. It does not demonstrate a general propensity-clipping correction or a failure of randomization. [Organizers' explanation of targets and replication](https://sites.google.com/view/acic2019datachallenge/faq)

**Boundary of the evidence.** These are accessible development files with no natural outcome-release or treatment clock. They provide an external known-truth control for standard DML and learner scoring. They support no claim about serial-HAC coverage, break detection or effect transfer. The repeated 100-draw designs in the challenge have the replication structure to assess population coverage. The 16 development files do not supply 16 interchangeable replicates of the same estimand. [Challenge design](https://sites.google.com/view/acic2019datachallenge/data-challenge) · [Temporal experiments in this repo](forward-validation.md)

The subsequent low-dimensional contest study used 100 draws per design. The nominal 95% AIPW intervals covered the published effect 0, 0 and 61 times in the three designs with statistically significant undercoverage after multiplicity adjustment. [All 3,200 contest fits and the propensity diagnostic](../2026-11-22_acic_coverage/README.md)

## Reproduce

Download the organizers' low- and high-dimensional **test datasets** using the two revised December 2018 development ZIP links. Keep their original filenames in a cache. The script checks both SHA-256 digests, requires eight matched observed files per track, accepts both documented-file naming variants for expected outcomes, and reads the ZIPs in memory.

```bash
audit=docs/audits/2026-11-21_evaluation_methods
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
    "$audit/additional/acic.py" /path/to/download-cache /tmp/acic-results.csv
diff "$audit/additional/acic-results.csv" /tmp/acic-results.csv
```
