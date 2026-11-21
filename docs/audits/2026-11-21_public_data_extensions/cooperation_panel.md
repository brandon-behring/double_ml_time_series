# A repeated-assignment microeconomics benchmark for time-series DML

The Econometric Society hosts the data-and-code supplement for Bojinov, Rambachan and Shephard’s panel experiment. It contains the original Andreoni–Samuelson workbook, the cleaning code and the authors’ estimator. The workbook’s first sheet requires the original authors’ permission for scientific use and asks for attribution. The calculation below uses the released estimator on synthetic outcomes and assignments. The workbook remains in the local downloaded package. [Econometric Society package](https://www.econometricsociety.org/publications/quantitative-economics/2021/11/01/Panel-experiments-and-dynamic-causal-effects-A-finite-population-perspective/supp/1598-7412-1-SP.zip) · [Published panel study, section 6](https://onlinelibrary.wiley.com/doi/10.3982/QE1744) · [Andreoni and Samuelson, experimental design](https://econweb.ucsd.edu/~jandreon/Publications/JET2006.pdf)

## Three definitions to settle before using participant outcomes

| Quantity | What the sources say | Implication for a replication |
|---|---|---|
| Treatment | The paper describes stakes with $\lambda>0.6$. The released cleaner uses $\lambda\geq0.6$ and a treatment probability of $5/11$. The original experiment sampled eleven equally likely levels, and its Table 1 lists 115 of 1,100 pairs at exactly 0.6. | The strict rule gives $4/11$. Moving the boundary changes treatment membership for both members of those 115 pairs in the published original frequencies. Specify the indicator and its design probability together. |
| Outcome | The panel paper describes first-stage cooperation. The cleaner selects rows with `round == 2` and sets the outcome to one when the decision text begins with `PULL`. The original experiment calls pushing to the partner cooperation and pulling to oneself defection. | The stage and action labels need reconciliation before interpreting a participant-level estimate as an effect on cooperation. |
| Assignment and folds | Two people share the same stake draw. In each of five sessions, 22 people were rematched for 20 games. A partnership never repeats. | Keep both partners together when redrawing an assignment. For training separate from evaluation, hold out a whole session or use strictly earlier games and respect the effect window. An individual-row split can put a partner’s copy of the same treatment in the training fold. |

The source files are `Code/andreoni-samuleson_clean-data_cooperation.R` and `Code/panel_analysis_functions.R` in the journal ZIP. These observations concern the documented implementations. An empirical discrepancy in an effect size would require an authorized reconstruction of the decision and payoff columns. [Released code](https://www.econometricsociety.org/publications/quantitative-economics/2021/11/01/Panel-experiments-and-dynamic-causal-effects-A-finite-population-perspective/supp/1598-7412-1-SP.zip) · [Original experiment, sections 4–5 and Table 1](https://econweb.ucsd.edu/~jandreon/Publications/JET2006.pdf) · [Panel study, section 6 and Table 3](https://onlinelibrary.wiley.com/doi/10.3982/QE1744)

## A fully enumerated test with rematching and carryover

Four artificial players complete three games with new partners in each game. Each of the six pair-game assignments is independently drawn with probability $5/11$. Enumerating all 64 paths makes the randomization means and standard deviations exact up to floating-point arithmetic. Participant $i$ in game $t$ has the synthetic response $Y_{it}=0.15+0.03i+0.10t+0.08W_{it}+0.12W_{i,t-1}+0.20W_{it}W_{i,t-1}$, with zero treatment before the first game. The lag-zero effect averages over the actual design distribution of past treatments. The lag-one effect averages equally over the two possible subsequent assignments, as in the published score. [Runnable base-R test](checks/panel_cooperation.R) · [Results](reference/cooperation-synthetic.txt)

| Lag | Analytical effect | Released estimator, mean / SD | History-adjusted score, mean / SD |
|---|---:|---:|---:|
| 0 | 0.14061 | 0.14061 / 0.48018 | 0.14061 / 0.03936 |
| 1 | 0.22000 | 0.22000 / 0.68304 | 0.22000 / 0.10106 |

At every path and both lags, our independent calculation agrees with the released Horvitz–Thompson function within $10^{-12}$. The adjusted score subtracts a nuisance prediction fitted on 20 separate simulated sessions, using only information available before the first assignment in the effect window. Both means retain the analytical target. As a negative control, subtracting $0.12W_{i,t-1}$ from the response gives a lag-one mean of **0.10000**. For that contrast the subtracted variable is the treatment whose effect we want. The same variable is allowable prior history for the contemporaneous effect. This provides a test for time alignment in a DML implementation with a known propensity. [Estimator and negative control](checks/panel_cooperation.R)

```bash
audit=docs/audits/2026-11-21_public_data_extensions
Rscript --vanilla "$audit/checks/panel_cooperation.R" output/time-series-dml/QE1744.zip
```

The script verifies the full ZIP checksum and evaluates only the published point-estimator definition from the R source. It uses base R and the `sha256sum` utility. All figures in the two-row table come from the artificial games. [Prepare the source ZIP](README.md#reproduce-the-calculations) · [Public-data comparisons on demand and electricity](time_series_dml.md)

## Does inference respect the shared pair assignment?

Under a sharp null, take the assignment-invariant part of the preceding outcome, $Y_{it}=0.15+0.03i+0.10t$. Each round has two pairs. The test fixes the rematching schedule, computes the absolute Horvitz–Thompson statistic at each lag and includes ties in its two-sided tail. The true randomization distribution weights each of the 64 legal histories by $(5/11)^k(6/11)^{6-k}$. For comparison, a person-level reference flips 12 independent coins, admitting paths on which partners receive different treatments. In both columns below, size is evaluated under the true six-coin design. Every legal assignment is enumerated, so these percentages have no Monte Carlo error. [Exact enumeration](checks/panel_randomization.py) · [Finite-population treatment paths and sharp-null inference](https://arxiv.org/abs/2003.09915)

| Nominal 5% test | Pair-round reference, 64 paths | Person-round reference, 4,096 paths |
|---|---:|---:|
| Contemporaneous effect | 4.574% | 17.720% |
| Lag-one effect | 4.269% | 17.778% |

An independent base-R enumeration using the released point-estimator function reproduced all four rates within $6\times10^{-13}$. Append `--check-null` to the R command above to run it. The two pair-based tests attain slightly less than 5% because of discreteness. Treating partners as independent causes both tests to reject more than three times as often as intended on this synthetic panel. This supplies an inference check to run alongside the earlier point-estimator check. For an extension with a learned adjustment, freeze a nuisance model trained on independent sessions, or recompute every assignment-dependent fitting step under each legal history. The measured rates concern this fixed sharp null and matching schedule. They do not measure coverage for a weak average-effect null or for the original participants. The exact enumeration also runs independently in the standard Python library. From the repository root use `.venv/bin/python docs/audits/2026-11-21_public_data_extensions/checks/panel_randomization.py`. [Exact masses and distinct test statistics](reference/cooperation-randomization.json) · [Original paired experiment](https://econweb.ucsd.edu/~jandreon/Publications/JET2006.pdf)
