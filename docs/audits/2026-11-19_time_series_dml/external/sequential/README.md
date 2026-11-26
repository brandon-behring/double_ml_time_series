# Italian bank capital and two micro-randomized trials

## Italian prudential capital shocks

Ciganovic, D’Amario and Tancioni estimate the cumulative response of Italian output, bank balance sheets, corporate lending and lending spreads to a rise in the Tier 1 capital ratio. Their July 2026 revision supplies the specification for Figure 2. The March preprint uses different tuning and folds. A causal interpretation requires the paper’s assumption that controls absorb other shocks related to the capital innovation and future outcomes. [Published paper](https://doi.org/10.1093/ectj/utag019) · [Revised Section 5 and Figure 2](https://arxiv.org/html/2603.10999v2#S5)

The archive’s four frequency-specific files rebuild 71 quarterly observations on 37 series byte-for-byte. The published sample spans 2005Q2–2024Q4 and omits 2020Q1–2021Q4. The author code transforms levels, aligns three lags and eight leads, then retains 54 estimation dates and 105 regressor columns including an intercept. The 54 dates run from 2006Q4 to 2019Q4, plus 2022Q1. It fits block-specific Lasso models under six reverse cross-fitting blocks of nine observations, selecting from 100 penalties with the authors’ stability criterion. Inference uses cumulative responses and Bartlett HAC with bandwidth `h+1` over horizons zero through eight. [Author package, including data and code](https://zenodo.org/records/21030557)

The replay deletes the distributed outputs in a temporary copy, rebuilds them, then checks all 45 point estimates and 180 confidence limits against the originals. The largest absolute difference was 1.15 × 10⁻¹³. The estimated impact difference between Tier 1 capital and risk-weighted assets provides the factor 0.0628041157 used to normalize the capital-ratio change to 50 basis points. Only the lending spread gets another factor of 100 to express it in basis points.

| Two-quarter cumulative response | Estimate | 95% interval |
| --- | ---: | ---: |
| Real GDP, published Figure 2 | −0.14381% | [−0.24426%, −0.04336%] |
| Real lending to private nonfinancial corporations, published Figure 2 | −0.15857% | [−0.31440%, −0.00275%] |
| Real GDP, repository forward fitting on the 54 paper origins | −0.01787% | [−0.20655%, 0.17082%] |
| Real GDP, repository forward fitting on 51 calendar-valid origins | −0.10276% | [−0.24305%, 0.03753%] |

The two forward estimates use the authors’ GDP design, the sum of GDP growth at horizons zero through two, the published unit conversion, and the repository’s `TemporalPLRDML`. Both use ridge nuisance regressions with penalty 1, three expanding splits, a two-row gap to keep training outcomes clear of test dates, and HAC bandwidth three. The authors’ preprocessing takes leads and lags by row. Requiring uninterrupted quarters from `t−4` through `t+2` protects the three lags of first differences and the full response horizon. This removes 2019Q3, 2019Q4 and 2022Q1. After the initial training period, forward fitting scores 39 and 36 rows respectively. Outcome prediction R² was −1.81 and −2.54. These estimates show how far this sample can move with a different learner, training direction and calendar support. The separate contributions of learner choice and training direction would require further specifications.

Zenodo licenses version 21030557 under CC BY 4.0. The README documents permission to redistribute the included data and identifies the subscription requirement for obtaining the FTSE series directly from Refinitiv. The opt-in command uses the archived inputs, including the authors’ platform-specific X-13 seasonal-adjustment program. It checks the ZIP’s SHA-256 and reproduces a compiled data SHA-256 of `b96cec8af4b0f3e666765ea387b5d843f2f5ee1ef5ae6ec3d2d54372c47d4738`. [Zenodo metadata and license](https://zenodo.org/api/records/21030557)

## Drink Less

Bell and colleagues randomized 349 people on each of 30 evenings to receive a new message, a standard notification or no notification. In 10,470 decisions, the numbers of app opens in the next hour were 384/3,146, 407/3,112 and 153/4,212 respectively. The target is the relative risk of opening the app after either notification versus no notification, among the trial’s repeatedly randomized users. The separate fixed-policy arms are outside this analysis. [Published study, Tables 2 and 6](https://mhealth.jmir.org/2023/1/e38342/) · [Data](https://osf.io/w3szp/) · [Author analysis](https://github.com/lauren-bell/DrinkLessMRT/tree/9beb720e39d987b268b0423a0fdf91aea0cfbb20)

| Outcome and method | Estimate | 95% interval |
| --- | ---: | ---: |
| One-hour open, published relative risk | 3.52329 | [2.91755, 4.25480] |
| Post hoc 24-hour open, published relative risk | 1.25969 | [1.18673, 1.33714] |
| One-hour open, relative risk using the designed assignment probability | 3.52433 | [2.91882, 4.25546] |
| One-hour open, cross-fitted absolute risk difference | 9.10 percentage points | [7.47, 10.73] |

The independent multiplicative estimating equation uses the seven numeric controls in the author script and sums score contributions within participant for its sandwich variance. Treating the 10,470 rows as independent changes the primary log standard error from 0.09625 to 0.08516. The archived `prob_A` equals 0.6 for observations assigned a notification and 0.4 for those assigned none. Supplying it to the author function as the probability of a notification yields constant working weights and reproduces the publication. Supplying the trial’s conditional probability of a notification, 0.6 on every evening, gives the third row. The fourth row uses augmented inverse probability weighting, the design probabilities and separate logistic outcome models trained on four of five folds of participants. Every participant stays in one fold and inference clusters by participant.

## myBPmyLife

Golbus and colleagues randomized 287 participants with usable data across 187,517 decisions. Activity and dietary notifications each had probability 0.125, and no notification had probability 0.75. The analysis models `log(steps + 0.5)` during the hour following the decision. The reported ratio is the exponentiated activity coefficient, relative to no notification. The data yield 181,710 complete decisions and 153,983 with heart rate present in the previous 30 minutes. [Published activity result and methods](https://www.jmir.org/2026/1/e78218/) · [Trial data, cohort and analytic notebook](https://github.com/WIRED-L-um/MRT_myBPmyLife/tree/fa86c0d44ced3876f4a7351bc2d475fc4249342e) · [Demographic data](https://github.com/WIRED-L-um/myBPmyLife-Primary-Analysis/tree/f47ce32b896f55b2c7279b35aab790a75f4398b1)

| Population | Adjusted ratio | 95% interval | p |
| --- | ---: | ---: | ---: |
| Pre-decision heart rate available | 1.01252 | [0.98293, 1.04211] | 0.40263 |
| All complete decisions, including those without pre-decision heart rate | 1.02241 | [0.98509, 1.05972] | 0.23309 |

The replay fits the centered activity and diet indicators, the notebook’s baseline and time-varying covariates and the participant-clustered sandwich. It uses the notebook’s delta-method interval on the ratio scale with 278 degrees of freedom. The public files use CSVs and several updated column names. The replay uses age above 65, white race derived from `Race_5cat`, and `base_step` from the baseline file. Its log estimate and standard error are 0.012444 and 0.014845, compared with 0.012468 and 0.014885 printed in the notebook. The numerical assertions check the published precision of 1.01 [0.98, 1.04], p = 0.40. Including occasions without prior wear changes the eligible population. Assuming independent rows in the eligible population gives a log standard error of 0.014572.

The two trial archives provide public downloads. We found no explicit data reuse license in the Drink Less OSF project metadata or the two pinned myBPmyLife repositories. The scripts cache the fingerprinted original files locally and print aggregate results.

## Run from the repository root

Use the repository’s Python environment for the estimators and base R’s `Rscript` for decoding Drink Less’s RDS file. The Italian author scripts have their own numerical pins. The following commands create their Python environment and keep all downloads outside the repository.

```bash
export DML_TS_BENCHMARK_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/dml-ts/external"
mkdir -p "$DML_TS_BENCHMARK_CACHE"
SEQ="docs/audits/2026-11-19_time_series_dml/external/sequential"
uv venv --python 3.12 "$DML_TS_BENCHMARK_CACHE/italy-venv"
export DML_TS_ITALY_PYTHON="$DML_TS_BENCHMARK_CACHE/italy-venv/bin/python"
uv pip install --python "$DML_TS_ITALY_PYTHON" numpy==1.26.4 pandas==2.3.0 \
  matplotlib==3.9.4 scikit-learn==1.6.1 statsmodels==0.14.4 scipy==1.15.3 \
  joblib==1.5.2 openpyxl==3.1.5
uv run --frozen python "$SEQ/italy.py" --cache-dir "$DML_TS_BENCHMARK_CACHE"
uv run --frozen python "$SEQ/drink_less.py" --cache-dir "$DML_TS_BENCHMARK_CACHE"
uv run --frozen python "$SEQ/mybp.py" --cache-dir "$DML_TS_BENCHMARK_CACHE"
DML_TS_EXTERNAL_REPLAY=1 uv run --frozen --extra dev pytest -o addopts='' -q test/integration/test_external_sequential.py
```
