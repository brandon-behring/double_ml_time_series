# Public-data checks for time-series double machine learning

These calculations test how well an estimator recovers a known dynamic effect, how the assignment unit changes inference, and how much baseline adjustment changes an observed experimental contrast. Each script runs independently of the package and can serve as a numerical reference when testing its implementation. [Temporal partially linear estimator](../../../dml_ts/dml/temporal_plr_dml.py)

| Experiment | What the data identify or calibrate | Result |
|---|---|---|
| Washington bike rentals | Observed hourly demand, independent simulated offers with a known lag profile | With 200 offer histories and a weekly gap, estimated hour/workday propensities recovered effects of 10, 5, 2 and 0 within 0.12 rentals on average. A more elaborate propensity fit produced large early-fold biases. [Bike analysis](time_series_dml.md#what-actually-happened-in-the-bike-benchmark) |
| Norwegian electricity offers | Observed group assignment and meter readings in 512 released Oslo households | The adjusted offer effect in the peak window on the last ten dates was −0.042 kWh/hour (95% interval −0.139 to 0.054). The original arm-specific retention rates would be needed to quantify selection. [Oslo analysis](time_series_dml.md#observed-microeconomic-intervention-with-hourly-outcomes) |
| Quarter-hour ride demand | Observed counts, simulated block assignments and carryover | One-hour and six-hour blocks target 33.93 and 78.32 additional rides per quarter, respectively, under a sustained effect of 100. Larger blocks change the estimand and the number of independent assignments. [Switchback analysis](time_series_dml.md#when-randomization-happens-in-time-blocks) |
| Repeated cooperation games | The published estimator and design, synthetic shared assignments and outcomes | Under a fixed sharp null the pair-based tests rejected in 4.57% and 4.27% of paths by design probability. Independent draws for partners inflated both rates to about 17.7%. [Point estimates, assignment and inference](cooperation_panel.md) |

## Reproduce the calculations

From the repository root, use the project's Python environment. In a fresh clone, `uv sync --frozen` creates `.venv` from the lockfile. The calculations use NumPy, SciPy, Python's time zone database, base R with `Rscript`, and GNU `sha256sum`. The replay used Python 3.13.17, NumPy 2.4.6, SciPy 1.17.1, and R 4.5.2 on Linux.

The commands below populate `output/time-series-dml/`, which git ignores. For an existing cache, run the checksum command against the four inputs before starting. The journal ZIP includes a permission condition on scientific use of the participant workbook. The panel script reads the released R estimator from the ZIP and generates all of its own assignments and outcomes. [UCI release](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) · [Zenodo v2 release](https://zenodo.org/records/8248802) · [Ride-demand data and simulation paper](https://proceedings.mlr.press/v300/blanchet26a.html) · [Panel study](https://doi.org/10.3982/QE1744)

```bash
cache=output/time-series-dml
audit=docs/audits/2026-11-21_public_data_extensions
mkdir -p "$cache"
curl -fsSL 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip' -o "$cache/bike.zip"
curl -fsSL 'https://zenodo.org/api/records/8248802/files/iFlex%20dataset.RData/content' -o "$cache/iflex.RData"
curl -fsSL 'https://raw.githubusercontent.com/wenqian-xing/conservative-inference-switchback-experiments/e29f6f09716e0df2a4edb8012c09a31c627338f5/lyft_ride_counts_15min.csv' -o "$cache/lyft_ride_counts_15min.csv"
curl -fsSL 'https://www.econometricsociety.org/publications/quantitative-economics/2021/11/01/Panel-experiments-and-dynamic-causal-effects-A-finite-population-perspective/supp/1598-7412-1-SP.zip' -o "$cache/QE1744.zip"
(cd "$cache" && sha256sum -c -) < "$audit/reference/inputs.sha256"
```

For a short bike run, use `--repetitions 2`. The reported bike results use 200 histories per configuration and the ride calculation uses 2,000 per block length. The seeds are fixed. The two panel calculations enumerate every assignment in their respective designs. The cooperation R command accepts `--check-null` to recalculate the Python rejection probabilities using the published estimator. [Bike, ride and Oslo commands](time_series_dml.md) · [Panel commands](cooperation_panel.md) · [Archived numerical outputs](reference/)

On the November 21 replay all 48 bike rows across the four runs exactly matched the historical numerical results. The current script adds a `last_block_only` flag to the three full-period outputs. The archived ride results, exact panel probabilities, six synthetic R point-estimator summaries and Oslo estimates also reproduced. The two panel implementations agreed on rejection probabilities to within $5 \times 10^{-13}$ at the R output’s printed precision. The reference text for Oslo trims trailing spaces from R output.

The local cache also contains the original 40-draw bike pilot (`bike-mc.json`) and its historical input/output manifest (`evidence.json`). That pilot used two propensity models. The three-model, 200-draw results provide the comparison reported here. The two script entries in the historical manifest identify the files in `checks/` by basename and SHA-256.

## Research that can build on these checks

1. Fit the project's temporal partially linear estimator to the same bike draws. Start at horizon zero, match the scored timestamps, then repeat at horizons 1, 2 and 6. At each horizon, purge by the latest training outcome time and compute HAC lags in elapsed hours. Differences in the training support and in the treatment nuisance model were large enough to affect coverage. [Bike design and results](time_series_dml.md#what-actually-happened-in-the-bike-benchmark)
2. Compare coverage for the switchback estimand under alternative carryover half-lives. Resample at the randomized block level and report the current-block contrast alongside the sustained contrast for each design. [Ride experiment](time_series_dml.md#when-randomization-happens-in-time-blocks)
3. For an empirical microeconomics application, seek the original assignment denominators for each Oslo arm. For the cooperation data, resolve permission, the treatment cutoff, the stage and the direction of the action coding before using participant outcomes. These are separate identification questions. [Oslo selection](time_series_dml.md#observed-microeconomic-intervention-with-hourly-outcomes) · [Panel definitions](cooperation_panel.md#three-definitions-to-settle-before-using-participant-outcomes)
4. Match EIA demand and as-issued weather forecasts to a common clock for one balancing authority. Reserve entire summers for evaluation. The current audit identifies the sources and defines two possible exposures. Joining the series and validating the forecast vintages remain open. [Observational extension](time_series_dml.md#observational-extension-with-aggregate-electricity-demand)
