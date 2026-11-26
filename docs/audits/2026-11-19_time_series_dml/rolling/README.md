# Can an earlier causal estimate predict the effect of a later intervention?

**Across four fully evaluable reminder cohorts, DML had slightly lower squared error than the earlier randomized estimate and higher squared error than zero. Both paired intervals included zero.** A simulation using the same calendar and bootstrap found that looking across those four quarters produced at least one apparent change in **95 of 500 panels with a stable effect**. Reports are organized by the question each study can answer.

| Report | Evidence and result |
|---|---|
| [How to validate an effect forecast](best-practices.md) | Define the intervention and decision date, record available data, distinguish persistence from a fresh assignment, and specify what would count as a consequential error. |
| [New reminder participants](reminders.md) | Seven quarterly origins and three forecasts. Two dates had no mature training cohort and the final cohort had one participant. Across the four shared comparisons, AIPW gained **0.000895 log²** over the earlier arm difference and lost **0.003301 log²** against zero. [Origin ledger](mybp-origins.csv) |
| [Power, false drift and changes in assignment](calibration.md) | At the first post-change origin the refitted bootstrap detected a real 0.10 decline in **404/500** simulations. Changing treatment probabilities alone moved the native PLR slope enough to flag all **500/500** January comparisons while both subgroup effects stayed fixed. [Bootstrap results](bootstrap-calibration.csv) · [All five methods](calibration-summary.csv) |
| [External experiments](external-controls.md) | The repeated effort task reproduced a rank correlation of **0.907**. Sierra Leone's paired infrastructure effect declined **0.094 SD** after one rollout. Three annual nudge-trial comparisons gave different method rankings. The retail logs exposed concentration within seconds in the item assignments. [Source results](external-controls.json) |
| [Blinded historical JTPA holdout](jtpa.md) | Among **1,832** adults with usable records the later earnings contrast was **−$14.29**, SE **$61.77**. A source marker present in 25 historical adult records materially affected both original forecasts. After correcting the decoder the primary historical and DML forecasts were **+$131.34** and **+$138.45**. The corrected forecasts were made after unblinding. [First issuance](jtpa-results.json) · [Corrected replay](jtpa-corrected-results.json) |

The JTPA replay used a public archive that had not previously been opened for outcome analysis in this project. We froze the protocol before reading participant wages and both historical forecasts before reading the held-out wage values. The source lacks contemporaneous operator extracts. The preserved original freezes document this project's research sequence; the dates in them reconstruct an earlier operator decision. Our JTPA and reminder results establish no general forecast advantage across time periods. [JTPA protocol](jtpa-protocol.json) · [Data verification](jtpa-source-audit.json)

The follow-up adds half the historical effect as a fourth forecast, replays the reminder and JTPA results, and records two forecasts of effects in future assignments. [Four-method comparison and prospective studies](../prospective/README.md)

## Reproduce from the repository root

The runner uses checksum-pinned sources and the existing external cache. Its five rolling entries cover reminders, the four-scenario calibration, the end-to-end bootstrap, four external control studies, and JTPA. Poppler supplies PDF table extraction for the effort experiment. First access to a source needs the network. Each study worker uses one numerical-library thread.

```bash
audit=docs/audits/2026-11-19_time_series_dml
cache=/tmp/dml-external-bench-cache
uv run --locked python "$audit/external/run.py" --suite rolling --cache-dir "$cache" --jobs 4
```

To regenerate tables and frozen forecasts in a separate directory, run the study adapters below. Parallelism in the two standalone simulations is across replications. Each replication's seed is independent of its worker assignment.

```bash
results=/tmp/dml-rolling-results
mkdir -p "$results"
uv run --locked python "$audit/external/forward/reminder_rolling.py" --cache-dir "$cache" --output-dir "$results"
uv run --locked python "$audit/external/forward/rolling_calibration.py" --cache-dir "$cache" --workers 8 --output "$results"
uv run --locked python "$audit/external/forward/reminder_bootstrap_calibration.py" --cache-dir "$cache" --workers 8 --output "$results/bootstrap-calibration.csv"
uv run --locked python "$audit/external/forward/rolling_controls.py" --cache-dir "$cache" > "$results/external-controls.json"
uv run --locked python "$audit/external/forward/jtpa_holdout.py" --cache-dir "$cache" > "$results/jtpa-replay.json"
```

To apply the freeze and score commands to another study, supply a protocol with the target, unit, horizon, ordered ISO decision dates, methods and design metadata. A forecast row contains origin, method, and value; an unavailable value requires a reason. An observation identifies the same origin and contrast, and supplies the later estimate or a reason it is unavailable. Unreported later estimates stay pending. The frozen file is created once. An identical retry returns its digest. Store that digest separately before accepting later observations. Results include signed errors and gains in squared error over the named zero and historical methods. [Implementation](../../../../dml_ts/validation/effect_forecast.py) · [Runnable examples and contract tests](../../../../test/validation/test_effect_forecast.py)

```bash
digest=$(uv run --locked python -m dml_ts.validation.effect_forecast freeze --protocol protocol.json --forecasts forecasts.json --output frozen.json)
uv run --locked python -m dml_ts.validation.effect_forecast score --frozen frozen.json --sha256 "$digest" --observations later.json
```

The source-inclusive integration checks run with `DML_TS_EXTERNAL_REPLAY=1` and `DML_TS_BENCHMARK_CACHE` set to the cache directory. For a second replay with the adjacent checkout of `temporalcv`, set `PYTHONPATH=../temporalcv/src`. [Audit runner](../external/run.py)

**Verification.** All five rolling suite entries completed with the locked `temporalcv` 2.2.0 and the sibling 2.3.0. The combined forward and rolling tests passed **72/72 on each version**, including the original-source replays. The command below reproduces that test selection from the repository root.

```bash
DML_TS_EXTERNAL_REPLAY=1 DML_TS_BENCHMARK_CACHE="$cache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --locked --extra dev python -m pytest -q test/integration/test_forward*.py \
  test/integration/test_{open_bandit,bracero,teacher_pay,weatherization}_benchmark.py \
  test/integration/test_{rolling_controls,rolling_reminders,jtpa_holdout}.py \
  test/validation/test_{effect_forecast,rolling_calibration,rolling_bootstrap_calibration}.py
```
