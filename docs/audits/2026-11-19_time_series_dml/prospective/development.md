# How did half-history perform in the experiments we had already seen?

Half the earlier randomized effect came closer than DML to the later randomized estimate in **all four fully evaluable reminder quarters**, and in both JTPA reporting-lag comparisons. Zero came closest on all four reminder quarters and the JTPA holdout. These episodes helped motivate the half-history rule. Their rankings are retrospective. [All forecasts, gains and source digests](development.json)

## Activity reminders

The target was the change in log(next-hour steps + 0.5) under assignment to an activity prompt versus no prompt. For a date with mature earlier data, the new comparator is exactly one-half of the original historical arm difference, on that log scale. The earlier DML procedure moved the forecast toward zero at each of the four shared dates. Half-history reduced its magnitude further. [Original study design and seven dates](../rolling/reminders.md)

| Issuance | Historical | Half-history | DML | Later randomized contrast | DML vs. half, conditional 95% result |
|---|---:|---:|---:|---:|---|
| 2022-07-01 | −0.0450 | −0.0225 | −0.0352 | +0.0728 | Half closer |
| 2022-10-01 | +0.0371 | +0.0185 | +0.0309 | −0.0276 | Unresolved |
| 2023-01-01 | +0.0424 | +0.0212 | +0.0370 | −0.0065 | Unresolved |
| 2023-04-01 | +0.0271 | +0.0136 | +0.0263 | −0.0313 | Unresolved |

The date-specific interval holds both forecast numbers fixed and varies the later randomized estimate by resampling whole participants. Equal squared error occurs at the midpoint of the two forecasts. A 95% target interval wholly on one side resolves which number is closer under that interval procedure. These are pointwise comparisons. Two earlier dates had no mature history; a fifth date with mature history had only one new participant, whose estimate was closer to DML than to half-history. The machine-readable schedule retains those three dates. [Original forecast freezes](../rolling/mybp-freezes/) · [Replay code](../external/forward/shrinkage_replay.py)

| Method | Mean squared error across the four shared quarters (log units squared) |
|---|---:|
| Zero | 0.001771 |
| Half-history | 0.003498 |
| DML | 0.005072 |
| Full history | 0.005968 |

The mean paired advantage of DML over half-history was **−0.001574 log units squared**, with a 95% interval of **−0.008428 to +0.000748**. This second calculation refits the earlier ridge models and recomputes all four forecasts and targets using 999 common participant-weight draws across the four fixed dates. It accounts for reuse of participants in later training histories. It leaves substantial uncertainty about the ranking of the methods in repeated samples. [Recomputed losses and intervals](development.json)

**How much precision would distinguish these forecasts?** If one forecast were exactly correct, the later benchmark would need an SE of **0.00221–0.00282** for 80% probability of a two-sided 95% interval resolving the ranking in its favor, under a normal approximation. The four benchmark SEs were **0.04459–0.05392** log units. A realized effect far beyond both forecasts can still yield a resolved ranking, as it did in July 2022. Judging small differences in accuracy requires more precision than catching a large miss.

## Job-training offers

For the 1,832 adults with usable holdout earnings, the offer-versus-control difference was **−$14.29**, SE **$61.77**. The primary reporting allowance is one quarter. The two-quarter allowance changes the earlier training cohort. All comparisons below score the **same** later cohort. The nominal USD² gain is the squared error of half-history minus the squared error of DML. The conditional intervals propagate the original independent-arm Neyman uncertainty of that common holdout. [JTPA source and ascertainment audit](../rolling/jtpa.md)

| Source treatment | Reporting allowance | Historical | Half-history | DML | Gain of DML over half, USD² (95% interval) |
|---|---:|---:|---:|---:|---:|
| Original issuance | 1 quarter | +$424.69 | +$212.34 | +$341.54 | −75,252 [−106,537, −43,967] |
| Original issuance | 2 quarters | +$694.60 | +$347.30 | +$524.62 | −159,672 [−202,609, −116,734] |
| Post-unblinding source correction | 1 quarter | +$131.34 | +$65.67 | +$138.45 | −16,935 [−34,559, +689] |
| Post-unblinding source correction | 2 quarters | +$173.56 | +$86.78 | +$177.13 | −26,425 [−48,303, −4,547] |

A missing-wage code inflated the original historical and DML forecasts. The original numeric forecasts remain archived as issued. Half-history here is derived retrospectively even for those rows. Correcting the source field after unblinding put the historical and DML forecasts closer together and reversed their ranking against each other. For the corrected primary half-history versus DML comparison the interval includes equal loss. [Preserved initial issuance](../rolling/jtpa-issuance.json) · [Source-coding sensitivity](../rolling/jtpa-corrected-results.json)

## Reproduction

The replay verifies each source forecast's SHA-256, retains its original numerical value, and attaches the separate half-history estimate. Reminder target uncertainty and joint refits use the pinned participant data. JTPA uses the two archived scored results from the original-source replay. Run from the repository root.

```bash
audit=docs/audits/2026-11-19_time_series_dml
OPENBLAS_NUM_THREADS=1 uv run --locked python "$audit/external/forward/shrinkage_replay.py" \
  --cache-dir /tmp/dml-external-bench-cache
```
