# Did the Indonesian teacher-pay effect on learning persist for another school year?

Accelerating teacher certification changed student test scores by **−0.0055 standard deviations** in the 2011 follow-up and **+0.0104** in 2012. Carrying the first estimate forward missed the later estimate by **−0.0158 SD**, with a 95% interval of **−0.0448 to +0.0132**. Predicting zero missed by **−0.0104 SD**. Both estimates concern the schools assigned in the original lottery. The two surveys give evidence about persistence over another year of the same reform. [Published study, Table V](https://doi.org/10.1093/qje/qjx040) · [Replication data and programs](https://doi.org/10.7910/DVN/MTVM5O)

## From the lottery to the two follow-ups

In 2009, one school in each of 120 matched triplets received accelerated access to certification for eligible teachers. Certification raised base pay through an allowance. Teachers in control schools could still qualify through the ordinary process. The estimate measures the effect on students of attending a school offered accelerated certification. Individual teachers' pay changes vary with their eligibility and take-up. [Published paper, section III and Figure II](https://research.vu.nl/files/127419178/Double_for_nothing_Experimental_evidence_on_an_unconditional_teacher_salary_increase_in_Indonesia.pdf)

The November 2009 baseline followed the eligibility letters. Test scores were subsequently collected in April–May 2011 and April–May 2012. Taking April 1, 2012 as the cutoff before the later survey gives the study operator ten months after the last possible month of 2011 fieldwork to score the tests and prepare an estimate. The individual scoring and delivery dates are undocumented. A full twelve-month delay from the end of 2011 fieldwork would put the completed first-wave estimates beyond this cutoff. The public replication archive first appeared in October 2017. An earlier public working paper was issued in December 2015. The 2011 estimate frozen here is a retrospective reconstruction. [Study timeline, section III.B](https://research.vu.nl/files/127419178/Double_for_nothing_Experimental_evidence_on_an_unconditional_teacher_salary_increase_in_Indonesia.pdf) · [Archive publication record](https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId=doi:10.7910/DVN/MTVM5O) · [Working paper](https://www.nber.org/papers/w21806)

## Published results and sensitivity to the analysis sample

| Model and schools | 2011 effect (SE) | 2012 effect (SE) | 2011 minus 2012, 95% interval |
|---|---:|---:|---:|
| Published specification, all available | −0.0055 (0.0244) | +0.0104 (0.0259) | −0.0158 [−0.0448, +0.0132] |
| Published specification, 348 schools observed twice | +0.0045 (0.0243) | +0.0116 (0.0260) | −0.0071 [−0.0341, +0.0199] |
| All available, omit post-letter baseline controls | −0.0065 (0.0422) | +0.0205 (0.0408) | −0.0270 [−0.0633, +0.0093] |

The first row reproduces both Table V coefficients, their school-clustered standard errors, the reported R² values (0.28 and 0.24), and the exact observation counts (279,066 and 274,993). The authors stack student–subject records, apply the archived weights, and absorb assignment-triplet fixed effects. They adjust for the individual and school baseline scores and accompanying missing-value indicators. One indicator is constant within triplets and absorbed. The dependent variable measures achievement in contemporaneous control standard deviations by subject, grade and year. Later control results enter the later outcome scale. Neither the first-wave regression nor the frozen forecast uses those results. [Table V and notes](https://research.vu.nl/files/127419178/Double_for_nothing_Experimental_evidence_on_an_unconditional_teacher_salary_increase_in_Indonesia.pdf) · [Archived author code](https://dataverse.harvard.edu/api/access/datafile/3050669?format=original)

The 2011 sample contained 79,757 students in 354 schools and 118 triplets. The 2012 sample contained 78,243 students in 351 schools and 117 triplets. In total, 348 schools and 53,504 student IDs recur, and every recurring student ID stays in the same school. Combining school-level contributions from both regressions gives a standard error of **0.0147** for the difference. Treating waves as independent would give **0.0356**. The joint calculation applies each wave's finite-sample correction, then uses the smaller school count minus one for the interval's t critical value. Restricting both waves to common schools uses later participation information, so the second row diagnoses changes in the sample. [Calculation](../external/forward/teacher_pay.py)

## Reproduce

```bash
uv run --locked python docs/audits/2026-11-19_time_series_dml/external/forward/teacher_pay.py \
  --cache-dir "${DML_TS_BENCHMARK_CACHE:-$HOME/.cache/dml-ts/external}"
```

The script verifies the original zipped Stata files by SHA-256 and checks the author program's Table V specifications. The tests compare the weighted estimator with an explicit dummy-variable regression, perturb the later outcomes to verify the earlier fit stays frozen, and exercise shared and disjoint school samples. [Integration tests](../../../../test/integration/test_teacher_pay_benchmark.py)
