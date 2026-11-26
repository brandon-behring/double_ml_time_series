# Audit Reports

This directory holds current evidence-based audits for the repository.

## Current audits

- [2026-11-22 Mathematical review of assumptions, estimands, learners, folds and inference](2026-11-22_mathematical_review/README.md)
- [2026-11-22 ACIC coverage study across 3,200 known-effect datasets](2026-11-22_acic_coverage/README.md)
- [2026-11-21 Adversarial audit of estimands, learners, forward evaluation and uncertainty](2026-11-21_evaluation_methods/README.md)
- [2026-11-19 Time-series DML audit and empirical replications](2026-11-19_time_series_dml/README.md)
- [Classical benchmarks and time-forward validation in simulations and pricing experiments](2026-11-19_time_series_dml/README.md#classical-benchmarks-and-time-forward-validation)
- [When does DML earn its place? Adversarial review and matched comparisons](2026-11-19_time_series_dml/adversarial/README.md)
- [Rolling causal-effect forecasts, calibration and held-out evidence](2026-11-19_time_series_dml/rolling/README.md)
- [Prospective effect forecasts for cricket and the General Social Survey](2026-11-19_time_series_dml/prospective/README.md)

The collection covers the working checkout and committed main, the manuscript and
web edition, fresh verification results, the research ledgers, four public-data pilots, and ten [external published-result replays](2026-11-19_time_series_dml/11-external-replication-benchmarks.md). The [substantive reanalyses](2026-11-19_time_series_dml/12-historical-causal-questions.md) examine what those designs can teach about five historical causal questions. The [forward-validation study](2026-11-19_time_series_dml/13-forward-causal-validation.md) compares frozen effects with later experiments and checks what data were actually available. All 600 joint 401(k) refits completed on November 20. The opening
report orders the findings by manuscript revision, research
priority and software readiness.

The [public-data comparison](2026-11-19_time_series_dml/README.md#four-further-tests-with-public-research-data) adds results on retail recommendations, bracero exclusion, teacher pay and weatherization.

Not every file these folders refer to is in the repository, and the reports describe the
November working copy, not `main`. Twenty-eight files of these folders that depend on the
pending redesign of `PanelDML` are held and are listed in
[`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md), which also says what the reports claim about
Chapters 6 and 8 that `main` does not yet hold. Three files that reproduce a provider's rows
without a licence that grants redistribution are held and are listed, with the licences and
attributions of the files that are here, in [`DATA_SOURCES.md`](DATA_SOURCES.md). That note
also names four files that are here by the owner's decision although their sources state no
licence that grants redistribution.

## Empirical and simulation checks

The [public-data experiments](2026-11-21_public_data_extensions/README.md) use bicycle demand, household electricity offers, ride counts and a repeated cooperation game to test recovery of known effects, treatment timing and inference under shared assignment. The collection includes the analysis scripts, reference estimates and input checksums.

## Superseded Audits

- [2026-05-30 Repo Quality And Methodology Audit](2026-05-30_repo_quality_methodology_audit.md)
  records the May gates and remediation. Its single-chapter web baseline,
  pre-migration namespace and deferred dynamic estimator belong to that snapshot.
- [2026-05-02 Repo Quality And Methodology Audit](2026-05-02_repo_quality_methodology_audit.md)
  — the first truth-first audit (findings F1–F16). Superseded by the 2026-05-30 audit;
  kept in place for history and because web Chapter 1 provenance references its path.
  13 of its 16 findings are now CLOSED.

## Future Audit Rule

When a newer audit supersedes this one:

1. Add the new file as `YYYY-MM-DD_short_scope.md`.
2. Update this README's list of current audits.
3. State which older claims are still valid and which are superseded.
4. Keep command evidence, source links, unresolved questions, and assumptions in
   the audit itself.
