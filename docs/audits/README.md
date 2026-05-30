# Audit Reports

This directory holds current evidence-based audits for the repository.

## Current Canonical Audit

- [2026-05-30 Repo Quality And Methodology Audit](2026-05-30_repo_quality_methodology_audit.md)

This audit is the current truth source for repo quality, methodology, organization,
docs/API truthfulness, the `web/` companion, and the forward-roadmap pointer. Every
gate in it was re-run live on 2026-05-30. Older reports in `docs/`, `docs/archive/`,
and `docs/plans/archive/` remain useful historical evidence, but they should not be
treated as current status unless a current audit or roadmap revalidates them.

## Superseded Audits

- [2026-05-02 Repo Quality And Methodology Audit](2026-05-02_repo_quality_methodology_audit.md)
  — the first truth-first audit (findings F1–F16). Superseded by the 2026-05-30 audit;
  kept in place for history and because web Chapter 1 provenance references its path.
  13 of its 16 findings are now CLOSED.

## Future Audit Rule

When a newer audit supersedes this one:

1. Add the new file as `YYYY-MM-DD_short_scope.md`.
2. Update this README so there is exactly one current canonical audit.
3. State which older claims are still valid and which are superseded.
4. Keep command evidence, source links, unresolved questions, and assumptions in
   the audit itself.
