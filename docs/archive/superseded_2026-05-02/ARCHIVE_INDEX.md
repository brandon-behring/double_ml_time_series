# Superseded Docs Archive

Archived: 2026-05-02

These files were moved out of `docs/` root during truth-first remediation. They
are historical evidence only. Use `docs/CURRENT_STATUS.md` and
`docs/audits/2026-05-02_repo_quality_methodology_audit.md` for current status.

## Why These Files Were Archived

The root-level docs mixed old audits, roadmaps, critique responses, status
files, and implementation plans. Several files contained claims that were
stale, conflicting, or ahead of runtime behavior, including:

- Project-complete claims that conflicted with active content expansion and
  remediation needs.
- DynamicDML claims that implied recursive dynamic g-estimation.
- Cross-implementation validation claims after the relevant module had been
  removed.
- TemporalCV integration claims that were never reflected in the code.
- Production-grade claims that exceeded the book-companion target.

## Use Policy

- Treat these files as historical context, not implementation instructions.
- Revalidate any methodology or status claim against live code, tests, and the
  current audit before using it.
- Do not add new current-status reports to this archive; put new canonical
  status in `docs/CURRENT_STATUS.md` or a new file under `docs/audits/`.
