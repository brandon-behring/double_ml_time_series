"""Check report completion, numerical tables, and frozen-evidence rebuild."""
import csv

import summarize_401k as audit


def test_600_joint_jobs_render_and_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    raw = tmp_path / "checkpoints"
    raw.mkdir()
    report = tmp_path / "07-401k-replication.md"
    template = ("# Audit\n\n<!-- begin numeric output -->\n\n<!-- end numeric output -->"
                "\n\n## Parallel execution\n\nKeep the surrounding report intact.\n")
    report.write_text(template)
    for (panel, folds), targets in audit.TARGETS.items():
        for part in range(1, 101):
            path = raw / f"all-{panel}-k{folds}-r{part:03d}.csv"
            with path.open("w", newline="") as stream:
                rows = csv.DictWriter(stream, ("panel", "folds", "partition", "method", "estimate", "se", "seconds"))
                rows.writeheader()
                for index, (method, (theta, se, _)) in enumerate(zip(audit.METHODS, targets)):
                    rows.writerow({"panel": panel, "folds": folds, "partition": part, "method": method,
                                   "estimate": theta + index + 1, "se": se + 1, "seconds": 123.4})

    draws = audit.get_draws(raw)
    summary = audit.summarize(draws)
    bundles = audit.bundle_counts(draws)
    assert len(draws) == 4200
    assert all(bundles[key] == 100 for key in audit.TARGETS)
    assert len(summary) == 42 and all(r["finite"] == 100 and r["source"] == "joint" for r in summary)
    audit.render(summary, bundles)
    rendered = report.read_text()
    assert "600 of 600 joint jobs" in rendered
    assert "42 from joint runs and 0 from single-learner runs" in rendered
    assert "median absolute difference from publication was $4.00" in rendered
    assert "largest was $7.00" in rendered
    assert rendered.partition("<!-- begin numeric output -->")[0] == template.partition("<!-- begin numeric output -->")[0]
    assert rendered.partition("<!-- end numeric output -->")[2] == template.partition("<!-- end numeric output -->")[2]
    for metric in ("estimate", "median_se", "adjusted_se"):
        table = audit.comparison_table(summary, metric)
        assert table in rendered and table.count(" / ") == 42

    paths = [report, tmp_path / "401k-summary.csv", tmp_path / "401k-draws.csv.gz"]
    snapshot = [p.read_bytes() for p in paths]
    stored = audit.get_draws(None)
    audit.render(audit.summarize(stored), audit.bundle_counts(stored))
    assert [p.read_bytes() for p in paths] == snapshot

    for row in draws:
        row["run"] = "single"
        row["origin"] = "single"
    singles = audit.summarize(draws)
    assert all(row["finite"] == 100 for row in singles)
    assert not audit.bundle_counts(draws)
    audit.render(singles, audit.bundle_counts(draws))
    assert "0 of 600 joint jobs" in report.read_text()
    assert "0 from joint runs and 42 from single-learner runs" in report.read_text()
    assert not audit.bundle_counts([dict(row, origin="component") for row in stored])
    assert sum(audit.bundle_counts(stored[:-1]).values()) == 599
    stored[-1]["se"] = "NaN"
    assert sum(audit.bundle_counts(stored).values()) == 599
