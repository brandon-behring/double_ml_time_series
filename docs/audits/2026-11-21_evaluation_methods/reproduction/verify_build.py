"""Describe completed local publication builds and test logs in a compact receipt."""

import argparse
import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions():
    result = {}
    for name in [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "temporalcv",
        "econml",
        "sphinx",
        "pytest",
    ]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def verify(run):
    editions = run / "editions"
    build_log = (editions / "main.log").read_text()
    page_match = re.search(r"Output written on main\.pdf \((\d+) pages", build_log)
    if not page_match or "There were undefined references" in build_log:
        raise ValueError("Final PDF build is missing or has unresolved references")
    pdf = editions / "main.pdf"
    chapter = editions / "web/dist/chapters/chapter09-heterogeneity/index.html"
    for doc in [chapter.read_text(), (editions / "book.txt").read_text()]:
        if "policy gain" not in doc.lower() or "held-out" not in doc.lower():
            raise ValueError("Chapter 9 text missing in rendered edition")
    web = (run / "web-build.log").read_text()
    sphinx = (run / "sphinx.log").read_text()
    if "10 chapter(s) checked" not in web or "build succeeded." not in sphinx:
        raise ValueError("Publication build is incomplete")
    source_files = [
        "chapters/chapter_02.tex",
        "chapters/chapter_03.tex",
        "chapters/chapter_09.tex",
        "web/src/content/chapters/chapter02-neyman-orthogonality-dml.mdx",
        "web/src/content/chapters/chapter03-validation-framework.mdx",
        "web/src/content/chapters/chapter09-heterogeneity.mdx",
    ]
    source = {f: {"current": digest(ROOT / f), "built": digest(editions / f)} for f in source_files}
    tests = {}
    for name in [
        "current-core-pytest",
        "complete-suite",
        "edition-drift-tests",
        "temporalcv-sibling-tests",
        "monitor-tests",
        "effect-forecast-tests",
        "audit-contract-tests",
    ]:
        log = run / (name + ".log")
        summaries = (
            [
                line.strip("= ")
                for line in log.read_text().splitlines()
                if re.search(r"(?:passed|failed|skipped|error).* in [\d.]+s", line)
            ]
            if log.exists()
            else []
        )
        tests[name] = summaries[-1] if summaries else "pending"
    return {
        "versions_in_verification_environment": package_versions(),
        "pdf": {"pages": int(page_match[1]), "sha256": digest(pdf), "unresolved_references": False},
        "web": {
            "academic_chapters": 10,
            "pages": int(re.search(r"(\d+) page\(s\) built", web)[1]),
            "chapter_9_sha256": digest(chapter),
        },
        "edition_source_files": source,
        "sphinx_strict": "passed",
        "ci_mypy": (run / "ci-mypy.log").read_text().strip(),
        "tests": tests,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.run), indent=2) + "\n")
