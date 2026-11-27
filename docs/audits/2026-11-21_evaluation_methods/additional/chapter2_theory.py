"""Replay Chapter 2's CATE, omitted-confounder and matched learner examples in both editions."""

import argparse
import contextlib
import io
import json
import re
from importlib.metadata import version
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]


def read_listings(path):
    text = path.read_text()
    if path.suffix == ".tex":
        code_pattern = r"\\begin\{minted\}\{python\}\n(.*?)\n\\end\{minted\}"
        output_pattern = r"\\begin\{verbatim\}\n(.*?)\n\\end\{verbatim\}"
    else:
        code_pattern = r"^```python\n(.*?)\n```"
        output_pattern = r"^```text\n(.*?)\n```"
    return re.findall(code_pattern, text, re.M | re.S), re.findall(
        output_pattern, text, re.M | re.S
    )


def replay(path, check):
    code, printed = read_listings(path)
    keys = {
        "income_grid =": "profile_cate",
        "delta_fit =": "confounder",
        "crossfit = double_ml": "exercise_setup",
        "same_sample =": "exercise_split",
        '"Deep tree": DecisionTreeRegressor': "exercise_learners",
    }
    measured = {}
    exercises = {}
    for listing in code:
        for marker, key in keys.items():
            if marker not in listing:
                continue
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                exec(
                    compile(listing, f"{path.name}:{key}", "exec"),
                    exercises if key.startswith("exercise") else {},
                )
            result = captured.getvalue().strip()
            if result and check and result not in printed:
                raise ValueError(
                    f"Measured {key} output differs from the printed {path.name} result"
                )
            measured[key] = result
    if set(measured) != set(keys.values()):
        raise ValueError(f"Missing listings in {path}")
    return measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        BASE / "chapters/chapter_02.tex",
        next((BASE / "web/src/content/chapters").glob("chapter02*.mdx")),
    ]
    a, b = [replay(path, check=True) for path in paths]
    if a != b:
        raise ValueError("Chapter 2 editions produced different outputs")
    args.output.write_text(
        json.dumps(
            {
                "econml": version("econml"),
                "sklearn": version("scikit-learn"),
                "matched_editions": [path.name for path in paths],
                "results": a,
            },
            indent=2,
        )
        + "\n"
    )
