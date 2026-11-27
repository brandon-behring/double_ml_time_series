"""Execute Chapter 2's three linked PLR listings and compare the two editions."""

import argparse
import json
import os
import re
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]


def listings():
    tex = (ROOT / "chapters/chapter_02.tex").read_text()
    mdx = (ROOT / "web/src/content/chapters/chapter02-neyman-orthogonality-dml.mdx").read_text()
    books = (
        re.findall(r"\\begin\{minted\}\{python\}\n(.*?)\\end\{minted\}", tex, re.S),
        re.findall(r"^```python\n(.*?)^```", mdx, re.S | re.M),
    )
    for keyword in (
        "tau_dml = dml.ate(",
        "ci_lower, ci_upper = dml.ate_interval",
        "first_stage_T_r2.append",
    ):
        selected = []
        for book in books:
            blocks = [
                b
                for b in book
                if keyword in b
                and (keyword != "ci_lower, ci_upper = dml.ate_interval" or "CI_Width" in b)
            ]
            if len(blocks) != 1:
                raise ValueError(f"Expected one listing for {keyword}, found {len(blocks)}")
            selected.append(blocks[0].strip())
        if selected[0] != selected[1]:
            raise ValueError(f"Editions differ for {keyword}")
        yield selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    context = {}
    outputs = []
    with TemporaryDirectory() as tmp, threadpool_limits(limits=1):
        os.chdir(tmp)
        for source in listings():
            buffer = StringIO()
            with redirect_stdout(buffer):
                exec(compile(source, "chapter02-listing", "exec"), context)
            outputs.append(buffer.getvalue().strip())
        os.chdir(ROOT)
    result = {"econml": version("econml"), "sklearn": version("scikit-learn"), "stdout": outputs}
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
