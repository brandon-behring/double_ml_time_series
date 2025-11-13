#!/usr/bin/env python3
"""
Clean up Pandoc LaTeX output for inclusion in amsbook document.

Removes standalone preamble, fixes citations, marks sections needing manual work.
"""

import re
import sys
from pathlib import Path
from typing import List


def clean_citations(line: str) -> str:
    r"""Fix broken Pandoc citations: cite:{[}author{]} → \cite{author}"""
    # Pattern: cite:{[}author1,author2{]}
    line = re.sub(r"cite:\{\[\}([^{]+)\{\]\}", r"\\cite{\1}", line)
    return line


def is_preamble_line(line: str) -> bool:
    """Check if line is part of standalone preamble we don't need."""
    preamble_markers = [
        r"\documentclass",
        r"\usepackage",
        r"\PassOptionsToPackage",
        r"\newcommand{\Alert",  # Pandoc syntax highlighting commands
        r"\newcommand{\Annotation",
        r"\newcommand{\Attribute",
        r"\DefineVerbatimEnvironment",
        r"\newenvironment{Shaded}",
        r"\title{Chapter",
        r"\subtitle{",
        r"\author{}",
        r"\date{",
        r"\maketitle",
        r"\begin{document}",
        r"\end{document}",
        r"\makeatletter",
        r"\makeatother",
        r"\providecommand",
        r"\setlength{\parindent}",
        r"\setcounter{secnumdepth}",
        r"\\IfFileExists",
        r"\\ifPDFTeX",
        r"\\else",
        r"\\fi",
    ]

    for marker in preamble_markers:
        if marker in line:
            return True
    return False


def add_manual_work_markers(lines: List[str]) -> List[str]:
    """Add TODO comments for sections needing manual conversion."""
    result = []
    in_definition = False

    for line in lines:
        # Mark bold definitions that should be \begin{definition}
        if re.search(r"\\textbf\{Definition \d+\.\d+", line):
            result.append("% TODO: Convert to \\begin{definition}[...]\n")
            in_definition = True

        # Mark bold theorems
        elif re.search(r"\\textbf\{Theorem \d+\.\d+", line):
            result.append("% TODO: Convert to \\begin{theorem}[...]\n")

        # Mark bold examples
        elif re.search(r"\\textbf\{Example \d+\.\d+", line):
            result.append("% TODO: Convert to \\begin{example}[...]\n")

        # Mark code blocks that should be minted
        elif r"\begin{Shaded}" in line or r"\begin{Highlighting}" in line:
            result.append("% TODO: Convert to \\begin{minted}{python}...\\end{minted}\n")

        result.append(line)

    return result


def clean_latex_math(line: str) -> str:
    """Clean up LaTeX math formatting."""
    # Convert \(...\) to $...$ for inline math (more readable)
    # Actually, keep \(...\) as it's the LaTeX2e standard

    # Fix \text{...} spacing issues
    line = re.sub(r"\\text\{Outcome unit \}", r"\\text{Outcome unit }", line)
    line = re.sub(
        r"\\text\{ would have if assigned treatment \}",
        r"\\text{ would have if assigned treatment }",
        line,
    )

    return line


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_pandoc.tex> <output_clean.tex>", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Cleaning Pandoc output: {input_file}")

    # Read input
    lines = input_file.read_text(encoding="utf-8").splitlines(keepends=True)

    # Find first \section (content starts here)
    content_start = None
    for i, line in enumerate(lines):
        if line.startswith(r"\section{"):
            content_start = i
            break

    if content_start is None:
        print("Warning: No \\section found in input", file=sys.stderr)
        content_start = 0

    # Process only content lines
    content_lines = lines[content_start:]

    # Clean up
    cleaned = []
    for line in content_lines:
        # Skip preamble-like lines even in content
        if is_preamble_line(line):
            continue

        # Fix citations
        line = clean_citations(line)

        # Clean math
        line = clean_latex_math(line)

        cleaned.append(line)

    # Add manual work markers
    cleaned = add_manual_work_markers(cleaned)

    # Add header comment
    header = f"""% Generated from Pandoc output with automated cleanup.
%
% MANUAL WORK NEEDED:
%   - Convert bold definitions to \\begin{{definition}}[...]
%   - Convert bold theorems to \\begin{{theorem}}[...]
%   - Convert bold examples to \\begin{{example}}[...]
%   - Convert code blocks to \\begin{{minted}}{{python}}...\\end{{minted}}
%   - Review all TODO comments below
%
% See chapters/chapter_template.tex for examples.

"""

    # Write output
    output_file.write_text(header + "".join(cleaned), encoding="utf-8")

    print(f"  ✓ Cleaned {len(lines)} lines → {len(cleaned)} lines")
    print(f"  ✓ Removed preamble (first {content_start} lines)")
    print(f"  ✓ Fixed citations")
    print(f"  ✓ Added manual work markers")
    print(f"  Saved to: {output_file}")
    print()
    print("Next steps:")
    print("  1. Review the output file")
    print("  2. Search for 'TODO' comments")
    print("  3. Convert marked sections to proper LaTeX environments")
    print("  4. Replace code blocks with minted")


if __name__ == "__main__":
    main()
