#!/usr/bin/env python3
"""
Convert remaining Pandoc code blocks to minted syntax.
Extracts clean Python code from verbose Pandoc Highlighting syntax.
"""

import re
import sys
from pathlib import Path
from typing import List


def extract_python_code(lines: List[str]) -> List[str]:
    """Extract clean Python code from Pandoc tokens."""
    code_lines: List[str] = []

    for line in lines:
        # Skip Highlighting wrapper lines and TODO comments
        if any(
            x in line
            for x in [
                r"\begin{Shaded}",
                r"\end{Shaded}",
                r"\begin{Highlighting}",
                r"\end{Highlighting}",
                "% TODO",
            ]
        ):
            continue

        # Remove all Pandoc token commands
        # Pattern: \TokenType{content}
        cleaned = re.sub(r"\\[A-Z][a-zA-Z]*Tok\{([^}]*)\}", r"\1", line)
        cleaned = re.sub(r"\\[A-Z][a-zA-Z]*\{([^}]*)\}", r"\1", cleaned)

        # Clean up common patterns
        cleaned = cleaned.replace(r"\textasciitilde{}", "~")
        cleaned = cleaned.replace(r"\textgreater{}", ">")
        cleaned = cleaned.replace(r"\textless{}", "<")
        cleaned = cleaned.replace(r"\textbackslash{}", "\\")
        cleaned = cleaned.replace(r"\_", "_")
        cleaned = cleaned.replace(r"\^{}", "^")
        cleaned = cleaned.replace(r"{-}", "-")
        cleaned = cleaned.replace(r"\{", "{")
        cleaned = cleaned.replace(r"\}", "}")
        cleaned = cleaned.replace(r"\textquotesingle{}", "'")
        cleaned = cleaned.replace(r"\#", "#")  # Fix escaped hashes

        if cleaned.strip():
            code_lines.append(cleaned)

    return code_lines


def convert_code_blocks(input_file: Path, output_file: Path) -> None:
    """Convert all code blocks in file."""
    lines = input_file.read_text(encoding="utf-8").splitlines(keepends=True)

    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for start of code block
        if r"\begin{Shaded}" in line:
            # Skip TODO comment if present
            if i > 0 and "TODO" in lines[i - 1]:
                result.pop()  # Remove TODO line

            # Collect code block lines
            code_block = []
            i += 1
            while i < len(lines) and r"\end{Shaded}" not in lines[i]:
                code_block.append(lines[i])
                i += 1

            # Convert to minted
            clean_code = extract_python_code(code_block)
            result.append(r"\begin{minted}{python}" + "\n")
            for code_line in clean_code:
                result.append(code_line)
            result.append(r"\end{minted}" + "\n")

            i += 1  # Skip \end{Shaded}
        else:
            result.append(line)
            i += 1

    output_file.write_text("".join(result), encoding="utf-8")
    print(f"✓ Converted code blocks: {input_file} → {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.tex> <output.tex>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    convert_code_blocks(input_path, output_path)
