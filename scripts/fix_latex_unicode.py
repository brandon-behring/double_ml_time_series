#!/usr/bin/env python3
"""
Fix Unicode characters and LaTeX packages in Pandoc-generated LaTeX.

Replaces common Unicode math symbols with LaTeX commands and adds
necessary packages for proper compilation.
"""

import re
import sys
from pathlib import Path


# Unicode → LaTeX replacements
UNICODE_REPLACEMENTS = {
    "∎": r"$\qed$",  # QED symbol
    "≥": r"$\geq$",  # Greater than or equal
    "≤": r"$\leq$",  # Less than or equal
    "≈": r"$\approx$",  # Approximately equal
    "≠": r"$\neq$",  # Not equal
    "→": r"$\to$",  # Right arrow
    "←": r"$\leftarrow$",  # Left arrow
    "⇒": r"$\Rightarrow$",  # Double right arrow
    "⇔": r"$\Leftrightarrow$",  # Double arrow both ways
    "∈": r"$\in$",  # Element of
    "∉": r"$\notin$",  # Not element of
    "⊂": r"$\subset$",  # Subset
    "⊆": r"$\subseteq$",  # Subset or equal
    "∪": r"$\cup$",  # Union
    "∩": r"$\cap$",  # Intersection
    "∅": r"$\emptyset$",  # Empty set
    "∀": r"$\forall$",  # For all
    "∃": r"$\exists$",  # There exists
    "∞": r"$\infty$",  # Infinity
    "∑": r"$\sum$",  # Summation
    "∏": r"$\prod$",  # Product
    "∫": r"$\int$",  # Integral
    "±": r"$\pm$",  # Plus-minus
    "×": r"$\times$",  # Multiplication
    "÷": r"$\div$",  # Division
    "√": r"$\sqrt{}$",  # Square root
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ε": r"$\varepsilon$",
    "θ": r"$\theta$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "π": r"$\pi$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "φ": r"$\phi$",
    "χ": r"$\chi$",
    "ψ": r"$\psi$",
    "ω": r"$\omega$",
    "Δ": r"$\Delta$",
    "Θ": r"$\Theta$",
    "Λ": r"$\Lambda$",
    "Σ": r"$\Sigma$",
    "Φ": r"$\Phi$",
    "Ψ": r"$\Psi$",
    "Ω": r"$\Omega$",
}


def fix_unicode(content: str) -> str:
    """Replace Unicode characters with LaTeX commands."""
    for unicode_char, latex_cmd in UNICODE_REPLACEMENTS.items():
        content = content.replace(unicode_char, latex_cmd)
    return content


def add_packages(content: str) -> str:
    """Add necessary LaTeX packages after \\documentclass."""
    # Find the \\begin{document} line
    begin_doc_match = re.search(r"\\begin\{document\}", content)
    if not begin_doc_match:
        print(
            "Warning: Could not find \\begin{document}, skipping package additions", file=sys.stderr
        )
        return content

    packages = r"""
% Additional packages for proper rendering
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{mathtools}

"""

    # Insert packages just before \\begin{document}
    insert_pos = begin_doc_match.start()
    return content[:insert_pos] + packages + content[insert_pos:]


def fix_equation_nesting(content: str) -> str:
    """Fix common equation nesting issues from Pandoc conversion."""
    # Remove extra \\[ \\] inside align environments (common Pandoc issue)
    content = re.sub(r"(\\begin\{align[*]?\})\s*\\\[\s*", r"\1\n", content)
    content = re.sub(r"\s*\\\]\s*(\\end\{align[*]?\})", r"\n\1", content)
    return content


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <latex_file>", file=sys.stderr)
        print("  Fixes Unicode characters and adds necessary packages in-place", file=sys.stderr)
        sys.exit(1)

    latex_file = Path(sys.argv[1])

    if not latex_file.exists():
        print(f"Error: File not found: {latex_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Fixing LaTeX file: {latex_file}")

    # Read content
    content = latex_file.read_text(encoding="utf-8")

    # Apply fixes
    content = fix_unicode(content)
    content = add_packages(content)
    content = fix_equation_nesting(content)

    # Write back
    latex_file.write_text(content, encoding="utf-8")

    print(f"  ✓ Fixed Unicode characters")
    print(f"  ✓ Added LaTeX packages")
    print(f"  ✓ Fixed equation nesting")
    print(f"  Saved to: {latex_file}")


if __name__ == "__main__":
    main()
