#!/usr/bin/env python3
"""
Fix Unicode characters in LaTeX files.

Replaces Unicode characters with proper LaTeX commands:
- Greek letters: τ → \tau, β → \beta, etc.
- Box drawing: ─├└ → verbatim or ASCII equivalents
- Math symbols: ≈ → \approx, ≥ → \geq, etc.
- Warning symbols: ⚠ → \textbf{WARNING:}
- Subscripts: ₁ → _1 (in math mode)
"""

import re
import sys
from pathlib import Path

# Unicode → LaTeX mappings
GREEK_LETTERS = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "θ": r"\theta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "τ": r"\tau",
    "σ": r"\sigma",
    "ϕ": r"\phi",
    "η": r"\eta",
}

MATH_SYMBOLS = {
    "≈": r"\approx",
    "≥": r"\geq",
    "≤": r"\leq",
    "→": r"\to",
    "∈": r"\in",
    "∞": r"\infty",
    "√": r"\sqrt",
    "∎": r"\qed",
}

SUBSCRIPTS = {
    "₀": r"_0",
    "₁": r"_1",
    "₂": r"_2",
    "₃": r"_3",
    "₄": r"_4",
}

# Box drawing → ASCII art replacements
BOX_DRAWING = {
    "─": "-",
    "│": "|",
    "├": "+",
    "└": "+",
    "┌": "+",
    "┐": "+",
    "┘": "+",
}

# Warning symbols → LaTeX text
WARNING_SYMBOLS = {
    "⚠": r"\textbf{WARNING:}",
    "⚠️": r"\textbf{WARNING:}",  # With variation selector
}


def fix_unicode_in_file(filepath: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Fix Unicode characters in a LaTeX file.

    Args:
        filepath: Path to .tex file
        dry_run: If True, only count replacements without modifying

    Returns:
        (total_replacements, unique_chars_replaced)
    """
    content = filepath.read_text(encoding="utf-8")
    original_content = content
    total_replacements = 0
    unique_chars = set()

    # Fix warning symbols (do first, since they're most disruptive)
    for unicode_char, latex_cmd in WARNING_SYMBOLS.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, latex_cmd)
            total_replacements += count
            unique_chars.add(unicode_char)
            print(f"  {unicode_char} → {latex_cmd}: {count} replacements")

    # Fix Greek letters (math mode)
    for unicode_char, latex_cmd in GREEK_LETTERS.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            # Use $ for inline math if not already in math mode
            # This is conservative - assumes standalone Greek letters need math mode
            content = content.replace(unicode_char, f"${latex_cmd}$")
            total_replacements += count
            unique_chars.add(unicode_char)
            print(f"  {unicode_char} → ${latex_cmd}$: {count} replacements")

    # Fix math symbols
    for unicode_char, latex_cmd in MATH_SYMBOLS.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, f"${latex_cmd}$")
            total_replacements += count
            unique_chars.add(unicode_char)
            print(f"  {unicode_char} → ${latex_cmd}$: {count} replacements")

    # Fix subscripts
    for unicode_char, latex_cmd in SUBSCRIPTS.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, f"${latex_cmd}$")
            total_replacements += count
            unique_chars.add(unicode_char)
            print(f"  {unicode_char} → ${latex_cmd}$: {count} replacements")

    # Fix box drawing (ASCII replacement)
    for unicode_char, ascii_char in BOX_DRAWING.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, ascii_char)
            total_replacements += count
            unique_chars.add(unicode_char)
            print(f"  {unicode_char} → {ascii_char}: {count} replacements")

    # Remove variation selectors (invisible Unicode modifiers)
    variation_selector = "\ufe0f"
    if variation_selector in content:
        count = content.count(variation_selector)
        content = content.replace(variation_selector, "")
        total_replacements += count
        unique_chars.add(variation_selector)
        print(f"  U+FE0F (variation selector) removed: {count} occurrences")

    # Write changes if not dry run
    if not dry_run and content != original_content:
        filepath.write_text(content, encoding="utf-8")
        print(f"  ✓ File updated: {filepath}")
    elif dry_run and content != original_content:
        print(f"  [DRY RUN] Would update: {filepath}")

    return total_replacements, len(unique_chars)


def main():
    """Fix Unicode in all chapter files."""
    dry_run = "--dry-run" in sys.argv

    chapter_files = [
        Path("chapters/chapter_01.tex"),
        Path("chapters/chapter_02.tex"),
        Path("chapters/chapter_03.tex"),
    ]

    print("=" * 70)
    print("Unicode → LaTeX Conversion")
    print("=" * 70)

    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified\n")

    total_replacements_all = 0
    total_unique_all = set()

    for chapter_file in chapter_files:
        if not chapter_file.exists():
            print(f"\n⚠️  File not found: {chapter_file}")
            continue

        print(f"\n📄 Processing: {chapter_file}")
        replacements, unique_chars = fix_unicode_in_file(chapter_file, dry_run=dry_run)

        total_replacements_all += replacements
        total_unique_all.update(range(unique_chars))  # Count unique types

        if replacements == 0:
            print("  ✓ No Unicode characters found")

    print("\n" + "=" * 70)
    print(f"Summary: {total_replacements_all} total replacements")
    print("=" * 70)

    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
