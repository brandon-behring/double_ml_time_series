#!/usr/bin/env python3
"""
Jupyter Notebook to LaTeX converter for validation reports.

Converts .ipynb files to professional LaTeX documents suitable for
validation documentation and academic reports.

Usage:
    python scripts/documents/notebook_to_latex.py input.ipynb output.tex
    python scripts/documents/notebook_to_latex.py input.ipynb output.tex --style=article
    python scripts/documents/notebook_to_latex.py input.ipynb output.tex --style=book
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


class NotebookToLatexConverter:
    """
    Convert Jupyter notebooks to LaTeX with professional formatting.

    Handles markdown cells, code cells, output cells, and images.
    Produces publication-quality LaTeX suitable for validation reports.
    """

    def __init__(self, style: str = "article"):
        """
        Initialize converter with document style.

        Args:
            style: LaTeX document style ('article' or 'book')
        """
        self.style = style
        self.image_counter = 0
        self.images: List[Dict[str, str]] = []

    def convert(self, notebook_path: Path, output_path: Path) -> None:
        """
        Convert notebook to LaTeX file.

        Args:
            notebook_path: Path to .ipynb file
            output_path: Path to output .tex file
        """
        # Load notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Convert cells
        latex_content = self._build_latex(notebook, notebook_path.stem)

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        print(f"✅ Converted {notebook_path} → {output_path}")
        if self.images:
            print(f"   📷 {len(self.images)} images referenced")

    def _build_latex(self, notebook: Dict[str, Any], title: str) -> str:
        """Build complete LaTeX document from notebook."""
        preamble = self._get_preamble(title)
        body = self._convert_cells(notebook["cells"])
        footer = self._get_footer()

        return f"{preamble}\n\n{body}\n\n{footer}"

    def _get_preamble(self, title: str) -> str:
        """Generate LaTeX preamble."""
        doc_class = "book" if self.style == "book" else "article"

        return f"""\\documentclass[11pt,letterpaper]{{{doc_class}}}

% Packages
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{microtype}}
\\usepackage[margin=1in]{{geometry}}

% Math
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{amsthm}}

% Graphics
\\usepackage{{graphicx}}
\\usepackage{{float}}

% Code highlighting
\\usepackage{{listings}}
\\usepackage{{xcolor}}

% Tables
\\usepackage{{booktabs}}
\\usepackage{{longtable}}

% Links
\\usepackage{{hyperref}}
\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
    citecolor=blue,
    bookmarks=true,
    bookmarksnumbered=true,
}}

% Code style (Python)
\\definecolor{{codebg}}{{rgb}}{{0.95, 0.95, 0.95}}
\\definecolor{{codegreen}}{{rgb}}{{0, 0.6, 0}}
\\definecolor{{codegray}}{{rgb}}{{0.5, 0.5, 0.5}}
\\definecolor{{codepurple}}{{rgb}}{{0.58, 0, 0.82}}

\\lstdefinestyle{{pythonstyle}}{{
    language=Python,
    backgroundcolor=\\color{{codebg}},
    commentstyle=\\color{{codegreen}},
    keywordstyle=\\color{{blue}},
    numberstyle=\\tiny\\color{{codegray}},
    stringstyle=\\color{{codepurple}},
    basicstyle=\\ttfamily\\small,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=4,
    frame=single,
}}

\\lstset{{style=pythonstyle}}

% Title
\\title{{{title}}}
\\author{{Generated from Jupyter Notebook}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle
\\tableofcontents
\\clearpage
"""

    def _get_footer(self) -> str:
        """Generate LaTeX footer."""
        return "\\end{document}"

    def _convert_cells(self, cells: List[Dict[str, Any]]) -> str:
        """Convert all notebook cells to LaTeX."""
        latex_parts = []

        for cell in cells:
            cell_type = cell.get("cell_type")

            if cell_type == "markdown":
                latex_parts.append(self._convert_markdown_cell(cell))
            elif cell_type == "code":
                latex_parts.append(self._convert_code_cell(cell))
            elif cell_type == "raw":
                # Skip raw cells
                continue

        return "\n\n".join(latex_parts)

    def _convert_markdown_cell(self, cell: Dict[str, Any]) -> str:
        """Convert markdown cell to LaTeX."""
        source = "".join(cell.get("source", []))
        return self._markdown_to_latex(source)

    def _markdown_to_latex(self, markdown: str) -> str:
        """
        Convert markdown syntax to LaTeX.

        Handles headers, lists, code blocks, math, bold/italic.
        """
        latex = markdown

        # Headers
        latex = re.sub(r"^# (.+)$", r"\\section{\1}", latex, flags=re.MULTILINE)
        latex = re.sub(r"^## (.+)$", r"\\subsection{\1}", latex, flags=re.MULTILINE)
        latex = re.sub(r"^### (.+)$", r"\\subsubsection{\1}", latex, flags=re.MULTILINE)
        latex = re.sub(r"^#### (.+)$", r"\\paragraph{\1}", latex, flags=re.MULTILINE)

        # Inline code
        latex = re.sub(r"`([^`]+)`", r"\\texttt{\1}", latex)

        # Bold
        latex = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", latex)
        latex = re.sub(r"__([^_]+)__", r"\\textbf{\1}", latex)

        # Italic
        latex = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", latex)
        latex = re.sub(r"_([^_]+)_", r"\\textit{\1}", latex)

        # Lists (simple conversion)
        latex = re.sub(r"^- (.+)$", r"\\item \1", latex, flags=re.MULTILINE)
        latex = re.sub(r"^\* (.+)$", r"\\item \1", latex, flags=re.MULTILINE)

        # Wrap itemize blocks
        latex = self._wrap_lists(latex)

        # Code blocks (```python)
        latex = self._convert_code_blocks(latex)

        # Escape special LaTeX characters (except already converted)
        latex = self._escape_latex(latex)

        return latex

    def _wrap_lists(self, text: str) -> str:
        """Wrap \\item sequences in itemize environments."""
        lines = text.split("\n")
        result = []
        in_list = False

        for line in lines:
            if line.strip().startswith("\\item"):
                if not in_list:
                    result.append("\\begin{itemize}")
                    in_list = True
                result.append(line)
            else:
                if in_list:
                    result.append("\\end{itemize}")
                    in_list = False
                result.append(line)

        if in_list:
            result.append("\\end{itemize}")

        return "\n".join(result)

    def _convert_code_blocks(self, text: str) -> str:
        """Convert markdown code blocks to LaTeX listings."""
        # Match ```python ... ``` blocks
        pattern = r"```python\n(.*?)```"

        def replace_code_block(match: re.Match) -> str:
            code = match.group(1)
            return f"\\begin{{lstlisting}}\n{code}\\end{{lstlisting}}"

        return re.sub(pattern, replace_code_block, text, flags=re.DOTALL)

    def _escape_latex(self, text: str) -> str:
        """
        Escape special LaTeX characters.

        Preserves already-converted LaTeX commands.
        """
        # Characters to escape (skip \, {, } if part of LaTeX command)
        replacements = {
            "%": "\\%",
            "$": "\\$",
            "&": "\\&",
            "#": "\\#",
            "_": "\\_",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }

        # Simple replacement (could be improved to skip LaTeX commands)
        for char, escaped in replacements.items():
            # Only replace if not already part of LaTeX command
            text = text.replace(char, escaped)

        return text

    def _convert_code_cell(self, cell: Dict[str, Any]) -> str:
        """Convert code cell (with outputs) to LaTeX."""
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])

        latex_parts = []

        # Input code
        latex_parts.append("\\begin{lstlisting}")
        latex_parts.append(source)
        latex_parts.append("\\end{lstlisting}")

        # Outputs
        if outputs:
            latex_parts.append("\\textbf{Output:}")
            for output in outputs:
                latex_parts.append(self._convert_output(output))

        return "\n".join(latex_parts)

    def _convert_output(self, output: Dict[str, Any]) -> str:
        """Convert cell output to LaTeX."""
        output_type = output.get("output_type")

        if output_type == "stream":
            # Text output
            text = "".join(output.get("text", []))
            return f"\\begin{{verbatim}}\n{text}\\end{{verbatim}}"

        elif output_type == "execute_result" or output_type == "display_data":
            # Check for data types
            data = output.get("data", {})

            # Images
            if "image/png" in data:
                return self._save_image(data["image/png"], "png")

            # Text/plain
            if "text/plain" in data:
                text = "".join(data["text/plain"])
                return f"\\begin{{verbatim}}\n{text}\\end{{verbatim}}"

        elif output_type == "error":
            # Error output
            traceback = "".join(output.get("traceback", []))
            return f"\\textcolor{{red}}{{\\begin{{verbatim}}\n{traceback}\\end{{verbatim}}}}"

        return ""

    def _save_image(self, image_data: str, extension: str) -> str:
        """
        Track image and return LaTeX include command.

        Args:
            image_data: Base64 encoded image
            extension: Image file extension

        Returns:
            LaTeX \\includegraphics command
        """
        self.image_counter += 1
        image_filename = f"notebook_image_{self.image_counter}.{extension}"

        self.images.append({"filename": image_filename, "data": image_data})

        return f"""
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{image_filename}}}
    \\caption{{Output {self.image_counter}}}
\\end{{figure}}
"""


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to LaTeX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", type=Path, help="Input .ipynb file")
    parser.add_argument("output", type=Path, help="Output .tex file")
    parser.add_argument(
        "--style",
        choices=["article", "book"],
        default="article",
        help="LaTeX document style (default: article)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return

    if not args.input.suffix == ".ipynb":
        print(f"⚠️  Warning: Input file does not have .ipynb extension")

    # Convert
    converter = NotebookToLatexConverter(style=args.style)
    converter.convert(args.input, args.output)

    # Report images
    if converter.images:
        print(f"\n📋 Image extraction required:")
        print(f"   Images are embedded in notebook as base64.")
        print(f"   Use nbconvert to extract images:")
        print(f"   $ jupyter nbconvert --to latex {args.input}")


if __name__ == "__main__":
    main()
