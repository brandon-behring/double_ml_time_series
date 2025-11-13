#!/bin/bash
# Build high-quality PDFs from AsciiDoc chapters with perfect equation rendering
#
# Usage: ./scripts/build_chapter_pdf.sh chapters/chapter_01.adoc
#
# Workflow:
#   1. AsciiDoc → DocBook (asciidoctor)
#   2. DocBook → LaTeX (pandoc)
#   3. LaTeX → PDF (pdflatex, 2 passes for cross-references)
#
# This produces professional-quality PDFs with properly rendered LaTeX equations,
# unlike asciidoctor-pdf which has poor equation rendering.

set -e  # Exit on error

if [ $# -ne 1 ]; then
    echo "Usage: $0 <chapter.adoc>"
    echo "Example: $0 chapters/chapter_01.adoc"
    exit 1
fi

INPUT_FILE="$1"
BASENAME=$(basename "$INPUT_FILE" .adoc)
OUTPUT_DIR="output"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Building PDF for: $INPUT_FILE"
echo "=========================================="

# Step 1: AsciiDoc → DocBook
echo "[1/3] Converting AsciiDoc → DocBook XML..."
asciidoctor -b docbook -o "$OUTPUT_DIR/${BASENAME}.xml" "$INPUT_FILE"

if [ $? -ne 0 ]; then
    echo "ERROR: AsciiDoc → DocBook conversion failed"
    exit 1
fi

echo "  ✓ Created: $OUTPUT_DIR/${BASENAME}.xml"

# Step 2: DocBook → LaTeX
echo "[2/3] Converting DocBook → LaTeX..."
pandoc "$OUTPUT_DIR/${BASENAME}.xml" \
    -f docbook \
    -t latex \
    -o "$OUTPUT_DIR/${BASENAME}.tex" \
    --standalone \
    --number-sections \
    --toc \
    --toc-depth=3

if [ $? -ne 0 ]; then
    echo "ERROR: DocBook → LaTeX conversion failed"
    exit 1
fi

echo "  ✓ Created: $OUTPUT_DIR/${BASENAME}.tex"

# Step 2.5: Fix Unicode and add packages
echo "       Fixing Unicode characters and LaTeX packages..."
python3 scripts/fix_latex_unicode.py "$OUTPUT_DIR/${BASENAME}.tex"

if [ $? -ne 0 ]; then
    echo "ERROR: LaTeX fixing failed"
    exit 1
fi

# Step 3: LaTeX → PDF (first pass)
echo "[3/3] Compiling LaTeX → PDF (pass 1)..."
cd "$OUTPUT_DIR"
pdflatex -interaction=nonstopmode "${BASENAME}.tex" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: LaTeX compilation failed (pass 1)"
    echo "See $OUTPUT_DIR/${BASENAME}.log for details"
    exit 1
fi

# Second pass for cross-references (ignore errors from aux file issues)
echo "       Compiling LaTeX → PDF (pass 2 for cross-references)..."
pdflatex -interaction=nonstopmode "${BASENAME}.tex" > /dev/null 2>&1 || true

cd ..

PDF_SIZE=$(ls -lh "$OUTPUT_DIR/${BASENAME}.pdf" | awk '{print $5}')
PDF_PAGES=$(pdfinfo "$OUTPUT_DIR/${BASENAME}.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}')

echo "  ✓ Created: $OUTPUT_DIR/${BASENAME}.pdf ($PDF_SIZE, $PDF_PAGES pages)"
echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "   PDF: $OUTPUT_DIR/${BASENAME}.pdf"
echo "   LaTeX source: $OUTPUT_DIR/${BASENAME}.tex"
echo "=========================================="
