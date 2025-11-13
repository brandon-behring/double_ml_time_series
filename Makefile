# Makefile for Double ML Volume 2
#
# Builds professional PDFs with perfect equation rendering using asciidoctor-mathematical

.PHONY: all clean chapter1 chapter2 help

# Output directory
OUTPUT_DIR = output

# asciidoctor-pdf with mathematical rendering (SVG equations)
ASCIIDOCTOR_PDF = asciidoctor-pdf -r asciidoctor-mathematical -r asciidoctor-bibtex -a mathematical-format=svg

# All chapters
CHAPTERS = chapter_01 chapter_02

all: $(CHAPTERS)

chapter_01: $(OUTPUT_DIR)/chapter_01.pdf

chapter_02: $(OUTPUT_DIR)/chapter_02.pdf

$(OUTPUT_DIR)/chapter_01.pdf: chapters/chapter_01.adoc chapters/bibliography.bib
	@echo "Building Chapter 1 with perfect equation rendering..."
	@mkdir -p $(OUTPUT_DIR)
	$(ASCIIDOCTOR_PDF) -o $@ $<
	@ls -lh $@
	@echo "✓ Chapter 1 PDF created: $@"

$(OUTPUT_DIR)/chapter_02.pdf: chapters/02_orthogonality_dml.adoc chapters/bibliography.bib
	@echo "Building Chapter 2 with perfect equation rendering..."
	@mkdir -p $(OUTPUT_DIR)
	$(ASCIIDOCTOR_PDF) -o $@ $<
	@ls -lh $@
	@echo "✓ Chapter 2 PDF created: $@"

clean:
	@echo "Cleaning output directory..."
	rm -f $(OUTPUT_DIR)/*.pdf $(OUTPUT_DIR)/*.xml $(OUTPUT_DIR)/*.tex $(OUTPUT_DIR)/*.log $(OUTPUT_DIR)/*.aux
	@echo "✓ Cleaned"

help:
	@echo "Double ML Volume 2 - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         Build all chapters (default)"
	@echo "  chapter_01  Build Chapter 1 only"
	@echo "  chapter_02  Build Chapter 2 only"
	@echo "  clean       Remove all generated files"
	@echo "  help        Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - asciidoctor-pdf gem"
	@echo "  - asciidoctor-mathematical gem (for perfect equation rendering)"
	@echo "  - asciidoctor-bibtex gem (for citations)"
	@echo ""
	@echo "Example usage:"
	@echo "  make              # Build all chapters"
	@echo "  make chapter_01   # Build Chapter 1 only"
	@echo "  make clean        # Clean generated files"
