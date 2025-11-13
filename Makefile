# Makefile for Double ML Volume 2 (Native LaTeX)
#
# Build professional PDFs with native LaTeX equation rendering

.PHONY: all clean chapter1 chapter2 help view install-deps

# Main targets
MAIN = main
PDF = $(MAIN).pdf

# LaTeX compilation commands
LATEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
BIBTEX = bibtex

# Default target: build complete book
all: $(PDF)

# Build complete book (all chapters)
$(PDF): $(MAIN).tex chapters/chapter_01.tex chapters/chapter_02.tex chapters/bibliography.bib
	@echo "=========================================="
	@echo "Building complete book..."
	@echo "=========================================="
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@echo ""
	@echo "✓ Build complete: $(PDF)"
	@ls -lh $(PDF)
	@echo ""

# Build individual chapter (for testing)
chapter1: chapters/chapter_01.tex
	@echo "Building Chapter 1 (standalone)..."
	@mkdir -p output
	$(LATEX) -output-directory=output chapters/chapter_01.tex
	@echo "✓ Chapter 1 built: output/chapter_01.pdf"

chapter2: chapters/chapter_02.tex
	@echo "Building Chapter 2 (standalone)..."
	@mkdir -p output
	$(LATEX) -output-directory=output chapters/chapter_02.tex
	@echo "✓ Chapter 2 built: output/chapter_02.pdf"

# View PDF in default viewer
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Install required dependencies
install-deps:
	@echo "Installing Python dependencies for minted..."
	pip install Pygments
	@echo ""
	@echo "✓ Dependencies installed"
	@echo ""
	@echo "LaTeX packages required (install via TeX distribution):"
	@echo "  - amsbook (usually included)"
	@echo "  - minted"
	@echo "  - hyperref"
	@echo "  - booktabs"
	@echo ""

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).bbl $(MAIN).blg
	rm -f $(MAIN).synctex.gz $(MAIN).fdb_latexmk $(MAIN).fls
	rm -f chapters/*.aux
	rm -rf _minted-$(MAIN)/
	@echo "✓ Cleaned"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "✓ All generated files removed"

# Check for LaTeX errors in log
check-errors:
	@if [ -f $(MAIN).log ]; then \
		echo "Checking for errors in $(MAIN).log..."; \
		if grep -E "^!" $(MAIN).log; then \
			echo ""; \
			echo "❌ Errors found in LaTeX compilation"; \
			exit 1; \
		else \
			echo "✓ No errors found"; \
		fi; \
		echo ""; \
		echo "Checking for warnings..."; \
		if grep -i "warning" $(MAIN).log; then \
			echo ""; \
			echo "⚠️  Warnings found (review log)"; \
		else \
			echo "✓ No warnings found"; \
		fi; \
	else \
		echo "No log file found. Run 'make' first."; \
	fi

# Help message
help:
	@echo "Double ML Volume 2 - LaTeX Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all            Build complete book (default)"
	@echo "  chapter1       Build Chapter 1 only (for testing)"
	@echo "  chapter2       Build Chapter 2 only (for testing)"
	@echo "  view           Build and open PDF in viewer"
	@echo "  install-deps   Install Python dependencies (Pygments)"
	@echo "  clean          Remove auxiliary files (keep PDF)"
	@echo "  distclean      Remove all generated files including PDF"
	@echo "  check-errors   Check LaTeX log for errors/warnings"
	@echo "  help           Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (TeX Live or MiKTeX)"
	@echo "  - bibtex"
	@echo "  - Python + Pygments (for minted code highlighting)"
	@echo ""
	@echo "Example usage:"
	@echo "  make               # Build complete book"
	@echo "  make chapter1      # Test Chapter 1 compilation"
	@echo "  make view          # Build and open PDF"
	@echo "  make clean         # Clean auxiliary files"
	@echo ""
	@echo "Compilation process:"
	@echo "  1. pdflatex (first pass)"
	@echo "  2. bibtex (process bibliography)"
	@echo "  3. pdflatex (second pass, resolve citations)"
	@echo "  4. pdflatex (third pass, resolve cross-references)"
	@echo ""
