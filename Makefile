# Makefile for Double ML Volume 2 (Native LaTeX)
#
# Build professional PDFs with native LaTeX equation rendering

.PHONY: all clean chapter1 chapter2 help view install-deps test test-unit test-integration test-all test-watch test-coverage docs examples check-errors

# Main targets
MAIN = main
PDF = $(MAIN).pdf

# LaTeX compilation commands
LATEX = lualatex -shell-escape -interaction=nonstopmode -file-line-error
BIBER = biber
PYTHON = venv/bin/python

# Default target: build complete book
all: $(PDF)

# Build complete book (all chapters)
$(PDF): $(MAIN).tex chapters/chapter_01.tex chapters/chapter_02.tex chapters/bibliography.bib
	@echo "=========================================="
	@echo "Building complete book..."
	@echo "=========================================="
	$(LATEX) $(MAIN)
	$(BIBER) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@echo ""
	$(MAKE) check-errors
	@echo "Build complete: $(PDF)"
	@ls -lh $(PDF)
	@echo ""

# Build individual chapter (for testing)
chapter1: chapters/chapter_01.tex
	@echo "Building Chapter 1 (standalone)..."
	@mkdir -p output
	$(LATEX) -output-directory=output chapters/chapter_01.tex
	@echo "Chapter 1 built: output/chapter_01.pdf"

chapter2: chapters/chapter_02.tex
	@echo "Building Chapter 2 (standalone)..."
	@mkdir -p output
	$(LATEX) -output-directory=output chapters/chapter_02.tex
	@echo "Chapter 2 built: output/chapter_02.pdf"

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
	$(PYTHON) -m pip install Pygments
	@echo ""
	@echo "Dependencies installed"
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
	@echo "Cleaned"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "All generated files removed"

# Check for LaTeX errors in log
check-errors:
	@if [ -f $(MAIN).log ]; then \
		echo "Checking for errors in $(MAIN).log..."; \
		if grep -E "^!" $(MAIN).log; then \
			echo ""; \
			echo "Errors found in LaTeX compilation"; \
			exit 1; \
		else \
			echo "No fatal TeX errors found"; \
		fi; \
		echo ""; \
		echo "Reporting overfull/underfull boxes (non-blocking in this milestone)..."; \
		if grep -E "Overfull|Underfull" $(MAIN).log; then \
			echo ""; \
			echo "Box warnings found; review log."; \
		else \
			echo "No overfull/underfull boxes found"; \
		fi; \
	else \
		echo "No log file found. Run 'make' first."; \
		exit 1; \
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
	@echo "  - lualatex (TeX Live or MiKTeX)"
	@echo "  - biber"
	@echo "  - Python + Pygments (for minted code highlighting)"
	@echo ""
	@echo "Example usage:"
	@echo "  make               # Build complete book"
	@echo "  make chapter1      # Test Chapter 1 compilation"
	@echo "  make view          # Build and open PDF"
	@echo "  make clean         # Clean auxiliary files"
	@echo ""
	@echo "Compilation process:"
	@echo "  1. lualatex (first pass)"
	@echo "  2. biber (process bibliography)"
	@echo "  3. lualatex (second pass, resolve citations)"
	@echo "  4. lualatex (third pass, resolve cross-references)"
	@echo ""
	@echo "Test targets:"
	@echo "  test-unit        Run unit tests only (~3s, pre-commit level)"
	@echo "  test-integration Run integration tests (~10min, excludes slow)"
	@echo "  test-all         Run full test suite (~30min, includes slow)"
	@echo "  test-watch       TDD watch mode (auto-run unit tests on save)"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""

# ============================================================================
# Test Targets (TDD Workflow)
# ============================================================================

# Alias for tier1 tests (default test target)
test: test-unit

# Fast tier1 tests
test-unit:
	@echo "Running tier1 tests..."
	$(PYTHON) -m pytest -m tier1 --no-cov -q
	@echo "Tier1 tests complete"

# Tier1 + tier2 tests
test-integration:
	@echo "Running tier1+tier2 tests..."
	$(PYTHON) -m pytest -m "tier1 or tier2" --no-cov -q
	@echo "Tier1+tier2 tests complete"

# Full test suite
test-all:
	@echo "Running full test suite..."
	$(PYTHON) -m pytest
	@echo "Full test suite complete"

# TDD watch mode - auto-run unit tests on file save
# Usage: make test-watch (then edit code, tests auto-run)
# Press Ctrl+C to exit
test-watch:
	@echo "Starting TDD watch mode..."
	@echo "  - Tests auto-run when files change"
	@echo "  - Press Ctrl+C to exit"
	@echo ""
	venv/bin/ptw test -- -m tier1 --no-cov -q

# Coverage report (generates HTML report in htmlcov/)
test-coverage:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated: htmlcov/index.html"

docs:
	$(PYTHON) -m sphinx -b html -W --keep-going docs/sphinx docs/sphinx/_build/html

examples:
	@for f in examples/*.py; do \
		echo "Running $$f"; \
		$(PYTHON) "$$f"; \
	done
