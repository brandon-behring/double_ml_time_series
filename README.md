# Volume 2: Double Machine Learning for Time Series

**Production-grade Double Machine Learning book for time series causal inference**

## Overview

This project develops a comprehensive reference book on Double Machine Learning (DML) methodology with focus on time series applications, particularly insurance/annuity competitor pricing with macroeconomic controls.

**Status**: Phase 1A Complete ✅ - Chapters 1-2 fully written (13,213 words, 43 pages)

## Project Structure

```
double_ml_time_series/
├── chapters/           # LaTeX book chapters (theory + code examples)
│   ├── chapter_01.tex          # Potential Outcomes + FWL
│   ├── chapter_02.tex          # Neyman Orthogonality + DML
│   ├── chapter_template.tex    # Reference guide
│   ├── bibliography.bib        # BibTeX references
│   └── archive_asciidoc/       # Original AsciiDoc (preserved)
├── main.tex            # Main LaTeX document (amsbook class)
├── Makefile            # Professional build system
├── notebooks/          # Jupyter notebooks (validation + applications)
│   ├── validation/     # 7-method validation suite
│   └── applications/   # Work examples
├── src/                # Python modules
│   ├── validation/     # Synthetic DGP generator
│   ├── data/           # FRED API fetcher
│   └── dml/            # DML implementations
├── scripts/            # Conversion and build tools
│   ├── clean_pandoc_output.py    # Pandoc cleanup automation
│   └── convert_code_blocks.py    # Code block batch converter
├── tests/              # Unit tests (pytest)
└── docs/               # State tracking and planning
```

## Phases

### Phase 1: Foundation (40-50 hours)
- **1A**: Chapters 1-2 (Potential Outcomes, FWL, Neyman Orthogonality, DML Algorithm)
- **1B**: Chapter 3 (7-method validation suite) ⚠️ CRITICAL GATE
- **1C**: Chapter 4 (Cross-sectional application)

### Phase 2: Time Series Extension (40-50 hours)
- **2A**: Chapters 5-7 (Dynamic treatment, DynamicDML, FRED integration)
- **2B**: Chapter 8 (Insurance competitor pricing) ⚠️ PRODUCTION TEMPLATE

### Phase 3: Advanced Topics (30-40 hours)
- **3A**: Chapters 9-10 (Heterogeneity, production pipeline)
- **3B**: Appendix (Julia DML.jl roadmap)

## Setup

### Prerequisites
- Python 3.11+
- LaTeX distribution (TeX Live or MiKTeX)
  - Required packages: `amsbook`, `minted`, `hyperref`, `booktabs`
- Pygments (for minted code highlighting): `pip install Pygments`
- 64-core system recommended (parallelization optimized for Threadripper)

### Installation

**Step 1: Install Python environment**

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_environment.py

# Install pre-commit hooks
pre-commit install
```

**Note**: Works with Python 3.11+ (tested on 3.13.7). R integration (rpy2) commented out by default - requires R installation for cross-implementation validation.

### Verify Installation

After installation, verify all dependencies are working:

```bash
python scripts/verify_environment.py
```

Expected output: ✅ All 21 packages verified successfully

If any packages fail to import, ensure venv is activated and run `pip install -r requirements.txt` again.

### Document Compilation

The book is written in native LaTeX using the `amsbook` document class for professional mathematical typesetting.

```bash
# Build complete book (all chapters)
make

# Build and open PDF
make view

# Build individual chapters (for testing)
make chapter1
make chapter2

# Clean auxiliary files (keep PDF)
make clean

# Remove all generated files including PDF
make distclean

# Install Python dependencies (Pygments for minted)
make install-deps

# Check LaTeX log for errors/warnings
make check-errors
```

**LaTeX Compilation Process**:
1. `pdflatex -shell-escape` (first pass)
2. `bibtex` (process bibliography)
3. `pdflatex` (second pass, resolve citations)
4. `pdflatex` (third pass, resolve cross-references)

**Output**: `main.pdf` (43 pages, ~350KB)

**Why LaTeX?**
- ✅ Native equation rendering (searchable, scalable)
- ✅ 88% smaller PDFs vs AsciiDoc SVG images
- ✅ Professional theorem environments
- ✅ Cross-references with hyperlinks
- ✅ Minted code syntax highlighting
- ✅ Zero compilation errors/warnings

**Manual compilation** (if not using Make):
```bash
pdflatex -shell-escape -interaction=nonstopmode main.tex
bibtex main
pdflatex -shell-escape -interaction=nonstopmode main.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex
```

### Writing New Chapters

Use the template file as a reference:

```bash
# Copy template
cp chapters/chapter_template.tex chapters/chapter_03.tex

# Edit chapter
# ... add content following template examples ...

# Add to main.tex
echo "\\include{chapters/chapter_03}" >> main.tex

# Build
make
```

**Template includes**:
- Theorem environments (definition, theorem, lemma, proposition, corollary)
- Example and exercise environments
- Python code blocks with `minted`
- Proof environments with QED symbols
- Common math commands (`\E`, `\Var`, `\Cov`, `\Prob`)
- Cross-reference examples

### Configuration

**Pre-commit hooks**: Black (100-char), mypy (type hints), large commit warning
**Testing**: pytest + coverage (target: 80%+ for modules)
**Parallelization**: n_jobs=48 (leave 16 cores for system)

## Validation

7-method comprehensive validation suite:
1. Published results replication (Chernozhukov 2018, Facure 2022)
2. Synthetic Monte Carlo (1000 runs, 95% coverage)
3. Cross-implementation (Manual vs EconML vs R DoubleML)
4. Diagnostics (first-stage R², residuals, sensitivity)
5. Real-world known outcomes
6. Public dataset benchmarks
7. Synthetic DGP generator (parametric, unit tested)

## Development Workflow

**Branches**:
- `main`: Stable, reviewed, approved
- `dev`: Work in progress (active development)

**Workflow**:
1. Work on `dev` branch
2. Commit frequently (after each section)
3. Update state files (CURRENT_WORK.md after every subsection)
4. Milestone review → merge to `main` → tag

**State Files**:
- `docs/plans/active/DOUBLE_ML_VOL2_2025-11-13.md` - Master plan
- `docs/plans/active/CHAPTER_STATUS.md` - Granular progress
- `docs/CURRENT_WORK.md` - Current task (30-sec resume)
- `docs/plans/active/DEFERRED_IDEAS.md` - Future enhancements

## Integration

- **ProjectRegistry**: Registered in archimedes_lever
- **Backups**: S3 via archimedes_lever system
- **Morning Reports**: Progress tracked automatically

## Hardware Optimization

**64-core Threadripper configuration**:
```python
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent nested parallelism
os.environ['MKL_NUM_THREADS'] = '1'

N_JOBS = 48  # Leave 16 cores for system
```

- Monte Carlo: Parallelized (1000 runs across 48 cores)
- Validation suite: Batched parallel (3-4 notebooks concurrent)
- EconML: All models use n_jobs=48

## Contact

Brandon Behring

## License

Personal research project - not for distribution
