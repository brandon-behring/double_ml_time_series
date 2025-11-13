# Volume 2: Double Machine Learning for Time Series

**Production-grade Double Machine Learning book for time series causal inference**

## Overview

This project develops a comprehensive reference book on Double Machine Learning (DML) methodology with focus on time series applications, particularly insurance/annuity competitor pricing with macroeconomic controls.

**Status**: Phase 1A - Infrastructure Complete, Beginning Chapter 1

## Project Structure

```
double_ml_time_series/
├── chapters/           # AsciiDoc book chapters (theory + embedded examples)
├── notebooks/          # Jupyter notebooks (validation + applications)
│   ├── validation/     # 7-method validation suite
│   └── applications/   # Work examples
├── src/                # Python modules
│   ├── validation/     # Synthetic DGP generator
│   ├── data/           # FRED API fetcher
│   └── dml/            # DML implementations
├── tests/              # Unit tests (pytest)
├── code/               # Standalone code examples
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
- Ruby gems: `asciidoctor-pdf`, `asciidoctor-mathematical`, `asciidoctor-bibtex`
- 64-core system recommended (parallelization optimized for Threadripper)

### Installation

**Step 1: Install Ruby gems for PDF generation**

```bash
gem install asciidoctor-pdf asciidoctor-mathematical asciidoctor-bibtex
```

**Step 2: Install Python environment**

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

The book uses AsciiDoc with professional LaTeX equation rendering via `asciidoctor-mathematical`:

```bash
# Build all chapters with perfect equation rendering
make all

# Build individual chapters
make chapter_01
make chapter_02

# Clean generated files
make clean
```

**How it works:**
- `asciidoctor-mathematical` converts LaTeX equations to high-quality SVG images
- Images are embedded in PDF for perfect rendering (no font issues, no Unicode problems)
- Larger PDFs (~4MB per chapter) but equations look professional

**Manual compilation** (if not using Make):
```bash
asciidoctor-pdf \
  -r asciidoctor-mathematical \
  -r asciidoctor-bibtex \
  -a mathematical-format=svg \
  -o output/chapter_01.pdf \
  chapters/chapter_01.adoc
```

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
