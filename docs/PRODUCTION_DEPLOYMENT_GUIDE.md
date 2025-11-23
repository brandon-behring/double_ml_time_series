# Production Deployment Guide
## Synthetic → Real Data Swap Process for Chapter 8 Template

**Purpose**: Enable safe, systematic deployment of the Chapter 8 production template (Insurance/Annuity Competitor Pricing) with real proprietary data.

**Target User**: Data scientist/actuary deploying DML analysis in production environment

**Last Updated**: 2025-11-21

---

## Overview

The Chapter 8 template is designed as a **production-ready DML application** that can be deployed with minimal changes by swapping synthetic data for real proprietary data.

**Key Principle**: Configuration-driven design allows data swap without code changes.

---

## Prerequisites

Before deploying with real data:

- [ ] Chapter 8 template works successfully on synthetic data
- [ ] All validation checks pass on synthetic data
- [ ] Real data matches expected schema (see Schema Requirements below)
- [ ] Appropriate data access permissions obtained
- [ ] Development environment with test data available
- [ ] Production environment configured

---

## Architecture: Configuration-Driven Design

### File Structure

```
competitor_pricing_template/
├── config/
│   ├── data_sources.yaml          # Data source configuration
│   ├── model_parameters.yaml      # DML model settings
│   └── validation_rules.yaml      # Validation thresholds
├── data/
│   ├── synthetic/                 # Synthetic data (committed)
│   ├── real/ (gitignored)         # Real data (NEVER commit)
│   └── schemas/                   # Data validation schemas
├── src/
│   ├── data_loader.py             # Schema-validated data loading
│   ├── preprocessing.py           # Data preparation
│   ├── dml_pipeline.py            # Core DML implementation
│   └── validation.py              # Quality checks
├── tests/
│   ├── test_synthetic_data.py     # Tests on synthetic (always run)
│   ├── test_real_data.py          # Tests on real (run before deployment)
│   └── smoke_tests.py             # Quick validation tests
├── .env.example                   # Template for environment variables
├── .gitignore                     # CRITICAL: Excludes real data
└── README.md                      # Deployment instructions
```

---

## Step-by-Step Deployment Process

### Step 1: Schema Validation

**Purpose**: Ensure real data matches expected structure before any processing.

**Schema Requirements** (`data/schemas/competitor_pricing_schema.yaml`):

```yaml
treatment:
  name: "competitor_rate_change"
  type: "binary"  # 0 = no change, 1 = rate increase
  required: true
  validation:
    - no_missing_values
    - binary_only

outcome:
  name: "pricing_response"
  type: "float"
  required: true
  validation:
    - no_missing_values
    - positive_values
    - outlier_check: [p01, p99]  # Flag if outside 1st-99th percentile

controls:
  macroeconomic:
    - name: "interest_rate_10y"
      type: "float"
      source: "FRED:DGS10"
      required: true
    - name: "inflation_rate"
      type: "float"
      source: "FRED:CPIAUCSL"
      required: true
    - name: "unemployment_rate"
      type: "float"
      source: "FRED:UNRATE"
      required: true

  company_specific:
    - name: "market_share"
      type: "float"
      range: [0.0, 1.0]
    - name: "product_type"
      type: "categorical"
      categories: ["annuity", "life", "disability"]

time:
  name: "observation_date"
  type: "datetime"
  frequency: "monthly"  # or "quarterly"
  required: true
```

**Validation Script** (`src/validate_schema.py`):

```python
from pydantic import BaseModel, validator
import pandas as pd

class CompetitorPricingData(BaseModel):
    """Schema validation for real data."""

    class Config:
        arbitrary_types_allowed = True

    @validator('treatment')
    def validate_treatment(cls, v):
        if v.isna().any():
            raise ValueError("Treatment has missing values")
        if not set(v.unique()).issubset({0, 1}):
            raise ValueError("Treatment must be binary (0/1)")
        return v

    @validator('outcome')
    def validate_outcome(cls, v):
        if v.isna().any():
            raise ValueError("Outcome has missing values")
        if (v < 0).any():
            raise ValueError("Outcome has negative values")
        return v

def validate_real_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate real data against schema.

    Returns:
        ValidationResult with pass/fail and detailed errors
    """
    try:
        schema = CompetitorPricingData(**df.to_dict('list'))
        return ValidationResult(passed=True, errors=[])
    except ValidationError as e:
        return ValidationResult(passed=False, errors=e.errors())
```

**Run validation**:
```bash
python src/validate_schema.py --data data/real/competitor_pricing.csv
```

**Expected output**:
```
✅ Schema validation PASSED
  - Treatment: 1,245 observations, binary (0/1)
  - Outcome: 1,245 observations, range [$1,200 - $15,000]
  - Controls: 8 variables, 0 missing values
  - Time: 2020-01-01 to 2024-12-31 (60 months)
```

---

### Step 2: Configuration File Setup

**Data Source Configuration** (`config/data_sources.yaml`):

```yaml
# DEVELOPMENT (uses synthetic data)
development:
  data_path: "data/synthetic/competitor_pricing.csv"
  fred_api_key: null  # Uses cached FRED data
  use_cache: true

# PRODUCTION (uses real data)
production:
  data_path: "data/real/competitor_pricing.csv"  # NEVER commit this file
  fred_api_key: "${FRED_API_KEY}"  # From environment variable
  use_cache: false  # Fetch latest macro data

# TEST (uses subset of real data for smoke tests)
test:
  data_path: "data/real/competitor_pricing_test_subset.csv"
  fred_api_key: null
  use_cache: true
  sample_size: 100  # Small subset for quick tests
```

**Switch environments**:
```python
# In code
config = load_config(environment=os.getenv('DEPLOY_ENV', 'development'))
data = load_data(config['data_path'])
```

**Set environment**:
```bash
# Development (default)
python run_analysis.py

# Production
DEPLOY_ENV=production python run_analysis.py
```

---

### Step 3: Environment Variables & Secrets

**Create `.env` file** (NEVER commit):

```bash
# .env (gitignored, for local development/production)
DEPLOY_ENV=production
FRED_API_KEY=your_actual_api_key_here
DATA_ENCRYPTION_KEY=your_encryption_key_here

# Database credentials (if using DB instead of CSV)
DB_HOST=your_database_host
DB_PORT=5432
DB_NAME=competitor_pricing_db
DB_USER=your_username
DB_PASSWORD=your_password
```

**Provide `.env.example`** (committed as template):

```bash
# .env.example (template for users)
DEPLOY_ENV=development
FRED_API_KEY=get_your_key_from_https://fred.stlouisfed.org/docs/api/api_key.html
DATA_ENCRYPTION_KEY=generate_with_python_secrets_module

# Database credentials (if needed)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=competitor_pricing_db
DB_USER=your_username
DB_PASSWORD=your_secure_password
```

**Load in code**:
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

fred_api_key = os.getenv('FRED_API_KEY')
if fred_api_key is None:
    raise EnvironmentError("FRED_API_KEY not set. Copy .env.example to .env and configure.")
```

---

### Step 4: Git Safety Checklist

**CRITICAL**: Never commit real data or secrets to git.

**Required `.gitignore` entries**:

```
# Real data (NEVER commit)
data/real/
*.csv  # Be cautious - only ignore data CSVs, not code
*.parquet
*.feather
*.xlsx

# Environment variables (NEVER commit)
.env
.env.local
.env.production

# Credentials (NEVER commit)
credentials.json
secrets.yaml
*.key
*.pem

# Logs that may contain sensitive info
logs/
*.log

# Python cache
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/
```

**Pre-commit hook** (recommended):

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Check for accidentally staged real data
if git diff --cached --name-only | grep -E "data/real/|\.env$"; then
    echo "❌ ERROR: Attempting to commit real data or .env file!"
    echo "Files:"
    git diff --cached --name-only | grep -E "data/real/|\.env$"
    exit 1
fi

# Check for API keys in code
if git diff --cached | grep -iE "api[_-]?key.*=.*['\"][a-zA-Z0-9]{20,}"; then
    echo "⚠️  WARNING: Possible API key found in code!"
    echo "Use environment variables instead."
    exit 1
fi

echo "✅ Pre-commit checks passed"
```

**Make executable**:
```bash
chmod +x .git/hooks/pre-commit
```

---

### Step 5: Data Quality Checks

**Run comprehensive checks before DML**:

```python
from src.validation import DataQualityChecker

checker = DataQualityChecker(config=config)

# 1. Schema validation
schema_result = checker.validate_schema(df)
if not schema_result.passed:
    raise DataQualityError(f"Schema validation failed: {schema_result.errors}")

# 2. Missing data check
missing_result = checker.check_missing_data(df, threshold=0.05)
if not missing_result.passed:
    logger.warning(f"Missing data exceeds 5%: {missing_result.summary}")

# 3. Outlier detection
outlier_result = checker.detect_outliers(df, method='iqr', threshold=3.0)
if outlier_result.n_outliers > 0:
    logger.warning(f"Found {outlier_result.n_outliers} outliers")
    # Option: Flag or remove outliers

# 4. Treatment balance check
balance_result = checker.check_treatment_balance(df)
if balance_result.imbalance_ratio > 0.3:  # >30% imbalance
    logger.warning(f"Treatment imbalance: {balance_result.summary}")
    # May need propensity score weighting

# 5. Temporal consistency
temporal_result = checker.check_temporal_gaps(df, expected_frequency='monthly')
if temporal_result.has_gaps:
    logger.error(f"Temporal gaps found: {temporal_result.gaps}")
    raise DataQualityError("Cannot proceed with temporal gaps")
```

---

### Step 6: Testing Strategy for Real Data

**Three-tier testing approach**:

#### Tier 1: Smoke Tests (Fast, run always)

```python
# tests/smoke_tests.py
def test_real_data_loads():
    """Verify real data file exists and loads."""
    assert os.path.exists(config['data_path'])
    df = load_data(config['data_path'])
    assert len(df) > 0

def test_schema_validation_passes():
    """Verify real data passes schema validation."""
    df = load_data(config['data_path'])
    result = validate_schema(df)
    assert result.passed, f"Schema errors: {result.errors}"

def test_dml_runs_without_error():
    """Verify DML pipeline executes (doesn't check results)."""
    df = load_data(config['data_path'])
    result = run_dml_pipeline(df, quick_mode=True)  # Minimal CV folds
    assert result is not None
```

**Run**: `pytest tests/smoke_tests.py` (~1 minute)

#### Tier 2: Validation Tests (Moderate, run before deployment)

```python
# tests/test_real_data.py
def test_validation_suite_passes():
    """Run full 7-method validation on real data."""
    df = load_data(config['data_path'])
    validation = ValidationSuite(df)
    results = validation.run_all()

    # Check all methods within expected ranges
    assert results['bias_validation'].status in ['PASS', 'WARNING']
    assert results['coverage_test'].coverage >= 0.90
    assert results['sensitivity_analysis'].stable == True

def test_results_make_business_sense():
    """Sanity check on estimated treatment effects."""
    df = load_data(config['data_path'])
    result = run_dml_pipeline(df)

    # Business logic checks
    assert -1000 <= result.ate <= 5000, "ATE outside plausible range"
    assert result.ci_width < 3000, "CI too wide for actionable insights"
```

**Run**: `pytest tests/test_real_data.py` (~10-30 minutes)

#### Tier 3: Full Validation (Slow, run before major releases)

```python
# tests/test_full_validation.py
@pytest.mark.slow
def test_bootstrap_stability():
    """Verify results stable across bootstrap replications."""
    df = load_data(config['data_path'])
    results = run_bootstrap_validation(df, n_bootstrap=1000)

    assert results.std_error < 200, "Too much bootstrap variability"
    assert results.bias < 50, "Significant bootstrap bias"
```

**Run**: `pytest tests/test_full_validation.py -m slow` (~1-2 hours)

---

### Step 7: Deployment Checklist

**Pre-Deployment**:
- [ ] Real data validated against schema (Step 1)
- [ ] Configuration files updated for production (Step 2)
- [ ] Environment variables configured in `.env` (Step 3)
- [ ] Git safety checks in place (Step 4)
- [ ] Data quality checks passed (Step 5)
- [ ] Smoke tests passed (Step 6, Tier 1)
- [ ] Validation tests passed (Step 6, Tier 2)

**Deployment**:
- [ ] Set `DEPLOY_ENV=production`
- [ ] Run full DML pipeline: `python run_analysis.py`
- [ ] Monitor logs for warnings/errors
- [ ] Review results for business sense
- [ ] Generate validation report
- [ ] Save results with timestamp

**Post-Deployment**:
- [ ] Archive results with metadata
- [ ] Document any anomalies or surprises
- [ ] Update monitoring dashboards (if applicable)
- [ ] Schedule next run (if recurring analysis)

---

## Monitoring & Maintenance

### Logging Best Practices

**Structured logging** (`src/logging_config.py`):

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('competitor_pricing')

# Log key events
logger.info("Starting DML analysis", extra={
    'environment': os.getenv('DEPLOY_ENV'),
    'data_rows': len(df),
    'date_range': f"{df['date'].min()} to {df['date'].max()}"
})

logger.warning("Treatment imbalance detected", extra={
    'treated_pct': (df['treatment'] == 1).mean(),
    'threshold': 0.3
})
```

### Results Archiving

**Save results with metadata**:

```python
from datetime import datetime
import json

# Results directory structure
results_dir = f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
os.makedirs(results_dir, exist_ok=True)

# Save DML results
result.save(f"{results_dir}/dml_results.json")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'environment': os.getenv('DEPLOY_ENV'),
    'data_path': config['data_path'],
    'n_observations': len(df),
    'date_range': {
        'start': str(df['date'].min()),
        'end': str(df['date'].max())
    },
    'validation_status': 'PASS',  # From validation suite
    'ate_estimate': float(result.ate),
    'ci_lower': float(result.ci_lower),
    'ci_upper': float(result.ci_upper)
}

with open(f"{results_dir}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## Troubleshooting Guide

### Issue: Schema Validation Fails

**Symptom**: `ValueError: Treatment must be binary (0/1)`

**Solutions**:
1. Check treatment encoding in real data
2. Verify column names match schema exactly (case-sensitive)
3. Inspect for missing values: `df['treatment'].isna().sum()`
4. Check data types: `df.dtypes`

### Issue: FRED API Quota Exceeded

**Symptom**: `FREDAPIError: Rate limit exceeded`

**Solutions**:
1. Use cached FRED data: Set `use_cache: true` in config
2. Reduce update frequency
3. Consider FRED API premium tier (higher limits)

### Issue: Results Don't Make Business Sense

**Symptom**: ATE estimate is implausibly large or negative

**Solutions**:
1. Check treatment definition (is it coded correctly?)
2. Verify outcome variable units (dollars vs. cents vs. percentages)
3. Review confounders - are critical controls missing?
4. Run sensitivity analysis to understand drivers
5. Compare to naive OLS for sanity check

### Issue: Wide Confidence Intervals

**Symptom**: CI too wide for actionable insights

**Solutions**:
1. Increase sample size if possible
2. Reduce outcome variance (better controls, subgroup analysis)
3. Improve treatment-control overlap (check propensity scores)
4. Consider alternative estimators (AIPW may be more efficient)

---

## Security Considerations

**Data Encryption**:
- Encrypt real data at rest: `gpg -c data/real/competitor_pricing.csv`
- Decrypt in memory only: `gpg -d data/real/competitor_pricing.csv.gpg | python process_data.py`

**Access Control**:
- Limit production environment access (role-based)
- Audit logs for data access
- Separate dev/test/prod environments

**Compliance**:
- GDPR: Ensure no PII in analysis data
- SOC 2: Document all data handling procedures
- Internal policies: Follow company data governance

---

## Appendix: Example Deployment Session

```bash
# 1. Clone repository
git clone https://github.com/your-org/double_ml_time_series.git
cd double_ml_time_series/competitor_pricing_template

# 2. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure for production
cp .env.example .env
# Edit .env with real credentials
nano .env

# 4. Copy real data (NEVER commit)
cp /secure/location/competitor_pricing.csv data/real/

# 5. Validate schema
python src/validate_schema.py --data data/real/competitor_pricing.csv

# 6. Run smoke tests
pytest tests/smoke_tests.py

# 7. Run full validation tests
pytest tests/test_real_data.py

# 8. Deploy to production
DEPLOY_ENV=production python run_analysis.py

# 9. Review results
cat results/$(ls -t results/ | head -1)/metadata.json
```

---

**Guide Status**: Ready for Chapter 8 implementation
**Last Updated**: 2025-11-21
**Version**: 1.0
