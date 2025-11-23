# Chapter 7: FRED Integration Plan Template
**Chapter**: 7 - FRED Macroeconomic Controls for Time Series DML
**Target Words**: ~5,000-6,000 words
**Estimated Effort**: 10-13 hours
**Dependencies**: Chapter 6 (DynamicDML + Panel Data) complete

---

## Objective

Integrate FRED (Federal Reserve Economic Data) API for macroeconomic controls in time series DML applications, with focus on insurance/annuity competitor pricing use cases.

---

## FRED API Module Design

### Module Structure

**Location**: `src/data/fred_fetcher.py`

**Core Components**:
```python
class FREDFetcher:
    """
    Fetch and manage FRED macroeconomic data for DML applications.

    Features:
    - API integration with error handling
    - Local caching for reliability
    - Data quality validation
    - Automatic updates
    """

    def __init__(self, api_key: str, cache_dir: str = "data/fred_cache/"):
        """Initialize with API key and cache directory."""

    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch single time series with caching."""

    def fetch_multiple(self, series_ids: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch multiple series aligned on dates."""

    def update_cache(self, series_ids: List[str]) -> None:
        """Update cached series with latest data."""

    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """Check data quality (missingness, outliers, breaks)."""
```

### API Integration Details

**Authentication**:
- API key stored in environment variable (`FRED_API_KEY`)
- Never commit API keys to git
- `.env.example` provided with placeholder

**Rate Limiting**:
- FRED API limit: 120 requests/minute
- Implement exponential backoff for failures
- Batch requests where possible

**Error Handling**:
- Network failures → retry with backoff
- Invalid series ID → clear error message
- API quota exceeded → use cached data
- Missing dates → forward fill with warning

---

## Macroeconomic Variable Selection

### Core Variables for Insurance/Annuity Pricing

**Interest Rates** (Primary Driver):
- **10-Year Treasury Constant Maturity** (`DGS10`)
  - Frequency: Daily (aggregate to monthly/quarterly)
  - Use: Discount rate proxy, competitor pricing driver
  - Transformation: Level + first difference

**Inflation**:
- **Consumer Price Index** (`CPIAUCSL`)
  - Frequency: Monthly
  - Use: Real vs. nominal pricing adjustments
  - Transformation: Level + YoY growth rate

**Economic Growth**:
- **GDP Growth Rate** (`A191RL1Q225SBEA`)
  - Frequency: Quarterly
  - Use: Economic cycle indicator
  - Transformation: Level + lag

**Labor Market**:
- **Unemployment Rate** (`UNRATE`)
  - Frequency: Monthly
  - Use: Economic health proxy
  - Transformation: Level + first difference

**Housing Market** (for annuity/life insurance):
- **Case-Shiller Home Price Index** (`CSUSHPISA`)
  - Frequency: Monthly
  - Use: Wealth effect indicator
  - Transformation: Level + YoY growth rate

### Additional Variables (Optional)

- Federal Funds Rate (`FEDFUNDS`)
- Corporate Bond Yields (`BAMLC0A4CBBB`)
- Equity Market Index (`SP500`)
- Consumer Sentiment (`UMCSENT`)

---

## Data Update Frequency Strategy

### Caching Strategy

**Initial Load**:
1. Fetch historical data on first run
2. Save to local cache (`data/fred_cache/`)
3. Store metadata (fetch date, series info)

**Update Strategy**:
- **Daily use**: Check cache age, update if >1 day old
- **Production**: Scheduled updates (e.g., weekly cron job)
- **Development**: Use cached data unless explicit update

### Cache Structure

```
data/fred_cache/
├── DGS10_2020-01-01_2025-11-21.parquet
├── CPIAUCSL_2020-01-01_2025-11-21.parquet
├── metadata.json
└── last_updated.txt
```

**Benefits**:
- Resilient to API outages
- Faster development iteration
- Reproducible research (cached versions)
- Reduced API quota usage

---

## Error Handling & Fallback Mechanisms

### Error Scenarios & Responses

**API Unavailable**:
```python
try:
    data = fred_client.get_series(series_id)
except ConnectionError:
    logger.warning("FRED API unavailable, using cached data")
    data = load_from_cache(series_id)
    if data is None:
        raise FREDDataUnavailableError("No cached data available")
```

**Missing Data Points**:
```python
if data.isna().sum() > 0:
    logger.warning(f"{data.isna().sum()} missing values, forward filling")
    data = data.fillna(method='ffill')
    validate_fill_quality(data)  # Warn if too much filling
```

**Series Discontinued**:
```python
if series_not_found:
    logger.error(f"Series {series_id} not found or discontinued")
    suggest_alternatives(series_id)  # Provide similar series
    raise SeriesNotFoundError(...)
```

### Fallback Data Sources

1. **Primary**: FRED API
2. **Fallback 1**: Local cache (most recent)
3. **Fallback 2**: Static snapshot (included with repo for examples)
4. **Fallback 3**: Clear error message with manual download instructions

---

## Integration with DML Pipeline

### Data Preparation Workflow

```python
# 1. Fetch FRED data
fred = FREDFetcher(api_key=os.getenv('FRED_API_KEY'))
macro_data = fred.fetch_multiple(
    series_ids=['DGS10', 'CPIAUCSL', 'UNRATE'],
    start_date='2020-01-01',
    end_date='2025-11-21'
)

# 2. Align with treatment/outcome data
aligned_data = align_macro_with_treatment(
    macro_data=macro_data,
    treatment_data=competitor_pricing,
    frequency='monthly'
)

# 3. Create control matrix for DML
X_macro = create_macro_controls(
    macro_data=aligned_data,
    include_lags=True,
    include_growth_rates=True
)

# 4. Run DML with macro controls
dml = LinearDML(
    model_y=RandomForestRegressor(n_jobs=48),
    model_t=RandomForestClassifier(n_jobs=48),
    discrete_treatment=True,
    cv=5
)
dml.fit(Y=outcome, T=treatment, X=X_macro)
```

### Time Alignment Considerations

**Frequency Mismatches**:
- Daily → Monthly: End-of-month values or monthly average
- Quarterly → Monthly: Forward fill or interpolation
- Align all series to treatment observation frequency

**Lag Structure**:
- Use lagged macro variables (t-1, t-2) to avoid reverse causality
- Test sensitivity to lag specification
- Document lag choices in Chapter 7

---

## Testing Strategy

### Unit Tests

**Location**: `test/data/test_fred_fetcher.py`

**Test Coverage**:
- API connection (mocked)
- Cache hit/miss logic
- Data validation
- Error handling for all scenarios
- Date alignment
- Missing data handling

### Integration Tests

**With Real FRED API** (optional, controlled):
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('FRED_API_KEY'), reason="No API key")
def test_fred_fetch_real():
    """Test actual FRED API fetch (slow, optional)."""
    fred = FREDFetcher(api_key=os.getenv('FRED_API_KEY'))
    data = fred.fetch_series('DGS10', '2024-01-01', '2024-12-31')
    assert len(data) > 200  # Daily data for 1 year
```

### Example Notebooks

**Location**: `notebooks/chapter_7_fred_examples.ipynb`

**Demonstrations**:
1. Basic FRED data fetch
2. Multi-series alignment
3. DML with macro controls
4. Sensitivity to lag structure
5. Cache usage for reproducibility

---

## Chapter 7 Content Outline

### Section 1: Introduction to Macroeconomic Controls (~800 words)
- Why macro variables matter for time series causal inference
- Confounding from economic cycles
- FRED as comprehensive data source

### Section 2: FRED API Integration (~1,200 words)
- API setup and authentication
- FREDFetcher module walkthrough
- Caching strategy for reliability
- Code Example 7.1: Basic FRED fetch

### Section 3: Variable Selection for Insurance Pricing (~1,500 words)
- Interest rate environment (primary)
- Inflation and real pricing
- Economic cycle indicators
- Housing market for wealth effects
- Code Example 7.2: Multi-series fetch and alignment

### Section 4: Time Alignment and Lag Structure (~1,000 words)
- Frequency conversion (daily → monthly)
- Lag specification to avoid reverse causality
- Testing sensitivity to lags
- Code Example 7.3: Create lagged macro controls

### Section 5: DML with FRED Controls (~1,500 words)
- Integration into DML pipeline
- Cross-validation with time series structure
- Interpreting coefficients with macro controls
- Code Example 7.4: Complete DML with FRED

### Section 6: Practical Considerations (~500 words)
- API reliability and fallbacks
- Data quality validation
- Update frequency recommendations
- Cost/benefit of macro controls

---

## Implementation Checklist

**Before Starting Chapter 7**:
- [ ] Chapter 6 (DynamicDML) complete and approved
- [ ] FRED API key obtained (free: https://fred.stlouisfed.org/docs/api/api_key.html)
- [ ] FREDFetcher module skeleton created
- [ ] Test data downloaded for examples

**During Implementation**:
- [ ] FREDFetcher module (src/data/fred_fetcher.py)
- [ ] Comprehensive unit tests (≥80% coverage)
- [ ] Cache management utilities
- [ ] Data validation functions
- [ ] Integration with DML pipeline
- [ ] Example notebook complete

**Before Chapter 7 Completion**:
- [ ] All code examples executable
- [ ] Tests passing (including API mocks)
- [ ] Chapter content ~5,000-6,000 words
- [ ] LaTeX compiles with zero errors
- [ ] Cache strategy documented
- [ ] Error handling tested

---

## Success Criteria

- ✅ FREDFetcher module operational with comprehensive tests
- ✅ All 5 core macro variables fetchable and cacheable
- ✅ Cache provides 100% resilience to API outages
- ✅ DML pipeline accepts FRED controls seamlessly
- ✅ Chapter 7 content complete (~5,000-6,000 words)
- ✅ All code examples executable and tested
- ✅ Integration tests pass (mocked and real)

---

## Time Estimate Breakdown

| Task | Effort |
|------|--------|
| FREDFetcher module implementation | 3-4 hours |
| Unit tests (comprehensive) | 2-3 hours |
| DML pipeline integration | 1-2 hours |
| Example notebook creation | 1-2 hours |
| Chapter 7 writing (~5,500 words) | 3-4 hours |
| Review and polish | 1 hour |

**Total**: 10-13 hours

---

## Notes & Considerations

**API Key Management**:
- Store in `.env` file (never commit)
- Provide `.env.example` with instructions
- Document setup in README

**Static Snapshot for Examples**:
- Include 2020-2024 snapshot in repo
- Examples work without API key
- Reproducible for readers

**Future Enhancements**:
- Additional macro variables (volatility, spreads)
- International data (OECD indicators)
- Real-time vs. revised data handling
- Forecast integration (FRED nowcasts)

---

**Template Status**: Ready for Chapter 7 implementation
**Next**: Create production deployment guide template
