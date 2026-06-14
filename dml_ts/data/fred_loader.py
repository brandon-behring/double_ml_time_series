"""
FRED Data Loader for Macroeconomic Controls in DML.

Provides standardized access to Federal Reserve Economic Data (FRED) for
use as control variables in time series DML applications.

Key Features:
1. Automated retrieval of macro indicators
2. Time alignment and frequency conversion
3. Missing data handling with interpolation
4. Caching to avoid repeated API calls

Common Use Cases:
- GDP growth as business cycle control
- CPI/inflation as price level control
- Interest rates for monetary policy studies
- Unemployment for labor market analysis

References:
- FRED API: https://fred.stlouisfed.org/docs/api/fred/

Usage:
    >>> from dml_ts.data import FREDLoader
    >>> # Requires live FRED API key + network; skipped in the doctest gate
    >>> loader = FREDLoader(api_key="your_api_key")  # doctest: +SKIP
    >>> macro_data = loader.get_macro_controls(  # doctest: +SKIP
    ...     start_date="2010-01-01",
    ...     end_date="2020-12-31",
    ...     frequency="M"
    ... )
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# Try to import fredapi, fall back to mock if not available
try:
    from fredapi import Fred

    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False
    Fred = None  # noqa: N816


# Type aliases
FrequencyType = Literal["D", "W", "M", "Q", "A"]


@dataclass
class FREDSeries:
    """Container for a FRED series with metadata.

    Attributes:
        series_id: FRED series identifier (e.g., "GDP", "CPIAUCSL")
        name: Human-readable name
        description: Detailed description
        frequency: Data frequency (D/W/M/Q/A)
        units: Units of measurement
        data: Time series data as pandas Series
    """

    series_id: str
    name: str
    description: str
    frequency: str
    units: str
    data: pd.Series


@dataclass
class MacroControlsResult:
    """Result container for macro control variables.

    Attributes:
        data: DataFrame with aligned macro indicators
        metadata: Dict with series metadata
        start_date: Actual start date of data
        end_date: Actual end date of data
        frequency: Data frequency
        missing_pct: Dict of missing data percentages by series
    """

    data: pd.DataFrame
    metadata: dict[str, dict[str, str]]
    start_date: str
    end_date: str
    frequency: str
    missing_pct: dict[str, float]


# Standard FRED series for DML macro controls
STANDARD_MACRO_SERIES: dict[str, dict[str, str]] = {
    # Output measures
    "GDP": {
        "name": "Gross Domestic Product",
        "description": "Nominal GDP in billions of dollars (quarterly)",
        "default_transform": "pct_change",
    },
    "GDPC1": {
        "name": "Real GDP",
        "description": "Real GDP in billions of chained 2017 dollars (quarterly)",
        "default_transform": "pct_change",
    },
    # Price measures
    "CPIAUCSL": {
        "name": "Consumer Price Index",
        "description": "CPI for All Urban Consumers (monthly)",
        "default_transform": "pct_change_yoy",
    },
    "PCEPI": {
        "name": "PCE Price Index",
        "description": "Personal Consumption Expenditures Price Index (monthly)",
        "default_transform": "pct_change_yoy",
    },
    # Labor market
    "UNRATE": {
        "name": "Unemployment Rate",
        "description": "Civilian Unemployment Rate (monthly, %)",
        "default_transform": "level",
    },
    "PAYEMS": {
        "name": "Total Nonfarm Payrolls",
        "description": "Total Nonfarm Employment in thousands (monthly)",
        "default_transform": "pct_change",
    },
    # Interest rates
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "description": "Effective Federal Funds Rate (monthly, %)",
        "default_transform": "level",
    },
    "GS10": {
        "name": "10-Year Treasury Rate",
        "description": "10-Year Treasury Constant Maturity Rate (monthly, %)",
        "default_transform": "level",
    },
    "TB3MS": {
        "name": "3-Month Treasury Rate",
        "description": "3-Month Treasury Bill Rate (monthly, %)",
        "default_transform": "level",
    },
    # Financial conditions
    "SP500": {
        "name": "S&P 500 Index",
        "description": "S&P 500 Stock Price Index (daily)",
        "default_transform": "pct_change",
    },
    "VIXCLS": {
        "name": "VIX Index",
        "description": "CBOE Volatility Index (daily)",
        "default_transform": "level",
    },
    # Credit
    "TOTCI": {
        "name": "Commercial & Industrial Loans",
        "description": "Commercial and Industrial Loans at All Commercial Banks (weekly)",
        "default_transform": "pct_change_yoy",
    },
    # Housing
    "HOUST": {
        "name": "Housing Starts",
        "description": "Housing Starts in thousands (monthly)",
        "default_transform": "level",
    },
    # Consumer
    "UMCSENT": {
        "name": "Consumer Sentiment",
        "description": "University of Michigan Consumer Sentiment Index (monthly)",
        "default_transform": "level",
    },
    # Industrial
    "INDPRO": {
        "name": "Industrial Production",
        "description": "Industrial Production Index (monthly)",
        "default_transform": "pct_change",
    },
}

# Pre-defined macro control sets for common applications
MACRO_CONTROL_SETS: dict[str, list[str]] = {
    "basic": ["GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS"],
    "comprehensive": [
        "GDPC1",
        "CPIAUCSL",
        "UNRATE",
        "FEDFUNDS",
        "GS10",
        "INDPRO",
        "UMCSENT",
    ],
    "financial": ["SP500", "VIXCLS", "GS10", "TB3MS", "FEDFUNDS"],
    "labor": ["UNRATE", "PAYEMS", "GDPC1"],
    "inflation": ["CPIAUCSL", "PCEPI", "FEDFUNDS", "GS10"],
}


class FREDLoader:
    """FRED data loader for macroeconomic controls.

    Provides standardized access to FRED data with:
    - Automatic frequency alignment
    - Missing data handling
    - Caching for API efficiency
    - Pre-defined macro control sets

    Args:
        api_key: FRED API key (or set FRED_API_KEY env variable)
        cache_dir: Directory for caching data (None to disable)

    Example:
        >>> # Requires live FRED API key + network; skipped in the doctest gate
        >>> loader = FREDLoader()  # doctest: +SKIP
        >>> data = loader.get_macro_controls("2015-01-01", "2023-12-31")  # doctest: +SKIP
        >>> print(data.data.columns)  # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """Initialize FREDLoader.

        Args:
            api_key: FRED API key. If None, looks for FRED_API_KEY env variable.
            cache_dir: Directory for caching. Defaults to ~/.cache/fred_dml/
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self._fred: Any | None = None

        # Set up caching
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "fred_dml"
        else:
            self.cache_dir = Path(cache_dir)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache: dict[str, pd.Series] = {}

    def _get_fred_client(self) -> Any:
        """Get or create FRED API client."""
        if not FREDAPI_AVAILABLE:
            raise ImportError("fredapi package not installed. Install with: pip install fredapi")

        if self._fred is None:
            if self.api_key is None:
                raise ValueError(
                    "FRED API key required. "
                    "Set api_key parameter or FRED_API_KEY environment variable. "
                    "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
                )
            self._fred = Fred(api_key=self.api_key)

        return self._fred

    def _get_cache_path(self, series_id: str) -> Path:
        """Get cache file path for a series."""
        return self.cache_dir / f"{series_id}.parquet"

    def _load_from_cache(self, series_id: str) -> pd.Series | None:
        """Load series from cache if available and recent."""
        cache_path = self._get_cache_path(series_id)

        if not cache_path.exists():
            return None

        # Check if cache is recent (less than 1 day old)
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age > 86400:  # 24 hours
            return None

        try:
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]
        except Exception:
            return None

    def _save_to_cache(self, series_id: str, data: pd.Series) -> None:
        """Save series to cache."""
        if self.cache_dir is None:
            return

        cache_path = self._get_cache_path(series_id)
        try:
            df = pd.DataFrame({series_id: data})
            df.to_parquet(cache_path)
        except Exception as e:
            warnings.warn(f"Failed to cache {series_id}: {e}", stacklevel=2)

    def get_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
    ) -> FREDSeries:
        """Fetch a single FRED series.

        Args:
            series_id: FRED series identifier (e.g., "GDP", "CPIAUCSL")
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available

        Returns:
            FREDSeries with data and metadata

        Raises:
            ImportError: If fredapi not installed
            ValueError: If API key not set or series not found
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(series_id)
            if cached is not None:
                # Filter by date range
                if start_date:
                    cached = cached[cached.index >= start_date]
                if end_date:
                    cached = cached[cached.index <= end_date]

                # Get metadata from standard series or default
                meta = STANDARD_MACRO_SERIES.get(
                    series_id,
                    {"name": series_id, "description": "Custom FRED series"},
                )

                return FREDSeries(
                    series_id=series_id,
                    name=meta["name"],
                    description=meta.get("description", ""),
                    frequency="unknown",
                    units="unknown",
                    data=cached,
                )

        # Fetch from FRED API
        fred = self._get_fred_client()

        try:
            data = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            info = fred.get_series_info(series_id)
        except Exception as e:
            raise ValueError(f"Failed to fetch FRED series {series_id}: {e}") from e

        # Cache the full series (before date filtering)
        if use_cache:
            full_data = fred.get_series(series_id)
            self._save_to_cache(series_id, full_data)

        return FREDSeries(
            series_id=series_id,
            name=info.get("title", series_id),
            description=info.get("notes", ""),
            frequency=info.get("frequency_short", "unknown"),
            units=info.get("units", "unknown"),
            data=data,
        )

    def get_multiple_series(
        self,
        series_ids: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, FREDSeries]:
        """Fetch multiple FRED series.

        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data

        Returns:
            Dict mapping series_id to FREDSeries
        """
        results = {}
        for series_id in series_ids:
            try:
                results[series_id] = self.get_series(series_id, start_date, end_date, use_cache)
            except Exception as e:
                warnings.warn(f"Failed to fetch {series_id}: {e}", stacklevel=2)
                continue

        return results

    def _convert_frequency(
        self,
        series: pd.Series,
        target_freq: FrequencyType,
        method: str = "mean",
    ) -> pd.Series:
        """Convert series to target frequency.

        Args:
            series: Input time series
            target_freq: Target frequency (D/W/M/Q/A)
            method: Aggregation method for downsampling

        Returns:
            Resampled series
        """
        freq_map = {
            "D": "D",
            "W": "W",
            "M": "ME",
            "Q": "QE",
            "A": "YE",
        }

        target = freq_map.get(target_freq, target_freq)

        if method == "mean":
            return series.resample(target).mean()
        elif method == "last":
            return series.resample(target).last()
        elif method == "first":
            return series.resample(target).first()
        elif method == "sum":
            return series.resample(target).sum()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _apply_transform(
        self,
        series: pd.Series,
        transform: str,
    ) -> pd.Series:
        """Apply transformation to series.

        Args:
            series: Input time series
            transform: Transformation type

        Returns:
            Transformed series
        """
        if transform == "level":
            return series
        elif transform == "pct_change":
            return series.pct_change() * 100
        elif transform == "pct_change_yoy":
            return series.pct_change(12) * 100  # Assumes monthly
        elif transform == "diff":
            return series.diff()
        elif transform == "log":
            return np.log(series)
        elif transform == "log_diff":
            return np.log(series).diff() * 100
        else:
            raise ValueError(f"Unknown transform: {transform}")

    def get_macro_controls(
        self,
        start_date: str,
        end_date: str,
        series_set: str = "basic",
        frequency: FrequencyType = "M",
        custom_series: list[str] | None = None,
        transforms: dict[str, str] | None = None,
        fill_method: str = "ffill",
    ) -> MacroControlsResult:
        """Get aligned macro control variables for DML.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            series_set: Pre-defined set ("basic", "comprehensive", "financial", etc.)
            frequency: Target frequency (D/W/M/Q/A)
            custom_series: Additional series to include
            transforms: Dict of series_id -> transform type
            fill_method: Method for filling missing values

        Returns:
            MacroControlsResult with aligned DataFrame and metadata

        Example:
            >>> # Requires live FRED API key + network; skipped in the doctest gate
            >>> loader = FREDLoader()  # doctest: +SKIP
            >>> result = loader.get_macro_controls(  # doctest: +SKIP
            ...     "2015-01-01", "2023-12-31",
            ...     series_set="comprehensive",
            ...     frequency="Q"
            ... )
            >>> X_macro = result.data.values  # doctest: +SKIP
        """
        # Determine series to fetch
        series_ids = MACRO_CONTROL_SETS.get(series_set, MACRO_CONTROL_SETS["basic"]).copy()

        if custom_series:
            series_ids.extend(custom_series)

        # Remove duplicates while preserving order
        series_ids = list(dict.fromkeys(series_ids))

        # Fetch all series
        series_data = self.get_multiple_series(series_ids, start_date, end_date)

        if not series_data:
            raise ValueError("No FRED series could be fetched")

        # Build aligned DataFrame
        all_series = []
        metadata = {}
        missing_pct = {}

        # Determine transforms
        if transforms is None:
            transforms = {}

        for series_id, fred_series in series_data.items():
            # Get transform (use default if not specified)
            default_transform = STANDARD_MACRO_SERIES.get(series_id, {}).get(
                "default_transform", "level"
            )
            transform = transforms.get(series_id, default_transform)

            # Convert frequency
            aligned = self._convert_frequency(fred_series.data, frequency)

            # Apply transform
            transformed = self._apply_transform(aligned, transform)

            # Track missing data
            missing_pct[series_id] = float(transformed.isna().mean() * 100)

            all_series.append(transformed.rename(series_id))

            metadata[series_id] = {
                "name": fred_series.name,
                "description": fred_series.description,
                "units": fred_series.units,
                "transform": transform,
            }

        # Combine into DataFrame
        df = pd.concat(all_series, axis=1)

        # Filter to date range
        if start_date is not None or end_date is not None:
            df = df.loc[start_date:end_date]

        # Fill missing values
        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        elif fill_method == "interpolate":
            df = df.interpolate(method="time")

        # Drop remaining NaN rows
        df = df.dropna()

        return MacroControlsResult(
            data=df,
            metadata=metadata,
            start_date=str(df.index.min().date()),
            end_date=str(df.index.max().date()),
            frequency=frequency,
            missing_pct=missing_pct,
        )

    @staticmethod
    def list_available_series() -> dict[str, dict[str, str]]:
        """List pre-defined FRED series with descriptions.

        Returns:
            Dict of series_id -> metadata
        """
        return STANDARD_MACRO_SERIES.copy()

    @staticmethod
    def list_control_sets() -> dict[str, list[str]]:
        """List pre-defined macro control sets.

        Returns:
            Dict of set_name -> list of series_ids
        """
        return MACRO_CONTROL_SETS.copy()


def create_synthetic_fred_data(
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    frequency: FrequencyType = "M",
    seed: int | None = None,
) -> MacroControlsResult:
    """Create synthetic FRED-like data for testing without API access.

    Generates realistic macro data with:
    - GDP growth: AR(1) with trend
    - Inflation: AR(1) with mean reversion
    - Unemployment: Counter-cyclical to GDP
    - Interest rates: Taylor rule approximation

    Args:
        start_date: Start date
        end_date: End date
        frequency: Data frequency
        seed: Random seed for reproducibility

    Returns:
        MacroControlsResult with synthetic macro data

    Example:
        >>> data = create_synthetic_fred_data(seed=42)
        >>> X_macro = data.data.values  # Use as controls in DML
        >>> list(data.data.columns)
        ['GDPC1', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS']
        >>> bool(X_macro.shape[1] == 4)
        True
    """
    rng = np.random.default_rng(seed)

    # Create date index (use pandas frequency aliases)
    freq_map = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "A": "YE"}
    pd_freq = freq_map.get(frequency, frequency)
    date_range = pd.date_range(start_date, end_date, freq=pd_freq)
    n = len(date_range)

    # GDP growth: AR(1) with positive drift
    gdp_growth = np.zeros(n)
    gdp_growth[0] = 2.0 + rng.normal(0, 0.5)
    for t in range(1, n):
        gdp_growth[t] = 0.5 * gdp_growth[t - 1] + 0.5 * 2.0 + rng.normal(0, 1.0)

    # Inflation: Mean-reverting AR(1)
    inflation = np.zeros(n)
    inflation[0] = 2.0 + rng.normal(0, 0.3)
    for t in range(1, n):
        inflation[t] = 0.7 * inflation[t - 1] + 0.3 * 2.0 + rng.normal(0, 0.5)

    # Unemployment: Counter-cyclical to GDP
    unemployment = np.zeros(n)
    unemployment[0] = 5.0 + rng.normal(0, 0.2)
    for t in range(1, n):
        gdp_effect = -0.3 * (gdp_growth[t] - 2.0)
        unemployment[t] = 0.9 * unemployment[t - 1] + 0.1 * 5.0 + gdp_effect + rng.normal(0, 0.3)
    unemployment = np.clip(unemployment, 2.0, 15.0)

    # Interest rate: Simple Taylor rule
    fed_rate = np.zeros(n)
    fed_rate[0] = 2.0 + rng.normal(0, 0.1)
    for t in range(1, n):
        taylor = 2.0 + 1.5 * (inflation[t] - 2.0) - 0.5 * (unemployment[t] - 5.0)
        fed_rate[t] = 0.8 * fed_rate[t - 1] + 0.2 * taylor + rng.normal(0, 0.1)
    fed_rate = np.clip(fed_rate, 0.0, 10.0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "GDPC1": gdp_growth,
            "CPIAUCSL": inflation,
            "UNRATE": unemployment,
            "FEDFUNDS": fed_rate,
        },
        index=date_range,
    )

    # Metadata
    metadata = {
        "GDPC1": {
            "name": "Real GDP Growth (Synthetic)",
            "description": "AR(1) process with trend",
            "units": "Percent Change",
            "transform": "level",
        },
        "CPIAUCSL": {
            "name": "CPI Inflation (Synthetic)",
            "description": "Mean-reverting AR(1)",
            "units": "Percent Change YoY",
            "transform": "level",
        },
        "UNRATE": {
            "name": "Unemployment Rate (Synthetic)",
            "description": "Counter-cyclical to GDP",
            "units": "Percent",
            "transform": "level",
        },
        "FEDFUNDS": {
            "name": "Federal Funds Rate (Synthetic)",
            "description": "Taylor rule approximation",
            "units": "Percent",
            "transform": "level",
        },
    }

    return MacroControlsResult(
        data=df,
        metadata=metadata,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        missing_pct={"GDPC1": 0.0, "CPIAUCSL": 0.0, "UNRATE": 0.0, "FEDFUNDS": 0.0},
    )
