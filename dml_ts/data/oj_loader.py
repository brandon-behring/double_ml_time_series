"""Dominick's Orange Juice Dataset Loader.

Loads the OJ dataset from Microsoft Azure blob storage. This dataset contains
store-level panel data on orange juice sales from Dominick's Finer Foods,
a major Chicago-area supermarket chain.

The dataset is widely used for causal inference demonstrations, particularly
for estimating price elasticity of demand with confounders like income and
demographics.

Data Source
-----------
University of Chicago Booth School of Business, Kilts Center for Marketing.
https://research.chicagobooth.edu/kilts/marketing-databases/dominicks

GitHub mirror (used here):
https://github.com/gchoi/Dataset/raw/refs/heads/master/oj.csv

References
----------
Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment
and structural parameters. The Econometrics Journal, 21(1), C1-C68.

Montgomery, A. (1997). Creating Micro-Marketing Pricing Strategies Using
Supermarket Scanner Data. Marketing Science, 16(4), 315-337.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# GitHub mirror URL for OJ dataset (Azure blob now private)
OJ_DATASET_URL = "https://github.com/gchoi/Dataset/raw/refs/heads/master/oj.csv"

# Expected SHA256 hash for data integrity verification
OJ_DATASET_HASH = None  # Will be set after first download verification


@dataclass
class OJDataset:
    """Container for Orange Juice dataset prepared for DML analysis.

    This dataclass holds the outcome, treatment, and confounders in a format
    ready for DML estimation. The dataset is preprocessed to match standard
    econometric conventions.

    Attributes
    ----------
    Y : NDArray[np.float64]
        Outcome variable: log(units sold) = logmove.
        Shape: (n_samples,)
    T : NDArray[np.float64]
        Treatment variable: log(price).
        Shape: (n_samples,)
    X : NDArray[np.float64]
        Confounder matrix: [feat, INCOME, AGE60, ...].
        Shape: (n_samples, n_features)
    feature_names : list[str]
        Names of confounders in X columns.
    n_samples : int
        Number of observations.
    n_features : int
        Number of confounders.
    raw_df : pd.DataFrame
        Original dataframe before transformation (for reference).

    Examples
    --------
    >>> loader = OJDataLoader()
    >>> data = loader.load()
    >>> print(f"Samples: {data.n_samples}, Features: {data.n_features}")
    Samples: 28947, Features: 3
    >>> from dml_ts.dml import double_ml
    >>> result = double_ml(data.Y, data.T, data.X)
    """

    Y: NDArray[np.float64]
    T: NDArray[np.float64]
    X: NDArray[np.float64]
    feature_names: list[str]
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    raw_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Compute derived attributes after initialization."""
        self.n_samples = len(self.Y)
        self.n_features = self.X.shape[1] if self.X.ndim > 1 else 1

    def __repr__(self) -> str:
        return (
            f"OJDataset(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"features={self.feature_names})"
        )

    def summary(self) -> str:
        """Return formatted summary of dataset."""
        return f"""
Orange Juice Dataset Summary
============================
Observations:     {self.n_samples:,}
Features:         {self.n_features}
Feature names:    {", ".join(self.feature_names)}

Outcome (Y = log sales):
  Mean:           {np.mean(self.Y):.3f}
  Std:            {np.std(self.Y):.3f}
  Range:          [{np.min(self.Y):.3f}, {np.max(self.Y):.3f}]

Treatment (T = log price):
  Mean:           {np.mean(self.T):.3f}
  Std:            {np.std(self.T):.3f}
  Range:          [{np.min(self.T):.3f}, {np.max(self.T):.3f}]

Expected elasticity: ~-2.5 to -3.5 (standard finding in literature)
"""


class OJDataLoader:
    """Loader for Dominick's Orange Juice dataset.

    Downloads and preprocesses the OJ dataset for DML price elasticity
    estimation. Supports caching to avoid repeated downloads.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Directory to cache downloaded data. Defaults to None (no caching).
    features : list[str], optional
        Confounders to include. Defaults to ["feat", "INCOME", "AGE60"].
    brand : str, optional
        Filter to specific brand ("dominicks", "minute.maid", "tropicana").
        Defaults to None (all brands).

    Examples
    --------
    >>> loader = OJDataLoader()
    >>> data = loader.load()
    >>> print(data)
    OJDataset(n_samples=28947, n_features=3, features=['feat', 'INCOME', 'AGE60'])

    >>> # Custom features
    >>> loader = OJDataLoader(features=["feat", "INCOME", "AGE60", "EDUC"])
    >>> data = loader.load()
    """

    DEFAULT_FEATURES = ["feat", "INCOME", "AGE60"]

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        features: list[str] | None = None,
        brand: str | None = None,
    ) -> None:
        """Initialize the OJ data loader."""
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.features = features if features is not None else self.DEFAULT_FEATURES.copy()
        self.brand = brand
        self._raw_df: pd.DataFrame | None = None

    def load(self, force_download: bool = False) -> OJDataset:
        """Load and preprocess the OJ dataset.

        Parameters
        ----------
        force_download : bool
            If True, re-download even if cached. Default: False.

        Returns
        -------
        OJDataset
            Preprocessed dataset ready for DML analysis.

        Raises
        ------
        ValueError
            If requested features are not in dataset.
        RuntimeError
            If download fails.
        """
        # Load raw data
        df = self._load_raw(force_download)

        # Filter by brand if specified
        if self.brand is not None:
            df = df[df["brand"] == self.brand].copy()
            if len(df) == 0:
                raise ValueError(
                    f"Brand '{self.brand}' not found. "
                    f"Available brands: {df['brand'].unique().tolist()}"
                )

        # Validate features exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(
                f"Features not in dataset: {missing}. Available: {df.columns.tolist()}"
            )

        # Extract and transform variables
        Y = df["logmove"].values.astype(np.float64)
        T = np.log(df["price"].values).astype(np.float64)
        X = df[self.features].values.astype(np.float64)

        # Drop rows with any missing values
        mask = ~(np.isnan(Y) | np.isnan(T) | np.any(np.isnan(X), axis=1))
        if not mask.all():
            n_dropped = int((~mask).sum())
            warnings.warn(
                f"Dropped {n_dropped} rows with missing values from OJ dataset",
                stacklevel=2,
            )
            Y, T, X = Y[mask], T[mask], X[mask]
            df = df[mask].copy()

        return OJDataset(
            Y=Y,
            T=T,
            X=X,
            feature_names=self.features.copy(),
            raw_df=df,
        )

    def _load_raw(self, force_download: bool = False) -> pd.DataFrame:
        """Load raw dataframe from cache or download."""
        # Check cache first
        if self.cache_dir and not force_download:
            cache_path = self.cache_dir / "oj_large.csv"
            if cache_path.exists():
                self._raw_df = pd.read_csv(cache_path)
                return self._raw_df

        # Download from Azure
        try:
            self._raw_df = pd.read_csv(OJ_DATASET_URL)
        except Exception as e:
            raise RuntimeError(f"Failed to download OJ dataset from {OJ_DATASET_URL}: {e}") from e

        # Cache if directory specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._raw_df.to_csv(self.cache_dir / "oj_large.csv", index=False)

        return self._raw_df

    def get_available_features(self) -> list[str]:
        """Return list of available features in the dataset.

        Returns
        -------
        list[str]
            Column names available for use as confounders.
        """
        if self._raw_df is None:
            self._load_raw()
        assert self._raw_df is not None
        return list(self._raw_df.columns)

    def get_brand_options(self) -> list[str]:
        """Return list of available brand filters.

        Returns
        -------
        list[str]
            Brand names that can be used for filtering.
        """
        if self._raw_df is None:
            self._load_raw()
        assert self._raw_df is not None
        return list(self._raw_df["brand"].unique())
