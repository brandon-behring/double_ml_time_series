"""
Result storage and caching system for validation results.

Provides efficient storage, retrieval, and caching of:
- ValidationResult objects
- DGP configurations
- Simulation outputs
"""

from typing import List, Optional, Dict, Any, cast
from pathlib import Path
from datetime import datetime
import json
import pickle
import hashlib

from dml_ts.validation.validation_result import ValidationResult
from dml_ts.validation.dgp_generator import DGPResult


class ResultStorage:
    """
    Storage manager for validation results.

    Features:
    - JSON storage for ValidationResult (human-readable)
    - Pickle storage for DGPResult (efficient)
    - Automatic timestamping
    - Result indexing for fast retrieval
    - Cache management for DGPs

    Examples:
        >>> storage = ResultStorage(base_dir="results/validation")
        >>> storage.save_result(validation_result, method="bias")
        >>> results = storage.load_results(method="bias")
    """

    def __init__(self, base_dir: Path = Path("results/validation")):
        """
        Initialize result storage.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.dgp_cache_dir = self.base_dir / "dgp_cache"
        self.index_file = self.base_dir / "index.json"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dgp_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize index
        self.index = self._load_index()

    def save_result(
        self,
        result: ValidationResult,
        method: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save ValidationResult to storage.

        Args:
            result: ValidationResult to save
            method: Optional method name (overrides result.method)
            metadata: Optional additional metadata

        Returns:
            Path to saved result file
        """
        # Use method from result if not provided
        if method is None:
            method = result.method

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{method}_{timestamp}.json"
        filepath = self.results_dir / filename

        # Save result as JSON
        with open(filepath, "w") as f:
            result_dict = result.to_dict()
            if metadata:
                result_dict["storage_metadata"] = metadata
            json.dump(result_dict, f, indent=2)

        # Update index
        self._add_to_index(
            filepath=filepath,
            method=method,
            status=result.status,
            timestamp=result.timestamp.isoformat(),
            metadata=metadata or {},
        )

        return filepath

    def load_result(self, filepath: Path) -> ValidationResult:
        """
        Load ValidationResult from file.

        Args:
            filepath: Path to result JSON file

        Returns:
            ValidationResult object
        """
        with open(filepath, "r") as f:
            result_dict = json.load(f)

        return ValidationResult.from_dict(result_dict)

    def load_results(
        self,
        method: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ValidationResult]:
        """
        Load multiple validation results with filtering.

        Args:
            method: Filter by method name
            status: Filter by status (PASS/FAIL/WARNING)
            limit: Maximum number of results to return

        Returns:
            List of ValidationResult objects
        """
        # Filter index
        filtered_entries = []
        for entry in self.index["results"]:
            if method and entry["method"] != method:
                continue
            if status and entry["status"] != status:
                continue
            filtered_entries.append(entry)

        # Sort by timestamp (most recent first)
        filtered_entries.sort(key=lambda x: x["timestamp"], reverse=True)

        # Apply limit
        if limit:
            filtered_entries = filtered_entries[:limit]

        # Load results
        results = []
        for entry in filtered_entries:
            filepath = Path(entry["filepath"])
            if filepath.exists():
                results.append(self.load_result(filepath))

        return results

    def cache_dgp(self, dgp_result: DGPResult, dgp_config: Dict[str, Any]) -> str:
        """
        Cache DGPResult for reuse across validation methods.

        Uses hash of configuration for deduplication.

        Args:
            dgp_result: DGPResult to cache
            dgp_config: Configuration dict (for hash)

        Returns:
            Cache key (hash)
        """
        # Create hash of configuration
        config_str = json.dumps(dgp_config, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        # Save DGP as pickle (more efficient than JSON for numpy arrays)
        cache_path = self.dgp_cache_dir / f"{cache_key}.pkl"

        with open(cache_path, "wb") as f:
            pickle.dump(dgp_result, f)

        # Also save config for reference
        config_path = self.dgp_cache_dir / f"{cache_key}_config.json"
        with open(config_path, "w") as f:
            json.dump(dgp_config, f, indent=2)

        return cache_key

    def load_cached_dgp(self, dgp_config: Dict[str, Any]) -> Optional[DGPResult]:
        """
        Load cached DGPResult if available.

        Args:
            dgp_config: Configuration dict

        Returns:
            DGPResult if cached, None otherwise
        """
        # Create hash
        config_str = json.dumps(dgp_config, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        cache_path = self.dgp_cache_dir / f"{cache_key}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return cast(DGPResult, pickle.load(f))

        return None

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear DGP cache.

        Args:
            older_than_days: Only clear cache older than N days (None = all)

        Returns:
            Number of cache entries cleared
        """
        cleared = 0

        for cache_file in self.dgp_cache_dir.glob("*.pkl"):
            # Check age if specified
            if older_than_days:
                age_days = (
                    datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                ).days
                if age_days < older_than_days:
                    continue

            # Remove cache and config
            cache_file.unlink()
            config_file = cache_file.with_suffix("").with_name(cache_file.stem + "_config.json")
            if config_file.exists():
                config_file.unlink()

            cleared += 1

        return cleared

    def get_summary(self) -> Dict[str, Any]:
        """
        Get storage summary statistics.

        Returns:
            Dictionary with summary stats
        """
        from collections import Counter

        # Count by method and status
        methods = [entry["method"] for entry in self.index["results"]]
        statuses = [entry["status"] for entry in self.index["results"]]

        # Count cache entries
        cache_count = len(list(self.dgp_cache_dir.glob("*.pkl")))

        return {
            "total_results": len(self.index["results"]),
            "methods": dict(Counter(methods)),
            "statuses": dict(Counter(statuses)),
            "cache_entries": cache_count,
            "last_updated": self.index.get("last_updated", "Never"),
        }

    def _load_index(self) -> Dict[str, Any]:
        """Load or create result index."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        else:
            return {"results": [], "last_updated": None}

    def _save_index(self) -> None:
        """Save result index."""
        self.index["last_updated"] = datetime.now().isoformat()

        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def _add_to_index(
        self,
        filepath: Path,
        method: str,
        status: str,
        timestamp: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Add entry to index."""
        entry = {
            "filepath": str(filepath),
            "method": method,
            "status": status,
            "timestamp": timestamp,
            "metadata": metadata,
        }

        self.index["results"].append(entry)
        self._save_index()
