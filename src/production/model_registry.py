"""
Model registry for DML pipeline versioning and serialization.

This module handles the unique challenges of serializing causal inference models:
1. Two-stage nuisance models (propensity + outcome)
2. Cross-fitting fold structure
3. Treatment effect estimates with confidence intervals
4. Hyperparameters for both stages

Unlike standard ML model registries, DML requires:
- Serializing K nuisance model pairs (one per fold)
- Preserving cross-fitting metadata
- Storing covariate transformation pipelines
- Recording HAC bandwidth parameters
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DMLModelVersion:
    """
    Versioned DML model with full serialization support.

    Captures all components needed to reproduce inference:
    - Nuisance models (propensity + outcome) for each fold
    - Cross-fitting configuration
    - Feature transformers
    - HAC parameters for standard error computation

    Attributes:
        version_id: Unique identifier (hash of model + timestamp)
        created_at: Creation timestamp (ISO format)
        model_type: DML variant ("double_ml", "dynamic_dml", "panel_dml", etc.)
        n_folds: Number of cross-fitting folds
        nuisance_models: Serialized nuisance models by fold
        feature_names: List of feature/covariate names
        treatment_name: Name of treatment variable
        outcome_name: Name of outcome variable
        hyperparameters: All hyperparameters for reproducibility
        metrics: Training metrics (R², nuisance fit, etc.)
        metadata: Additional metadata (author, description, etc.)
    """

    version_id: str
    created_at: str
    model_type: str
    n_folds: int
    nuisance_models: Dict[int, Tuple[bytes, bytes]]  # fold -> (propensity, outcome)
    feature_names: List[str]
    treatment_name: str
    outcome_name: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _feature_transformer: Optional[bytes] = None

    @classmethod
    def create(
        cls,
        model_type: str,
        nuisance_models: Dict[int, Tuple[Any, Any]],
        feature_names: List[str],
        treatment_name: str,
        outcome_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feature_transformer: Optional[Any] = None,
    ) -> DMLModelVersion:
        """
        Create a new versioned DML model.

        Args:
            model_type: DML variant identifier
            nuisance_models: Dict mapping fold index to (propensity_model, outcome_model)
            feature_names: Names of covariates/features
            treatment_name: Treatment variable name
            outcome_name: Outcome variable name
            hyperparameters: Full hyperparameter specification
            metrics: Optional training metrics
            metadata: Optional additional metadata
            feature_transformer: Optional sklearn transformer for features

        Returns:
            New DMLModelVersion instance with unique version_id

        Raises:
            ValueError: If nuisance_models is empty or contains None models
        """
        if not nuisance_models:
            raise ValueError("nuisance_models cannot be empty")

        # Serialize nuisance models
        serialized_nuisance: Dict[int, Tuple[bytes, bytes]] = {}
        for fold_idx, (propensity, outcome) in nuisance_models.items():
            if propensity is None or outcome is None:
                raise ValueError(f"Fold {fold_idx} contains None model")
            serialized_nuisance[fold_idx] = (
                pickle.dumps(propensity),
                pickle.dumps(outcome),
            )

        # Serialize feature transformer if provided
        serialized_transformer = None
        if feature_transformer is not None:
            serialized_transformer = pickle.dumps(feature_transformer)

        # Generate version ID from content hash
        timestamp = datetime.utcnow().isoformat()
        content_hash = cls._compute_hash(
            model_type, serialized_nuisance, feature_names, hyperparameters, timestamp
        )
        version_id = f"dml-{content_hash[:12]}"

        return cls(
            version_id=version_id,
            created_at=timestamp,
            model_type=model_type,
            n_folds=len(nuisance_models),
            nuisance_models=serialized_nuisance,
            feature_names=feature_names,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            hyperparameters=hyperparameters,
            metrics=metrics or {},
            metadata=metadata or {},
            _feature_transformer=serialized_transformer,
        )

    @staticmethod
    def _compute_hash(
        model_type: str,
        nuisance_models: Dict[int, Tuple[bytes, bytes]],
        feature_names: List[str],
        hyperparameters: Dict[str, Any],
        timestamp: str,
    ) -> str:
        """Compute SHA-256 hash of model content for versioning."""
        hasher = hashlib.sha256()
        hasher.update(model_type.encode())
        hasher.update(json.dumps(feature_names, sort_keys=True).encode())
        hasher.update(json.dumps(hyperparameters, sort_keys=True, default=str).encode())
        hasher.update(timestamp.encode())
        for fold_idx in sorted(nuisance_models.keys()):
            prop_bytes, out_bytes = nuisance_models[fold_idx]
            hasher.update(prop_bytes)
            hasher.update(out_bytes)
        return hasher.hexdigest()

    def get_nuisance_models(
        self, fold: Optional[int] = None
    ) -> Union[Dict[int, Tuple[Any, Any]], Tuple[Any, Any],]:
        """
        Deserialize and return nuisance models.

        Args:
            fold: Optional specific fold index. If None, returns all folds.

        Returns:
            If fold specified: (propensity_model, outcome_model) tuple
            If fold None: Dict mapping fold index to model tuples

        Raises:
            KeyError: If specified fold not found
        """
        if fold is not None:
            if fold not in self.nuisance_models:
                raise KeyError(
                    f"Fold {fold} not found. Available: {list(self.nuisance_models.keys())}"
                )
            prop_bytes, out_bytes = self.nuisance_models[fold]
            return (pickle.loads(prop_bytes), pickle.loads(out_bytes))

        return {
            fold_idx: (pickle.loads(prop_bytes), pickle.loads(out_bytes))
            for fold_idx, (prop_bytes, out_bytes) in self.nuisance_models.items()
        }

    def get_feature_transformer(self) -> Optional[Any]:
        """Return deserialized feature transformer, or None if not set."""
        if self._feature_transformer is None:
            return None
        return pickle.loads(self._feature_transformer)

    def predict_cate(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict conditional average treatment effect (CATE).

        Uses stored nuisance models for orthogonalized prediction.
        For production, typically uses a single fold or averages across folds.

        Args:
            X: Feature matrix (n_samples, n_features)
            return_std: If True, also return standard errors

        Returns:
            CATE predictions, optionally with standard errors

        Note:
            This is a simplified inference path. For full uncertainty
            quantification, use the complete DML fit with bootstrap/HAC.
        """
        # For production inference, we typically use a meta-learner trained
        # on the orthogonalized residuals. This requires the full pipeline.
        # Here we provide a simplified averaging approach.

        n_samples = X.shape[0]
        cate_predictions = np.zeros((self.n_folds, n_samples))

        models = self.get_nuisance_models()
        transformer = self.get_feature_transformer()

        if transformer is not None:
            X = transformer.transform(X)

        # This is a placeholder - actual CATE prediction requires
        # the full treatment effect model, not just nuisance models.
        # In practice, you'd store the CATE model separately.
        raise NotImplementedError(
            "Full CATE prediction requires the treatment effect estimator, "
            "not just nuisance models. Use InsuranceDMLPipeline.predict() instead."
        )

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save model version to disk.

        Args:
            path: Directory to save model (will create version_id subdirectory)

        Returns:
            Path to saved model directory
        """
        path = Path(path)
        model_dir = path / self.version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        metadata = {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "model_type": self.model_type,
            "n_folds": self.n_folds,
            "feature_names": self.feature_names,
            "treatment_name": self.treatment_name,
            "outcome_name": self.outcome_name,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save nuisance models
        for fold_idx, (prop_bytes, out_bytes) in self.nuisance_models.items():
            with open(model_dir / f"propensity_fold_{fold_idx}.pkl", "wb") as f:
                f.write(prop_bytes)
            with open(model_dir / f"outcome_fold_{fold_idx}.pkl", "wb") as f:
                f.write(out_bytes)

        # Save feature transformer if present
        if self._feature_transformer is not None:
            with open(model_dir / "feature_transformer.pkl", "wb") as f:
                f.write(self._feature_transformer)

        return model_dir

    @classmethod
    def load(cls, path: Union[str, Path]) -> DMLModelVersion:
        """
        Load model version from disk.

        Args:
            path: Path to model directory (containing metadata.json)

        Returns:
            Loaded DMLModelVersion instance

        Raises:
            FileNotFoundError: If metadata.json not found
            ValueError: If required files are missing
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load nuisance models
        nuisance_models: Dict[int, Tuple[bytes, bytes]] = {}
        for fold_idx in range(metadata["n_folds"]):
            prop_path = path / f"propensity_fold_{fold_idx}.pkl"
            out_path = path / f"outcome_fold_{fold_idx}.pkl"

            if not prop_path.exists() or not out_path.exists():
                raise ValueError(f"Missing model files for fold {fold_idx}")

            with open(prop_path, "rb") as f:
                prop_bytes = f.read()
            with open(out_path, "rb") as f:
                out_bytes = f.read()

            nuisance_models[fold_idx] = (prop_bytes, out_bytes)

        # Load feature transformer if present
        transformer_path = path / "feature_transformer.pkl"
        transformer_bytes = None
        if transformer_path.exists():
            with open(transformer_path, "rb") as f:
                transformer_bytes = f.read()

        return cls(
            version_id=metadata["version_id"],
            created_at=metadata["created_at"],
            model_type=metadata["model_type"],
            n_folds=metadata["n_folds"],
            nuisance_models=nuisance_models,
            feature_names=metadata["feature_names"],
            treatment_name=metadata["treatment_name"],
            outcome_name=metadata["outcome_name"],
            hyperparameters=metadata["hyperparameters"],
            metrics=metadata.get("metrics", {}),
            metadata=metadata.get("metadata", {}),
            _feature_transformer=transformer_bytes,
        )

    def __repr__(self) -> str:
        return (
            f"DMLModelVersion(version_id='{self.version_id}', "
            f"model_type='{self.model_type}', "
            f"n_folds={self.n_folds}, "
            f"n_features={len(self.feature_names)}, "
            f"created_at='{self.created_at}')"
        )


class DMLModelRegistry:
    """
    Registry for managing multiple DML model versions.

    Provides:
    - Version tracking with unique IDs
    - Production vs staging promotion
    - Rollback capability
    - Lineage tracking (which data/config produced which model)

    Attributes:
        base_path: Root directory for model storage
        production_version: Currently deployed version ID (or None)
        staging_version: Candidate version for promotion (or None)
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize model registry.

        Args:
            base_path: Root directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.base_path / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load or initialize registry state."""
        if self._registry_file.exists():
            with open(self._registry_file) as f:
                state = json.load(f)
            self.production_version: Optional[str] = state.get("production_version")
            self.staging_version: Optional[str] = state.get("staging_version")
            self._versions: List[str] = state.get("versions", [])
        else:
            self.production_version = None
            self.staging_version = None
            self._versions = []

    def _save_registry(self) -> None:
        """Persist registry state."""
        state = {
            "production_version": self.production_version,
            "staging_version": self.staging_version,
            "versions": self._versions,
        }
        with open(self._registry_file, "w") as f:
            json.dump(state, f, indent=2)

    def register(self, model: DMLModelVersion) -> str:
        """
        Register a new model version.

        Args:
            model: DMLModelVersion to register

        Returns:
            Version ID of registered model
        """
        model_path = model.save(self.base_path)

        if model.version_id not in self._versions:
            self._versions.append(model.version_id)
            self._save_registry()

        return model.version_id

    def get(self, version_id: str) -> DMLModelVersion:
        """
        Retrieve a specific model version.

        Args:
            version_id: Version to retrieve

        Returns:
            DMLModelVersion instance

        Raises:
            KeyError: If version not found
        """
        if version_id not in self._versions:
            raise KeyError(f"Version {version_id} not found in registry")

        model_path = self.base_path / version_id
        return DMLModelVersion.load(model_path)

    def get_production(self) -> Optional[DMLModelVersion]:
        """Return current production model, or None if not set."""
        if self.production_version is None:
            return None
        return self.get(self.production_version)

    def get_staging(self) -> Optional[DMLModelVersion]:
        """Return current staging model, or None if not set."""
        if self.staging_version is None:
            return None
        return self.get(self.staging_version)

    def promote_to_staging(self, version_id: str) -> None:
        """
        Promote a version to staging.

        Args:
            version_id: Version to promote

        Raises:
            KeyError: If version not found
        """
        if version_id not in self._versions:
            raise KeyError(f"Version {version_id} not found")

        self.staging_version = version_id
        self._save_registry()

    def promote_to_production(self, version_id: Optional[str] = None) -> str:
        """
        Promote a version to production.

        Args:
            version_id: Version to promote (defaults to current staging)

        Returns:
            Version ID promoted to production

        Raises:
            ValueError: If no version specified and no staging version
            KeyError: If specified version not found
        """
        if version_id is None:
            if self.staging_version is None:
                raise ValueError("No staging version to promote")
            version_id = self.staging_version

        if version_id not in self._versions:
            raise KeyError(f"Version {version_id} not found")

        # Archive previous production version
        old_production = self.production_version

        self.production_version = version_id
        self.staging_version = None
        self._save_registry()

        return version_id

    def rollback(self, to_version: Optional[str] = None) -> str:
        """
        Rollback production to a previous version.

        Args:
            to_version: Specific version to rollback to.
                       If None, rolls back to previous production version.

        Returns:
            Version ID rolled back to

        Raises:
            ValueError: If no previous version available
            KeyError: If specified version not found
        """
        if to_version is not None:
            if to_version not in self._versions:
                raise KeyError(f"Version {to_version} not found")
            self.production_version = to_version
            self._save_registry()
            return to_version

        # Find previous version (simple: just go one back in list)
        if self.production_version is None:
            raise ValueError("No production version to rollback from")

        try:
            current_idx = self._versions.index(self.production_version)
            if current_idx == 0:
                raise ValueError("No previous version available")
            previous_version = self._versions[current_idx - 1]
        except ValueError:
            raise ValueError("Production version not found in version list")

        self.production_version = previous_version
        self._save_registry()
        return previous_version

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all registered versions with metadata.

        Returns:
            List of dicts with version info (id, created_at, is_production, is_staging)
        """
        result = []
        for version_id in self._versions:
            try:
                model = self.get(version_id)
                result.append(
                    {
                        "version_id": version_id,
                        "created_at": model.created_at,
                        "model_type": model.model_type,
                        "n_folds": model.n_folds,
                        "metrics": model.metrics,
                        "is_production": version_id == self.production_version,
                        "is_staging": version_id == self.staging_version,
                    }
                )
            except Exception as e:
                result.append(
                    {
                        "version_id": version_id,
                        "error": str(e),
                        "is_production": version_id == self.production_version,
                        "is_staging": version_id == self.staging_version,
                    }
                )
        return result

    def __repr__(self) -> str:
        return (
            f"DMLModelRegistry(base_path='{self.base_path}', "
            f"n_versions={len(self._versions)}, "
            f"production={self.production_version}, "
            f"staging={self.staging_version})"
        )
