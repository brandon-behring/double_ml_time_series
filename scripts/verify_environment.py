#!/usr/bin/env python3
"""
Environment Verification Script for Double ML Volume 2

Tests that all required Python packages are installed and importable
before beginning chapter work.

Usage:
    python scripts/verify_environment.py

Exit codes:
    0: All dependencies verified successfully
    1: One or more dependencies failed to import
"""

import sys
from typing import List, Tuple


def verify_imports() -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Test all required package imports.

    Returns:
        Tuple of (successful_imports, failed_imports)
        failed_imports contains (package_name, error_message) tuples
    """
    successful: List[str] = []
    failed: List[Tuple[str, str]] = []

    # Core causal inference packages
    packages = [
        ("econml", "EconML (Microsoft's causal ML library)"),
        ("econml.dml", "EconML DML module"),
        ("dowhy", "DoWhy (causal inference library)"),
        ("causalml", "CausalML (Uber's causal ML library)"),
        # Machine learning packages
        ("sklearn", "scikit-learn"),
        ("sklearn.linear_model", "scikit-learn linear models"),
        ("sklearn.ensemble", "scikit-learn ensemble models"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        # Data manipulation and numerical computing
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("scipy", "SciPy"),
        ("scipy.stats", "SciPy statistics"),
        # FRED API
        ("fredapi", "FRED API client"),
        # Plotting
        ("matplotlib", "Matplotlib"),
        ("matplotlib.pyplot", "Matplotlib pyplot"),
        ("seaborn", "Seaborn"),
        # Parallel execution
        ("joblib", "joblib (parallel execution)"),
        # Testing
        ("pytest", "pytest"),
        # Code quality
        ("black", "Black formatter"),
        ("mypy", "mypy type checker"),
    ]

    for package, description in packages:
        try:
            __import__(package)
            successful.append(f"✓ {description} ({package})")
        except ImportError as e:
            failed.append((package, str(e)))

    return successful, failed


def main() -> int:
    """Run verification and print results."""
    print("=" * 70)
    print("Double ML Volume 2: Environment Verification")
    print("=" * 70)
    print()

    successful, failed = verify_imports()

    # Print successful imports
    if successful:
        print(f"✅ Successfully imported {len(successful)} packages:")
        print()
        for item in successful:
            print(f"  {item}")
        print()

    # Print failed imports
    if failed:
        print(f"❌ Failed to import {len(failed)} packages:")
        print()
        for package, error in failed:
            print(f"  ✗ {package}")
            print(f"    Error: {error}")
        print()
        print("To fix: Activate venv and run 'pip install -r requirements.txt'")
        print()
        return 1

    # All passed
    print("=" * 70)
    print("✅ All dependencies verified successfully!")
    print("=" * 70)
    print()
    print("Environment is ready for Chapter 1 development.")
    print()

    # Print Python version info
    print(f"Python version: {sys.version}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
