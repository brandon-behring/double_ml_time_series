"""
Plotting utilities for validation results.

Provides consistent, professional styling across all validation plots.
All functions use the same color scheme, fonts, and layout.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional color palette (colorblind-friendly)
COLORS = {
    "primary": "#0173B2",  # Blue
    "secondary": "#DE8F05",  # Orange
    "success": "#029E73",  # Green
    "warning": "#D55E00",  # Red-orange
    "error": "#CC78BC",  # Purple
    "neutral": "#949494",  # Gray
}

# Plot styling defaults
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
}


def set_plot_style() -> None:
    """Set consistent plot style for all validation plots."""
    # Use seaborn for base styling
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # Apply custom parameters
    plt.rcParams.update(PLOT_STYLE)


def plot_bias_distribution(
    biases: np.ndarray,
    true_effect: float = 0.0,
    title: str = "Bias Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot distribution of bias estimates.

    Args:
        biases: Array of bias values (estimate - true_effect)
        true_effect: True treatment effect (for reference line)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(
        biases,
        bins=30,
        density=True,
        alpha=0.7,
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=0.5,
    )

    # KDE overlay
    from scipy import stats

    kde = stats.gaussian_kde(biases)
    x_range = np.linspace(biases.min(), biases.max(), 200)
    ax.plot(x_range, kde(x_range), color=COLORS["secondary"], linewidth=2, label="KDE")

    # Reference line at zero bias
    ax.axvline(0, color=COLORS["error"], linestyle="--", linewidth=2, label="Zero Bias", alpha=0.8)

    # Mean bias line
    mean_bias = np.mean(biases)
    ax.axvline(
        mean_bias,
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f"Mean Bias: {mean_bias:.3f}",
        alpha=0.8,
    )

    # Styling
    ax.set_xlabel("Bias (Estimate - True Effect)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Add summary statistics box
    stats_text = f"Mean: {np.mean(biases):.3f}\nStd: {np.std(biases):.3f}\n"
    stats_text += f"Median: {np.median(biases):.3f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_mse_curves(
    results: pd.DataFrame,
    x_var: str = "n",
    group_var: Optional[str] = None,
    title: str = "MSE vs Sample Size",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot MSE curves across varying parameter (typically sample size).

    Args:
        results: DataFrame with columns [x_var, 'mse', group_var (optional)]
        x_var: Variable for x-axis (e.g., 'n', 'p', 'confounding')
        group_var: Variable for grouping (creates multiple lines)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if group_var:
        # Multiple lines (one per group)
        for i, (group_val, group_df) in enumerate(results.groupby(group_var)):
            color = list(COLORS.values())[i % len(COLORS)]
            ax.plot(
                group_df[x_var],
                group_df["rmse"],  # Use RMSE for interpretability
                marker="o",
                linewidth=2,
                markersize=6,
                label=f"{group_var}={group_val}",
                color=color,
            )
    else:
        # Single line
        ax.plot(
            results[x_var],
            results["rmse"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS["primary"],
        )

    # Styling
    ax.set_xlabel(x_var.upper(), fontweight="bold")
    ax.set_ylabel("RMSE", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)

    if group_var:
        ax.legend(frameon=True, fancybox=True, shadow=True)

    # Log scale for x-axis if sample size
    if x_var == "n":
        ax.set_xscale("log")
        ax.set_xlabel("Sample Size (n)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_coverage_rates(
    results: pd.DataFrame,
    x_var: str = "n",
    group_var: Optional[str] = None,
    nominal_level: float = 0.95,
    title: str = "Coverage Rates",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confidence interval coverage rates.

    Args:
        results: DataFrame with columns [x_var, 'coverage', group_var (optional)]
        x_var: Variable for x-axis
        group_var: Variable for grouping
        nominal_level: Nominal coverage level (e.g., 0.95 for 95% CI)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Nominal level reference line
    ax.axhline(
        nominal_level,
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f"Nominal ({nominal_level:.0%})",
        alpha=0.8,
    )

    # Tolerance bands (±2%)
    ax.axhspan(
        nominal_level - 0.02,
        nominal_level + 0.02,
        alpha=0.2,
        color=COLORS["success"],
        label="±2% Tolerance",
    )

    if group_var:
        # Multiple lines
        for i, (group_val, group_df) in enumerate(results.groupby(group_var)):
            color = list(COLORS.values())[i % len(COLORS)]
            ax.plot(
                group_df[x_var],
                group_df["coverage"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=f"{group_var}={group_val}",
                color=color,
            )
    else:
        # Single line
        ax.plot(
            results[x_var],
            results["coverage"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS["primary"],
            label="Observed Coverage",
        )

    # Styling
    ax.set_xlabel(x_var.upper(), fontweight="bold")
    ax.set_ylabel("Coverage Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_ylim((0.88, 1.0))  # Focus on relevant range
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    if x_var == "n":
        ax.set_xscale("log")
        ax.set_xlabel("Sample Size (n)", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_power_curves(
    results: pd.DataFrame,
    x_var: str = "effect_size",
    group_var: Optional[str] = None,
    alpha: float = 0.05,
    title: str = "Statistical Power",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot statistical power curves.

    Args:
        results: DataFrame with columns [x_var, 'power', group_var (optional)]
        x_var: Variable for x-axis (typically 'effect_size')
        group_var: Variable for grouping
        alpha: Significance level
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Minimum power threshold (80%)
    ax.axhline(
        0.8, color=COLORS["warning"], linestyle="--", linewidth=2, label="80% Power", alpha=0.8
    )

    if group_var:
        # Multiple lines
        for i, (group_val, group_df) in enumerate(results.groupby(group_var)):
            color = list(COLORS.values())[i % len(COLORS)]
            ax.plot(
                group_df[x_var],
                group_df["power"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=f"{group_var}={group_val}",
                color=color,
            )
    else:
        # Single line
        ax.plot(
            results[x_var],
            results["power"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS["primary"],
        )

    # Styling
    ax.set_xlabel("Effect Size (Cohen's d)", fontweight="bold")
    ax.set_ylabel("Statistical Power", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_ylim((0, 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_validation_summary(
    results: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create summary dashboard of all validation results.

    Args:
        results: List of validation result dictionaries
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure with subplots
    """
    set_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Validation Summary Dashboard", fontsize=16, fontweight="bold", y=0.995)

    # Extract metrics
    methods = [r["method"] for r in results]
    biases = [r["bias"] for r in results]
    rmses = [r["mse"] ** 0.5 for r in results]  # Convert MSE to RMSE
    coverages = [r["coverage"] for r in results]
    statuses = [r["status"] for r in results]

    # Color by status
    status_colors = {
        "PASS": COLORS["success"],
        "WARNING": COLORS["warning"],
        "FAIL": COLORS["error"],
    }
    colors = [status_colors[s] for s in statuses]

    # Plot 1: Bias by method
    axes[0, 0].barh(methods, biases, color=colors)
    axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Bias", fontweight="bold")
    axes[0, 0].set_title("Bias by Method", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    # Plot 2: RMSE by method
    axes[0, 1].barh(methods, rmses, color=colors)
    axes[0, 1].set_xlabel("RMSE", fontweight="bold")
    axes[0, 1].set_title("RMSE by Method", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    # Plot 3: Coverage by method
    axes[1, 0].barh(methods, coverages, color=colors)
    axes[1, 0].axvline(0.95, color=COLORS["success"], linestyle="--", linewidth=2, label="Nominal")
    axes[1, 0].set_xlabel("Coverage Rate", fontweight="bold")
    axes[1, 0].set_title("Coverage by Method", fontweight="bold")
    axes[1, 0].set_xlim([0.88, 1.0])
    axes[1, 0].grid(True, alpha=0.3, axis="x")
    axes[1, 0].legend()

    # Plot 4: Status summary
    from collections import Counter

    status_counts = Counter(statuses)
    axes[1, 1].bar(
        status_counts.keys(),
        status_counts.values(),
        color=[status_colors[s] for s in status_counts.keys()],
    )
    axes[1, 1].set_ylabel("Count", fontweight="bold")
    axes[1, 1].set_title("Validation Status Summary", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
