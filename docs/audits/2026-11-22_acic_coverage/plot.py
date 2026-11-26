"""Plot repeated-draw coverage and pointwise exact Monte Carlo intervals by design."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.6), sharex=True)
    for ax, outcome in zip(axes, ["binary", "continuous"], strict=True):
        group = summary[summary.outcome_type == outcome].sort_values("design")
        for row, values in enumerate(group.itertuples()):
            color = "#a63b21" if values.reject_undercoverage else "#1e5265"
            lo, hi = 100 * np.array([values.coverage_mc_ci_low, values.coverage_mc_ci_high])
            ax.hlines(row, lo, hi, color=color, linewidth=1.5)
            ax.plot(
                100 * values.coverage,
                row,
                "D" if values.reject_undercoverage else "o",
                color=color,
                markersize=5,
            )
        ax.axvline(95, linestyle="--", color="#505050", linewidth=1, zorder=0)
        ax.set_yticks(range(len(group)), group.design)
        ax.invert_yaxis()
        ax.set_title(outcome.title() + " outcomes", loc="left", fontsize=13)
        ax.set_xlabel("Intervals containing the published PATE (%)")
        ax.set_ylabel("Organizer design")
        ax.set_xticks([0, 25, 50, 75, 95])
        ax.set_xlim(-3, 102)
        ax.grid(axis="x", color="#e3e6e7", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Coverage of nominal 95% AIPW intervals in ACIC 2019", fontsize=16)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#1e5265",
                marker="o",
                label="100 draws with exact 95% Monte Carlo interval",
            ),
            Line2D(
                [],
                [],
                color="#a63b21",
                marker="D",
                linestyle="none",
                label="Undercoverage after Holm adjustment",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
