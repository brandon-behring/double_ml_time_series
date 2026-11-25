"""Compare the repository's eligibility fits with published fivefold PLR benchmarks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """Run the two single-partition fits on the tracked 1991 SIPP extract."""
    from dml_ts.validation.empirical_replication import FourZeroOneKReplication

    fixture = ROOT / "test/fixtures/401k_data.csv"
    comparison = FourZeroOneKReplication(data_path=str(fixture), random_state=42)
    print("401(k) eligibility and net financial assets (1991 dollars)")
    print(f"{len(comparison.load_data()):,} observations from {fixture.name}")
    print("Python fits use one fivefold partition and 11 effect modifiers, including nifa and tw.")
    print("Table 2B uses nine controls, a constant slope and medians over 100 partitions.\n")

    results = {}
    for label, fit in (
        ("Random forest", comparison.replicate_plr_rf),
        ("Lasso", comparison.replicate_plr_lasso),
    ):
        result = fit()
        results[result.method] = result
        print(label)
        print(f"  Mean eligibility effect        ${result.ate_estimate:,.2f}")
        print(f"  SE for the mean effect         ${result.std_error:,.2f}")
        print(
            f"  95% interval                   [${result.ci_lower:,.2f}, ${result.ci_upper:,.2f}]"
        )
        print(f"  Table 2B fivefold reference    ${result.published_ate:,.2f}")
        print(
            f"  Python minus reference         {result.difference:+,.2f} dollars ({result.rel_difference:+.1%})\n"
        )

    print(
        "Table 3 estimates the effect of participation for compliers using eligibility as an instrument."
    )
    print("All 42 published cells and fresh author-code refits are in")
    print("docs/audits/2026-11-19_time_series_dml/07-401k-replication.md")
    return results


if __name__ == "__main__":
    main()
