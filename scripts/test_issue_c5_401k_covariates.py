"""Measure how two additional wealth variables change the repository's 401(k) fits.

Both Python specifications use the selected columns as effect modifiers on one
fivefold split. The paper uses nine controls in its constant-slope Table 2B.
"""

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(output_dir=ROOT / "results/impact_analysis"):
    """Compare signed estimates and mean-effect intervals on the same data and seed."""
    from dml_ts.validation.empirical_replication import FourZeroOneKReplication

    fixture = ROOT / "test/fixtures/401k_data.csv"
    data = pd.read_csv(fixture)
    rows = []
    with TemporaryDirectory() as tmp:
        nine_columns = Path(tmp) / "nine_controls.csv"
        data.drop(columns=["nifa", "tw"]).to_csv(nine_columns, index=False)
        for label, path in (
            ("Nine controls from the paper", nine_columns),
            ("Eleven controls including nifa and tw", fixture),
        ):
            comparison = FourZeroOneKReplication(data_path=str(path), random_state=42)
            row = {"configuration": label}
            for prefix, fit in (
                ("rf", comparison.replicate_plr_rf),
                ("lasso", comparison.replicate_plr_lasso),
            ):
                result = fit()
                row["n_covariates"] = result.metadata["n_controls"]
                row.update(
                    {
                        f"{prefix}_ate": result.ate_estimate,
                        f"{prefix}_ci_lower": result.ci_lower,
                        f"{prefix}_ci_upper": result.ci_upper,
                        f"{prefix}_table2b_fivefold": result.published_ate,
                        f"{prefix}_estimate_minus_table2b": result.difference,
                        f"{prefix}_diff_from_published": abs(result.difference),
                        f"{prefix}_pct_diff": 100 * abs(result.rel_difference),
                    }
                )
            rows.append(row)

    table = pd.DataFrame(rows)
    print(
        f"Eligibility effect on net financial assets (dollars), {len(data):,} observations, seed 42"
    )
    print(
        table[
            [
                "configuration",
                "rf_ate",
                "rf_ci_lower",
                "rf_ci_upper",
                "lasso_ate",
                "lasso_ci_lower",
                "lasso_ci_upper",
            ]
        ].to_string(index=False, float_format="{:,.2f}".format)
    )
    print("\nChange from adding nifa and tw to the nine effect modifiers")
    for method in ("rf", "lasso"):
        change = table.loc[1, f"{method}_ate"] - table.loc[0, f"{method}_ate"]
        print(f"  {method.upper()}  {change:+,.2f} dollars")
    print("The reference in each row is the published constant-slope coefficient.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = output_dir / f"issue_c5_impact_{timestamp}.csv"
    table.to_csv(destination, index=False)
    print(f"Saved {destination}")
    return table


if __name__ == "__main__":
    main()
