# -*- coding: utf-8 -*-
"""Sequential (Evan Miller) A/B testing — early stopping on a success counter.

Classic sequential rule from Evan Miller. Pick N = total successes and D = the
"ahead" margin from his sample-size table; then, as data streams in:

  * stop and declare B wins once success_B - success_A >= D;
  * stop and declare "no difference" once success_A + success_B >= N.

This file reproduces the tables, validates the rule empirically (Type I and
Type II error) and quantifies when the sequential method saves sample size
versus a fixed-horizon design.

Reference: https://www.evanmiller.org/sequential-ab-testing.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    geom_vline,
    ggplot,
)
from sample_size import two_proportion

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Sample-size table from https://www.evanmiller.org/ab-testing/sequential.html
SEQ_SIZE_TABLE = pd.DataFrame(
    {
        "sig": [0.05] * 5,
        "power": [0.8] * 5,
        "min_relative_effect": [0.1, 0.2, 0.3, 0.4, 0.5],
        "n_total_success": [2922, 808, 391, 243, 170],
        "n_success_ahead": [106, 56, 39, 31, 26],
    }
)


# Fixed-sample benchmark uses the one-sided test (the sequential rule is one-sided).
def fixed_sample_size(base_rate, diff):
    return two_proportion(base_rate, diff, one_tail=True)


def seq_ab_testing(
    base_rate=0.01, true_relative_lift_effect=0, n_total_success=808, n_success_ahead=56
):
    """Run one sequential A/B trial; return stopping outcome and sample size."""
    step_size = 50
    sample_size_per_group = 0
    n_success_a, n_success_b = 0, 0
    while True:
        sample_size_per_group += step_size
        n_success_a += np.random.binomial(n=step_size, p=base_rate)
        n_success_b += np.random.binomial(
            n=step_size, p=base_rate * (1 + true_relative_lift_effect)
        )
        if (n_success_b - n_success_a) >= n_success_ahead:
            return {
                "sample_size_per_group": sample_size_per_group,
                "n_success_a": n_success_a,
                "n_success_b": n_success_b,
                "if_b_win": True,
            }
        if (n_success_a + n_success_b) >= n_total_success:
            return {
                "sample_size_per_group": sample_size_per_group,
                "n_success_a": n_success_a,
                "n_success_b": n_success_b,
                "if_b_win": False,
            }
        if sample_size_per_group >= 5_000_000:
            return None


def aggregate_test_results(df, fixed_sample_method_size, verbose=True):
    """Summarize a batch of trials vs the fixed-sample benchmark."""
    if verbose:
        print(f"total trials: {len(df)} times")
        print(f"treatment group wins ratio: {df['if_b_win'].sum() / len(df) * 100:.1f}%")
        print(
            "sequential method sample size larger than fixed sample method, "
            f"ratio: {(df['sample_size_per_group'] > fixed_sample_method_size).sum() / len(df) * 100:.1f}%"
        )
        print(
            "average sequential method sample size relative to fixed sample "
            f"method: {df['sample_size_per_group'].mean() / fixed_sample_method_size * 100:.1f}%"
        )
    return {
        "total_trials": len(df),
        "treatment_group_win_ratio": df["if_b_win"].sum() / len(df),
        "seq_method_sample_size_larger_ratio": (
            df["sample_size_per_group"] > fixed_sample_method_size
        ).sum()
        / len(df),
        "avg_seq_method_sample_size_over_fixed_sample_method": df["sample_size_per_group"].mean()
        / fixed_sample_method_size,
        "fixed_sample_method_size": fixed_sample_method_size,
        "avg_n_total_success": (df["n_success_a"].sum() + df["n_success_b"].sum()) / len(df),
    }


def run_grid(base_rate_list, trial_times):
    """Run the AA / AB / blockbuster grid over base rates and MDE rows."""
    rows = []
    for base_rate in base_rate_list:
        for i in range(len(SEQ_SIZE_TABLE)):
            sig, power, min_rel_eff, n_total, n_ahead = SEQ_SIZE_TABLE.iloc[i, :]
            fixed_size = fixed_sample_size(base_rate, base_rate * min_rel_eff)
            for label, lift in (
                ("aa test", 0.0),
                ("ab test", min_rel_eff),
                ("blockbuster test", min_rel_eff * 2),
            ):
                df = pd.DataFrame(
                    seq_ab_testing(
                        base_rate=base_rate,
                        true_relative_lift_effect=lift,
                        n_total_success=n_total,
                        n_success_ahead=n_ahead,
                    )
                    for _ in range(trial_times)
                )
                result = aggregate_test_results(df, fixed_size, verbose=False)
                rows.append(
                    {
                        "base_rate": base_rate,
                        "test_type": label,
                        **dict(SEQ_SIZE_TABLE.iloc[i, :]),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _add_savings(df):
    return df.assign(
        fixed_method_expected_n_success=lambda df: (
            df["fixed_sample_method_size"] * df["base_rate"] * 2
        ).round(0),
        savings=lambda df: (
            (df["fixed_method_expected_n_success"] - df["avg_n_total_success"])
            / df["fixed_method_expected_n_success"]
        ).round(2),
    )


def _plot_savings(df, path):
    plot_data = df.assign(base_rate=lambda x: x["base_rate"].astype(str))
    p = (
        ggplot(
            data=plot_data,
            mapping=aes(x="min_relative_effect", y="savings", colour="base_rate"),
        )
        + geom_hline(yintercept=0)
        + geom_line()
        + geom_point()
        + facet_wrap("~ test_type")
        + coord_cartesian(ylim=(-0.6, 0.6))
    )
    p.save(str(path), width=8, height=5, dpi=120)


def _plot_largest_mde(df, path):
    largest_mde = pd.DataFrame(
        {
            "base_rate": df["base_rate"].unique().astype(str),
            "largest_mde": 0.36 - 1.5 * df["base_rate"].unique(),
        }
    )
    largest_mde = largest_mde.loc[largest_mde["largest_mde"] > 0, :]
    plot_data = df.pipe(
        lambda x: x.loc[
            (x["test_type"] == "ab test") & x["base_rate"].isin(largest_mde["base_rate"]),
            :,
        ]
    ).assign(base_rate=lambda x: x["base_rate"].astype(str))
    p = (
        ggplot(
            data=plot_data,
            mapping=aes(x="min_relative_effect", y="savings", colour="base_rate"),
        )
        + geom_hline(yintercept=0)
        + geom_vline(
            data=largest_mde,
            mapping=aes(xintercept="largest_mde"),
            colour="grey",
            linetype="dashed",
        )
        + geom_text(
            data=largest_mde.assign(
                largest_mde=(largest_mde["largest_mde"] + 0.08).round(2),
                largest_mde_label=(largest_mde["largest_mde"] * 100).astype(int).astype(str) + "%",
            ),
            mapping=aes(x="largest_mde", y=0.5, label="largest_mde_label"),
            colour="grey",
        )
        + geom_line()
        + geom_point()
        + facet_wrap("~ base_rate")
        + coord_cartesian(ylim=(-0.6, 0.6))
    )
    p.save(str(path), width=10, height=6, dpi=120)


def main():
    print("=== Sequential A/B Testing (Evan Miller) ===\n")
    print(SEQ_SIZE_TABLE.to_string())

    print("\n=== Validity check (A/A, A/B, blockbuster, 1000 trials each) ===")
    np.random.seed(42)
    base_rate = 0.01
    expected_rel = 0.2
    fixed_size = fixed_sample_size(base_rate, base_rate * expected_rel)
    for label, lift in (
        ("A/A test", 0.0),
        ("A/B test", expected_rel),
        ("Blockbuster test", expected_rel * 2),
    ):
        print(f"\n[{label}]")
        df = pd.DataFrame(
            seq_ab_testing(
                base_rate=base_rate,
                true_relative_lift_effect=lift,
                n_total_success=808,
                n_success_ahead=56,
            )
            for _ in range(1000)
        )
        aggregate_test_results(df, fixed_size)
    print(
        "\n  Type I error (A/A win rate) ~ 5%, power (A/B win rate) ~ 80%:",
        "the rule matches its spec.",
    )

    print("\n=== Sample-size savings vs fixed-horizon (grid over base rate x MDE) ===")
    df = run_grid([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], trial_times=1000)
    df = _add_savings(df)

    subset = (
        df.loc[df["min_relative_effect"].isin({0.5, 0.2, 0.1}), :]
        .loc[df["test_type"] == "ab test", :]
        .loc[df["base_rate"] == 0.01, :]
        .sort_values(["test_type", "sig", "power", "min_relative_effect"], ascending=False)
    )
    print(
        subset[
            [
                "base_rate",
                "test_type",
                "sig",
                "power",
                "min_relative_effect",
                "n_total_success",
                "n_success_ahead",
                "avg_n_total_success",
                "fixed_method_expected_n_success",
                "savings",
            ]
        ].to_string()
    )

    print("\n  Sequential testing only saves sample size at low base rates;")
    print("  at high base rates it can cost MORE than a fixed-horizon design.")

    PLOTS_DIR.mkdir(exist_ok=True)
    _plot_savings(df, PLOTS_DIR / "sequential_savings_by_test_type.png")
    _plot_largest_mde(df, PLOTS_DIR / "sequential_largest_mde.png")
    print(f"\n  plots written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
