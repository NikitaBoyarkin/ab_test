# -*- coding: utf-8 -*-
"""Example 03 — CTR is a ratio, and the naive per-user t-test gets it wrong.

Scenario
--------
A three-day ad test logged 600 users at ~40 impressions each, but per-user
impressions range from 4 to 181. CTR = clicks / impressions is a ratio of two
random variables, so the per-user ratio r_i = clicks_i / impressions_i is not
what the business tracks: it counts a user with 4 impressions the same as one
with 180. The analyst runs both tests on the same 600 rows and must decide
which number goes in the readout.

Run: uv run python examples/03_delta_method_ratio.py
"""

from _pathsetup import DATA, add_scripts_to_path

add_scripts_to_path()

import delta_method_ratio as dmr  # noqa: E402
import pandas as pd  # noqa: E402

df = pd.read_csv(DATA / "experiment_sample.csv")
control, treat = df[df["treat"] == 0], df[df["treat"] == 1]
x_a, y_a = control["impressions"].to_numpy(), control["clicks"].to_numpy()
x_b, y_b = treat["impressions"].to_numpy(), treat["clicks"].to_numpy()

delta = dmr.delta_method_test(x_a, y_a, x_b, y_b)
naive = dmr.naive_per_unit_ratio_test(x_a, y_a, x_b, y_b)
se_a, se_b = dmr.ratio_delta_se(x_a, y_a), dmr.ratio_delta_se(x_b, y_b)

print("=== 03 - CTR: naive per-user ratio vs the delta method ===\n")
print(f"users: control {len(x_a)}, treatment {len(x_b)}; ~{x_a.mean():.0f} impressions each")
print(f"per-arm delta SE: {se_a:.5f}, {se_b:.5f}  (combined {delta['se']:.5f})\n")
print("ctrl/treat below are the aggregate CTR for delta, the mean per-user ratio for naive")
print(f"  {'method':<17}{'ctrl':>9}{'treat':>9}{'diff':>10}{'SE':>10}{'p':>9}")
print(
    f"  {'delta (ratio)':<17}{delta['ratio_a']:>9.4%}{delta['ratio_b']:>9.4%}"
    f"{delta['diff']:>10.5f}{delta['se']:>10.5f}{delta['p_value']:>9.4f}"
)
print(
    f"  {'naive (per-user)':<17}{naive['ratio_a']:>9.4%}{naive['ratio_b']:>9.4%}"
    f"{naive['diff']:>10.5f}{naive['se']:>10.5f}{naive['p_value']:>9.4f}"
)
print(f"\ndelta 95% CI for the CTR gap: [{delta['ci_low']:.5f}, {delta['ci_high']:.5f}]")
print(f"naive calls it p = {naive['p_value']:.3f}; delta calls it p = {delta['p_value']:.3f}.\n")

if delta["significant"]:
    print("VERDICT: ship. The CTR gap survives the correct (delta-method) SE.")
else:
    print(f"VERDICT: no decision. The naive p = {naive['p_value']:.3f} is an artifact:")
    print("         low-impression users (a 4-impression ratio is 0, 0.25 or 0.5) carry")
    print(
        f"         the same weight as users with 180. The aggregate CTR gap is {delta['diff']:.4f}"
    )
    print(f"         with p = {delta['p_value']:.3f} and a CI that crosses zero. Do not ship.")
