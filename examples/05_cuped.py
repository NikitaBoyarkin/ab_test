# -*- coding: utf-8 -*-
"""Example 05 — CUPED: paying for sensitivity with the pre-period.

Scenario: the checkout experiment ran its planned week -- 600 users, 292
control / 308 treatment, conversion 15.8% vs 18.5%, a raw +2.8pp lift at
p=0.37. Nothing is significant, but `pre_sessions` was logged for every user
*before* assignment. Extend the test for another week, or read what is already
there more precisely?

Run: uv run python examples/05_cuped.py
"""

from _pathsetup import add_scripts_to_path

ROOT = add_scripts_to_path()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cuped import cuped_adjust, cuped_theta, welch_mean_test  # noqa: E402

df = pd.read_csv(ROOT / "data" / "experiment_sample.csv")
arm = df["treat"].to_numpy()
y = df["conv"].to_numpy(dtype=float)
x = df["pre_sessions"].to_numpy(dtype=float)

# theta comes from the pooled sample: x is pre-period, so pooling cannot leak
# the treatment effect into the adjustment.
theta = cuped_theta(y, x)
naive = welch_mean_test(y[arm == 0], y[arm == 1])
cuped = welch_mean_test(
    cuped_adjust(y[arm == 0], x[arm == 0], theta),
    cuped_adjust(y[arm == 1], x[arm == 1], theta),
)
corr = float(np.corrcoef(x, y)[0, 1])
var_cut = 1 - cuped["se"] ** 2 / naive["se"] ** 2
se_cut = 1 - cuped["se"] / naive["se"]

print("=== 05 - CUPED: one pre-period covariate, less variance ===\n")
print(f"users: {len(df)} ({int((arm == 0).sum())} control / {int(arm.sum())} treatment)")
print(f"covariate: pre_sessions, corr(X, Y) = {corr:.3f} -> theory cut {corr**2:.1%}")
print(f"theta = {theta:.5f}\n")
print(f"{'read-out':<10}{'diff':>9}{'SE':>9}{'p_value':>10}")
print(f"{'naive':<10}{naive['point']:>+9.4f}{naive['se']:>9.5f}{naive['p_value']:>10.3f}")
print(f"{'CUPED':<10}{cuped['point']:>+9.4f}{cuped['se']:>9.5f}{cuped['p_value']:>10.3f}")
print(f"\nvariance cut : {var_cut:.1%}   SE cut : {se_cut:.1%}")
print(f"worth        : {var_cut / (1 - var_cut):.0%} more traffic for the same power")

print()
if var_cut > 0.10:
    print("VERDICT: the pre-period variable is correlated enough to pay off -- CUPED")
    print("         cuts variance ~15% for free, so adopt it as the default read-out.")
    print("         The lift is still not significant, so extend the test, do not ship.")
else:
    print("VERDICT: corr(X, Y) is too weak to pay off. Skip CUPED and buy the")
    print("         sensitivity the expensive way, with more traffic.")
