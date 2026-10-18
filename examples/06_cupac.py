# -*- coding: utf-8 -*-
"""Example 06 — CUPAC: let a model use every pre-period column.

Scenario: the same checkout experiment as example 05 (600 users, conversion as
the metric). CUPED used the single covariate `pre_sessions` and cut variance
about 15%. The log carries a second pre-period column, `pre_conv`, that
predicts conversion too. Does an OLS over both columns buy enough extra
reduction to be worth one more moving part in the read-out?

Run: uv run python examples/06_cupac.py
"""

from _pathsetup import add_scripts_to_path

ROOT = add_scripts_to_path()

import cuped  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from cupac import cupac_test  # noqa: E402

df = pd.read_csv(ROOT / "data" / "experiment_sample.csv")
arm = df["treat"].to_numpy()
y = df["conv"].to_numpy(dtype=float)
x1 = df["pre_sessions"].to_numpy(dtype=float)
X = df[["pre_conv", "pre_sessions"]].to_numpy(dtype=float)

# Baseline to beat: CUPED on the one covariate used in example 05.
theta = cuped.cuped_theta(y, x1)
single = cuped.welch_mean_test(
    cuped.cuped_adjust(y[arm == 0], x1[arm == 0], theta),
    cuped.cuped_adjust(y[arm == 1], x1[arm == 1], theta),
)
# CUPAC: OLS outcome ~ every pre-period feature, fit on control, applied to both.
model = cupac_test(y[arm == 0], X[arm == 0], y[arm == 1], X[arm == 1])

n0, n1 = int((arm == 0).sum()), int((arm == 1).sum())
se_naive = np.sqrt(np.var(y[arm == 0], ddof=1) / n0 + np.var(y[arm == 1], ddof=1) / n1)
cut_single = 1 - single["se"] ** 2 / se_naive**2
cut_model = 1 - model["se"] ** 2 / se_naive**2

print("=== 06 - CUPAC: an OLS over every pre-period feature ===\n")
print(f"features : pre_conv, pre_sessions ({X.shape[1]} columns)")
print(f"model R^2: {model['r_squared']:.3f} (in-sample on the control arm)\n")
print(f"{'read-out':<22}{'SE':>9}{'variance cut':>14}")
print(f"{'naive':<22}{se_naive:>9.5f}{'--':>14}")
print(f"{'CUPED (1 covariate)':<22}{single['se']:>9.5f}{cut_single:>13.1%}")
print(f"{'CUPAC (2 features)':<22}{model['se']:>9.5f}{cut_model:>13.1%}")
print(f"\ndiff {model['diff']:+.4f}   p = {model['p_value']:.3f}")
print(f"extra cut vs CUPED: {cut_model - cut_single:+.1%}")

print()
if cut_model - cut_single > 0.02:
    print("VERDICT: the model's R^2 does buy more reduction than plain CUPED (+6pp")
    print("         here) from an untuned 2-feature OLS, so use CUPAC on all")
    print("         pre-period columns. Caveat: R^2 is in-sample -- a high-capacity")
    print("         model needs cross-fitted predictions or the adjustment is not honest.")
else:
    print("VERDICT: the extra columns add too little over one covariate; keep CUPED.")
