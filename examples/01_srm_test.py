# -*- coding: utf-8 -*-
"""Example 01 — Sample Ratio Mismatch: the check that runs before any metric.

Scenario
--------
A two-arm test ran for a week and logged ~1,000,000 users. The assignment
service had a rounding bug, so every user whose hash landed in the top 0.5%
band was routed to treatment. The metric still looks reasonable -- which is
exactly why nobody notices. Three lines catch it.

Run: uv run python examples/01_srm_test.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

from srm_test import srm_test  # noqa: E402

# What the assignment service actually logged, not what it intended.
observed = [503_200, 509_140]

result = srm_test(observed, [0.5, 0.5])

ratio_a, ratio_b = result["observed_ratio"]
print("=== 01 - SRM check before reading any metric ===\n")
print(f"logged split : {ratio_a:.4%} control / {ratio_b:.4%} treatment")
print(f"chi-square   : {result['chi2']:.1f}   p = {result['p_value']:.3g}")
print(f"threshold    : flagged when p < {result['alpha']}\n")

# The point of the example: a 0.5% traffic leak is invisible in the split
# percentages and fatal to every number downstream.
if result["srm_detected"]:
    print("VERDICT: SRM. Do not read the treatment effect -- the arms are not")
    print("         comparable. Fix bucketing/logging and re-run the experiment.")
else:
    print("VERDICT: split is clean. Proceed to the metric test.")
