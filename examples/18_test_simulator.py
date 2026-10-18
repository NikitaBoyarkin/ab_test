# -*- coding: utf-8 -*-
"""Example 18 — Calibrate the test before you trust it: Type I error and power.

Scenario
--------
Revenue per user is heavy-tailed, and the analyst has only 200 users per arm
for a checkout test. Before reading the result, simulate the whole pipeline on
a plug-in data-generating process: is the Welch t-test calibrated under the
null, and does it have power at the lift the team hopes for (+10% to +30%)?
Then swap in a bootstrap test to see whether a different test fixes it.

Run: uv run python examples/18_test_simulator.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from test_simulator import (  # noqa: E402
    calibration_curve,
    simulate,
    test_bootstrap_mean,
    test_welch,
)


def dgp_revenue(effect, n=200, sigma=1.5, seed=0):
    """Revenue per arm: lognormal (heavy tail), effect is multiplicative."""
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0, sigma, n)
    b = rng.lognormal(np.log1p(effect), sigma, n)
    return a, b


print("=== 18 - Can this test be trusted at n=200? ===\n")
print("empirical Type I error (true effect = 0, nominal alpha = 0.05):")
for name, test in (("Welch t-test", test_welch), ("bootstrap on mean", test_bootstrap_mean)):
    null = simulate(dgp_revenue, test, effect=0.0, n_trials=1000, seed=1)
    print(f"  {name:<20} reject rate = {null['rejection_rate']:.3f}  ({null['label']})")

print("\npower curve (Welch t-test, alpha = 0.05, n = 200 per arm):")
curve = calibration_curve(dgp_revenue, test_welch, [0.1, 0.2, 0.3, 0.5], n_trials=500, seed=10)
for row in curve:
    print(f"  true lift {row['effect']:+.0%}  ->  power {row['rejection_rate']:.3f}")

print("\nVERDICT: Welch is calibrated here (4.6% vs 5% nominal) and the bootstrap is")
print("         liberal (7.7%), so swapping tests is not the fix. Even well")
print("         calibrated, at n=200 a +30% lift is caught only ~18% of the time.")
print("         The arithmetic is fine -- the design is what breaks the estimate.")
print("         Do not read a null result as proof of no effect; raise n ~5x or")
print("         move to a lower-variance estimand before shipping this test.")
