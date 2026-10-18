# -*- coding: utf-8 -*-
"""Example 07 — Post-stratification: when the arms drift on a segment.

Scenario: an engagement experiment on a retail app, 8,000 users, outcome =
weekly active days (0-7). The pre-registered device mix is mobile 50 / desktop
32 / tablet 18, but the per-device bucketing key drifted: desktop -- the most
engaged segment, 4.6 active days against mobile's 3.1 -- landed 62% in
treatment instead of 50%. The raw read-out claims +0.40 days. Does that
survive the broken split?

Run: uv run python examples/07_post_stratification.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from post_stratification import naive_ate, stratified_ate  # noqa: E402

# segment -> (population share, pre-period mean, sd). The real effect is +0.25.
SEG = {"mobile": (0.50, 3.1, 1.2), "desktop": (0.32, 4.6, 1.0), "tablet": (0.18, 3.6, 1.2)}
DRIFT = {"mobile": 0.50, "desktop": 0.62, "tablet": 0.50}
EFFECT = 0.25

rng = np.random.default_rng(7)
n = 8_000
segments = rng.choice(list(SEG), size=n, p=[v[0] for v in SEG.values()])
arm = (rng.random(n) < [DRIFT[s] for s in segments]).astype(int)
mu = np.array([SEG[s][1] for s in segments]) + EFFECT * arm
days = np.clip(mu + np.array([SEG[s][2] for s in segments]) * rng.normal(size=n), 0, 7)

naive = naive_ate(arm, days)
strat = stratified_ate(arm, days, segments, weights={s: v[0] for s, v in SEG.items()})
var_cut = 1 - strat["se"] ** 2 / naive["se"] ** 2
overst = naive["diff"] / strat["ate"] - 1

print("=== 07 - Post-stratification: repairing a drifted split ===\n")
print(f"{'segment':<9}{'control':>9}{'treatment':>11}{'effect':>9}")
for s in SEG:
    c0 = ((segments == s) & (arm == 0)).sum()
    c1 = ((segments == s) & (arm == 1)).sum()
    print(
        f"{s:<9}{c0 / (arm == 0).sum():>8.1%}{c1 / (arm == 1).sum():>11.1%}"
        f"{strat['strata'][s]['diff']:>+9.3f}"
    )
p = "< 0.001" if strat["p_value"] < 0.001 else f"{strat['p_value']:.3f}"
print(f"\nraw difference : {naive['diff']:+.4f} days (SE {naive['se']:.4f})  <- not comparable")
print(f"weighted ATE   : {strat['ate']:+.4f} days (SE {strat['se']:.4f}, p {p})")
print(f"variance cut   : {var_cut:.1%}   truth {EFFECT:+.4f}   raw overstates {overst:.0%}")

print()
if overst > 0.1:
    print("VERDICT: report the weighted ATE, not the raw difference -- the raw number")
    print("         is inflated because treatment holds 37% desktop against control's")
    print("         26%, and the weighting also cuts variance 25%. Fix the bucketing")
    print("         key before the next run.")
else:
    print("VERDICT: the split drifted but the correction is small. Report the weighted")
    print("         ATE and fix the bucketing key before the next run.")
