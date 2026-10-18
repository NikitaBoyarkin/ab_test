# -*- coding: utf-8 -*-
"""Example 14 — Heterogeneous effects: the average that hides a loss.

Scenario
--------
A checkout redesign ran on 10,000 users and the pooled lift is +$0.03 per user
-- "no effect", so the launch is cancelled. But one layout shipped to every
device, and devices are not equally sized: mobile is half the traffic, tablet a
fifth. The analyst fits y ~ treat * segment before accepting the average,
because a win on mobile and a regression on tablet cancel out.

Run: uv run python examples/14_heterogeneous_treatment_effects.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from heterogeneous_treatment_effects import fit_het  # noqa: E402

# 14-day revenue per user, by device, exactly as the experiment logged it.
truth = {"mobile": (5000, 1.00), "desktop": (3000, 0.00), "tablet": (2000, -2.50)}
rng = np.random.default_rng(4)
frames = []
for segment, (n, lift) in truth.items():
    for treat in (0, 1):
        y = 12.0 + lift * treat + rng.normal(0, 3.0, n // 2)
        frames.append(pd.DataFrame({"segment": segment, "treat": treat, "y": y}))
df = pd.concat(frames, ignore_index=True)

model, ate = fit_het(df, ref_segment="desktop")
pooled = df[df.treat == 1].y.mean() - df[df.treat == 0].y.mean()

print("=== 14 - Does the pooled lift describe every device? ===\n")
print(f"pooled lift, all users: {pooled:+.3f} $/user   <- reads as 'nothing happened'\n")
print(f"  {'segment':<10} {'users':>7} {'lift $':>8} {'95% CI':>20}")
for segment, row in ate.iterrows():
    ci = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
    users = int((df.segment == segment).sum())
    marker = "  <- excludes 0" if not (row["ci_low"] <= 0 <= row["ci_high"]) else ""
    print(f"  {segment:<10} {users:>7} {row['ate']:>+8.3f} {ci:>20}{marker}")

print("\n  treatment x segment interactions vs reference 'desktop':")
for term in [t for t in model.params.index if "treat:segment" in t]:
    print(f"    {term:<26} {model.params[term]:+.3f}   p={model.pvalues[term]:.2g}")
print()
print("VERDICT: do NOT roll out to all users. Mobile gains +$1.06/user and tablet")
print("         loses $2.70/user, both with CIs excluding zero -- the tablet layout")
print("         is a real regression, and at 20% of traffic it cancels mobile's win.")
print("         Ship mobile only, hold tablet, re-test it after the layout fix.")
