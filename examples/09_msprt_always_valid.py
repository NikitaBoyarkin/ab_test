# -*- coding: utf-8 -*-
"""Example 09 — mSPRT: an always-valid p-value the PM may read any morning.

Scenario
--------
A checkout test adds 500 users per arm per day and the PM wants the number
every morning, so the team looks 10 times. A z-test p-value is valid at one
pre-set sample size only: recomputed daily it turns noise into "significance".
The mixture SPRT gives an always-valid p, so stopping the moment it crosses
alpha keeps the false-positive rate at alpha for any stopping time. True effect
here is 0.04 SD per user.

Run: uv run python examples/09_msprt_always_valid.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from msprt_always_valid import always_valid_pvalue, naive_z_pvalue  # noqa: E402

ALPHA = 0.05
RHO = 0.05  # prior SD of the effect in observation-SD units -- set near the MDE
PER_DAY, DAYS = 500, 10  # users per arm per day; daily checkpoints
EFFECT = 0.04  # true per-user effect in SD units

rng = np.random.default_rng(11)
diff = rng.normal(EFFECT, 1.0, PER_DAY * DAYS)  # per-user treat-minus-control, unit variance
S = np.cumsum(diff)  # cumulative score
t = np.arange(1, PER_DAY * DAYS + 1)  # users per arm
checks = np.arange(PER_DAY, PER_DAY * DAYS + 1, PER_DAY)

p_av = always_valid_pvalue(S[checks - 1], t[checks - 1], RHO)
p_naive = naive_z_pvalue(S[checks - 1], t[checks - 1])
z = S[checks - 1] / np.sqrt(t[checks - 1])

print("=== 09 - always-valid p-value, read every morning ===\n")
print(f"true effect {EFFECT} SD, mixing prior rho={RHO}, monitored at {DAYS} daily checkpoints")
print(f"  {'day':>3} {'users/arm':>9} {'z':>7} {'naive p':>10} {'valid p':>10}")
for d, (n, zz, pn, pa) in enumerate(zip(checks, z, p_naive, p_av, strict=True), start=1):
    print(f"  {d:>3} {n:>9} {zz:>+7.2f} {pn:>10.4f} {pa:>10.4f}")

stop_naive = next((int(n) for n, p in zip(checks, p_naive, strict=True) if p <= ALPHA), None)
stop_av = next((int(n) for n, p in zip(checks, p_av, strict=True) if p <= ALPHA), None)
print(
    f"\nnaive z-test dips below {ALPHA} at {stop_naive} users/arm, but it is recomputed daily, so"
)
print("that threshold is not a 5% test.")

if stop_av is None:
    print("VERDICT: the always-valid p never crosses alpha within 10 days -- the naive calls were")
    print("         noise. Keep running rather than shipping on a peek.")
else:
    print(f"VERDICT: the always-valid p crosses alpha at {stop_av} users/arm, the earliest honest")
    print("         stopping point. Keep running to it, then report the lift and stop.")
