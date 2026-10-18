# -*- coding: utf-8 -*-
"""Example 10 — Always-valid monitoring for a RATIO metric (CTR).

Scenario
--------
An ads test is meant to lift CTR and the PM wants to know after every batch of
traffic whether the lift is real. CTR is a ratio (clicks / impressions), so the
plain z-test SE is wrong -- the delta method fixes the SE -- and continuous
monitoring needs an always-valid p on top. Ten batches of 200 users per group
arrive; base CTR is 5% and the true lift is +10% relative. Can the lift be
called while the test is still running?

Run: uv run python examples/10_sequential_ratio.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sequential_ratio import gen_batch, msprt_ratio, ratio_interim  # noqa: E402

ALPHA = 0.05
BASE_RATE = 0.05  # control CTR
REL_LIFT = 0.10  # true +10% relative lift
TAU = 0.002  # prior SD of the lift in CTR-difference units -- set near the MDE
BATCH, N_BATCHES, SEED = 200, 10, 3

users, lifts, p_naive_hist, p_av_hist = [], [], [], []
xa = ya = xb = yb = np.array([], dtype=float)

print("=== 10 - always-valid CTR monitoring while the test runs ===\n")
print(
    f"  {'batch':>5} {'users/grp':>9} {'CTR A':>7} {'CTR B':>7} {'lift pp':>8} {'z':>6} {'naive p':>9} {'valid p':>9}"
)
for k in range(1, N_BATCHES + 1):
    (xa_k, ya_k), (xb_k, yb_k) = gen_batch(BATCH, BASE_RATE, REL_LIFT, seed=SEED * 1000 + k)
    xa, ya = np.concatenate([xa, xa_k]), np.concatenate([ya, ya_k])
    xb, yb = np.concatenate([xb, xb_k]), np.concatenate([yb, yb_k])
    r = ratio_interim(xa, ya, xb, yb)
    _, p_av = msprt_ratio(r["D"], r["z"], r["I"], TAU)
    p_naive = 2 * (1 - stats.norm.cdf(abs(r["z"])))
    users, lifts = users + [len(xa)], lifts + [r["D"]]
    p_naive_hist, p_av_hist = p_naive_hist + [p_naive], p_av_hist + [p_av]
    print(
        f"  {k:>5} {len(xa):>9} {r['R_a']:>7.4f} {r['R_b']:>7.4f} {r['D'] * 100:>+8.3f} {r['z']:>+6.2f} {p_naive:>9.4g} {p_av:>9.4g}"
    )

first_naive = next((n for n, p in zip(users, p_naive_hist, strict=True) if p <= ALPHA), None)
first_av = next((n for n, p in zip(users, p_av_hist, strict=True) if p <= ALPHA), None)
call_lift = next(
    (d * 100 for _, d, p in zip(users, lifts, p_av_hist, strict=True) if p <= ALPHA), None
)
print(
    f"\nfirst naive delta-z p<=0.05 at {first_naive} users/arm; first always-valid p<=0.05 at {first_av}."
)

if first_av is None:
    print(
        f"VERDICT: do not call the lift at {first_naive} users/arm -- that delta-z p is a peek. The"
    )
    print("         always-valid p never crosses alpha here, so keep the test running.")
else:
    print(
        f"VERDICT: do not call the lift at {first_naive} users/arm -- that delta-z p is a peek, and"
    )
    print(
        f"         the always-valid p is still far above alpha there. From {first_av} users/arm it"
    )
    print(
        f"         holds (lift {call_lift:+.2f}pp), so stop the test there and report the CTR gain."
    )
