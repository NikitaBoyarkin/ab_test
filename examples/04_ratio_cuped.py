# -*- coding: utf-8 -*-
"""Example 04 — CUPED on that same CTR ratio: free data buys back sample.

Scenario
--------
The ad team ran a five-week CTR test on 2,500 users per arm. Every user's
pre-period impressions and clicks were logged before the test started, so they
cost nothing to use. The plain delta-method readout is inconclusive: +0.29pp on
a 5.7% CTR, p = 0.056. Linearize the ratio (Z = Y - R*X), apply CUPED to Z with
the pre-period Z, and scale back by 1/mean(X). The shared fixture cannot serve
here -- it carries pre_conv and pre_sessions, not pre-period impressions -- so
the module's simulator supplies the pre/post pairs.

Run: uv run python examples/04_ratio_cuped.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import delta_method_ratio as dmr  # noqa: E402
import ratio_cuped as rc  # noqa: E402

N_PER_ARM = 2_500
x_pre_a, y_pre_a, x_a, y_a = rc.simulate_ratio_cuped(n=N_PER_ARM, seed=181, rel_lift=0.0)
x_pre_b, y_pre_b, x_b, y_b = rc.simulate_ratio_cuped(n=N_PER_ARM, seed=669, rel_lift=0.06)

delta = dmr.delta_method_test(x_a, y_a, x_b, y_b)
cuped = rc.ratio_cuped_test(x_a, y_a, x_b, y_b, x_pre_a, y_pre_a, x_pre_b, y_pre_b)
reduction = 1 - cuped["se"] ** 2 / delta["se"] ** 2
wks = 5 * (1 - reduction)

print("=== 04 - CUPED on the ratio metric (CTR) ===\n")
print(f"users/arm {N_PER_ARM:,}; pre-period impressions/user {x_pre_a.mean():.0f}")
print(f"  {'method':<14}{'ctrl':>9}{'treat':>9}{'diff':>10}{'SE':>10}{'p':>9}")
print(
    f"  {'delta only':<14}{delta['ratio_a']:>9.4%}{delta['ratio_b']:>9.4%}"
    f"{delta['diff']:>10.5f}{delta['se']:>10.5f}{delta['p_value']:>9.4f}"
)
print(
    f"  {'CUPED+delta':<14}{cuped['ratio_a']:>9.4%}{cuped['ratio_b']:>9.4%}"
    f"{cuped['diff']:>10.5f}{cuped['se']:>10.5f}{cuped['p_value']:>9.4f}"
)
print(f"\ntheta {cuped['theta']:.2f}, pre-period R^2 {cuped['r_squared']:.2f}")
print(f"SE {delta['se']:.5f} -> {cuped['se']:.5f}: {reduction:.1%} variance reduction")
print(f"same power on {1 - reduction:.0%} of the sample: 5 weeks -> {wks:.1f} weeks\n")

if cuped["significant"] and not delta["significant"]:
    print("VERDICT: use CUPED+delta. The pre-period columns were already in the")
    print(
        f"         warehouse, and they cut variance {reduction:.0%}: p = {delta['p_value']:.3f} becomes"
    )
    print(
        f"         p = {cuped['p_value']:.3f} on the same users. Declare CUPED before unblinding;"
    )
    print("         picking it after seeing p is p-hacking.")
else:
    print("VERDICT: report CUPED+delta. The pre-period columns were already logged,")
    print(
        f"         and they cut variance {reduction:.0%}, worth {1 / (1 - reduction):.2f}x the traffic."
    )
    print("         Declare CUPED before unblinding; picking it after seeing p is p-hacking.")
