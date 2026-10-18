# -*- coding: utf-8 -*-
"""Example 12 — Bayesian A/B: price the risk instead of reading a p-value.

Scenario
--------
A pricing-page test closed with 2,000 sessions per arm: control converted 120
(6.00%), treatment 150 (7.50%). The frequentist read is two-sided p = 0.059 --
just past the 0.05 gate, so the default call is "not significant, do not ship".
The analyst asks what the p-value cannot answer: if we ship and the lift is not
real, how much conversion do we lose? The Beta-Binomial posterior prices that.

Run: uv run python examples/12_bayesian_ab_test.py
"""

import math

from _pathsetup import add_scripts_to_path
from scipy import stats

add_scripts_to_path()

from bayesian_ab_test import beta_binomial, decide  # noqa: E402

ROPE = 0.005  # 0.5pp: below this the two arms are practically the same
N, CONV_A, CONV_B = 2_000, 120, 150

post_a, post_b = beta_binomial(
    s=(CONV_A, CONV_B), f=(N - CONV_A, N - CONV_B), prior=(1.0, 1.0), seed=1
)
d = decide(post_a, post_b, rope=ROPE)

# Frequentist benchmark, computed here so the two reads sit side by side.
p_pool = (CONV_A + CONV_B) / (2 * N)
z = (CONV_B - CONV_A) / N / math.sqrt(p_pool * (1 - p_pool) * 2 / N)
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print("=== 12 - Bayesian decision on a conversion metric ===\n")
print(f"control {CONV_A}/{N} = {CONV_A / N:.2%}   treatment {CONV_B}/{N} = {CONV_B / N:.2%}")
print(
    f"\nfrequentist: z = {z:.2f}, two-sided p = {p_value:.3f} -> "
    f"{'reject H0' if p_value < 0.05 else 'NOT significant, no ship'}"
)
print(
    f"posterior  : P(B > A) = {d['p_b_better']:.3f}, mean lift = {d['lift_mean']:+.4f} "
    f"({d['lift_mean'] * 2 * N:.0f} extra conv per {2 * N:,} sessions)"
)
print(
    f"             95% CrI = [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], "
    f"P(|lift| <= {ROPE}) = {d['p_equiv_within_rope']:.3f} -> not equivalent"
)
print(
    f"\nexpected loss, ship B : {d['expected_loss_choose_b']:.5f} rate "
    f"(~{d['expected_loss_choose_b'] * 200_000:.0f} conv lost per 200k users)"
)
print(
    f"expected loss, keep A : {d['expected_loss_choose_a']:.5f} rate "
    f"(~{d['expected_loss_choose_a'] * 200_000:,.0f} conv lost per 200k users)"
)
print(f"\nmodule decision: {d['decision']}")
print("\nVERDICT: ship B. The p-value stops at 0.059 and would have blocked a change")
print("         whose downside is ~0.01pp of conversion, while the posterior puts")
print("         P(B > A) at 0.97. A p-value answers 'could this be noise?'; expected")
print("         loss answers 'what does being wrong cost?'. Ship on the second number.")
