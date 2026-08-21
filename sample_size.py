# -*- coding: utf-8 -*-
"""Sample-size & power helpers for fixed-horizon A/B tests.

Used to size a test before launch, and as the fixed-sample benchmark that
sequential designs are compared against (see `sequential_ab_testing.py` and
`group_sequential.py`, which both reference these numbers).

Reference (two-proportion formula):
    https://stats.stackexchange.com/questions/178568/calculate-sample-size-based-on-conversion-rate-minimum-detectable-effect-stati
"""

import math

import numpy as np
from scipy import stats


def two_proportion(base_rate, absolute_diff, alpha=0.05, beta=0.2, one_tail=True):
    """Per-arm sample size for a two-proportion test at a fixed horizon.

    Parameters
    ----------
    base_rate : float
        Conversion rate of the control group.
    absolute_diff : float
        Minimum detectable absolute effect, e.g. 0.02 for a 2pp lift.
    alpha : float
        Significance level.
    beta : float
        Type II error (power = 1 - beta).
    one_tail : bool
        Evan Miller's sequential rule is one-sided, so the fixed-sample
        benchmark defaults to one-tailed to be comparable.
    """
    p0 = base_rate
    p1 = base_rate + absolute_diff
    z_alpha = stats.norm.ppf(1 - alpha / (1.0 if one_tail else 2.0))
    z_beta = stats.norm.ppf(1 - beta)
    n = (z_alpha + z_beta) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / absolute_diff**2
    return math.ceil(n)


def two_mean(effect_size, sd, alpha=0.05, beta=0.2, two_sided=True):
    """Per-arm sample size for a two-sample test of means (normal approx)."""
    z_alpha = stats.norm.ppf(1 - alpha / (2.0 if two_sided else 1.0))
    z_beta = stats.norm.ppf(1 - beta)
    n = 2 * sd**2 * (z_alpha + z_beta) ** 2 / effect_size**2
    return math.ceil(n)


def power_two_proportion(base_rate, absolute_diff, n, alpha=0.05, one_tail=True):
    """Power for a fixed sample of n per arm given a true absolute lift."""
    p0 = base_rate
    p1 = base_rate + absolute_diff
    se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
    crit = stats.norm.ppf(1 - alpha / (1.0 if one_tail else 2.0))
    return float(stats.norm.cdf((absolute_diff - crit * se) / se))


if __name__ == "__main__":
    print("=== Sample Size & Power ===\n")
    for diff in (0.01, 0.02, 0.05):
        n = two_proportion(base_rate=0.10, absolute_diff=diff)
        print(
            f"  base=10%, MDE={diff * 100:.0f}pp: n/arm = {n:>6}"
            f"  (power@n={power_two_proportion(0.10, diff, n) * 100:.0f}%)"
        )
    print()
    for effect in (0.1, 0.2, 0.5):
        n = two_mean(effect_size=effect, sd=1.0)
        print(f"  mean test, effect={effect:.1f} sd: n/arm = {n}")
