# -*- coding: utf-8 -*-
"""Example 13 — Bootstrap CIs: when the normal-theory interval is useless.

Scenario
--------
A two-week checkout pilot ran on 1,000 users, 500 per arm. Revenue per user is
heavy-tailed: most users spend a little, a few spend a lot. The mean lift is
+$0.34 and the t-test CI is [-3.87, +4.55] -- too wide to support a decision.
The analyst needs an interval that survives the tail, and one for the median,
which normal theory cannot produce at all.

Run: uv run python examples/13_bootstrap_ci.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from bootstrap_ci import bootstrap_diff, t_test  # noqa: E402
from scipy import stats  # noqa: E402

rng = np.random.default_rng(13)
control = rng.lognormal(1.5, 1.3, 500)
treat = rng.lognormal(1.7, 1.3, 500)

t = t_test(control, treat)
pct = bootstrap_diff(control, treat, statistic=np.mean, n_boot=3000, method="percentile", seed=1)
bca = bootstrap_diff(control, treat, statistic=np.mean, n_boot=3000, method="bca", seed=1)
med = bootstrap_diff(control, treat, statistic=np.median, n_boot=3000, method="percentile", seed=2)

print("=== 13 - Bootstrap CIs on a heavy-tailed revenue metric ===\n")
skew = stats.skew(np.concatenate([control, treat]))
print(
    f"skewness {skew:.1f}; control revenue: median ${np.median(control):.2f}, "
    f"mean ${control.mean():.2f}\n"
)
for stat, method, r in [
    ("mean diff", "t-test (normal)", t),
    ("mean diff", "bootstrap percentile", pct),
    ("mean diff", "bootstrap BCa", bca),
    ("median diff", "bootstrap percentile", med),
]:
    ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
    print(f"  {stat:<12} {method:<21} {r['point']:+.3f}  {ci:>20}")

print(f"\n  t-test p = {t['p_value']:.3f}; percentile and BCa lower bounds differ by")
print(
    f"  ${abs(pct['ci_low'] - bca['ci_low']):.2f} -- the bootstrap distribution is skewed, so the two are"
)
print("  not interchangeable. The whale tail, not the typical user, drives the mean.")
print()
print("VERDICT: report the BCa mean interval and the percentile median interval,")
print("         not the normal-theory one: its symmetry assumption does not hold.")
print("         The median (+$1.07, CI excludes zero) is the number to act on.")
