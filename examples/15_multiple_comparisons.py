# -*- coding: utf-8 -*-
"""Example 15 — Multiple comparisons: 12 segment cuts, 3 different answers.

Scenario
--------
The same 14-day revenue experiment, now cut 12 ways by market x device
(US/DE/BR/JP x mobile/desktop/tablet), 900 users per arm per cut. Reading the
p-values at alpha = 0.05 flags four winning cuts -- but with 12 tests the chance
of at least one false positive is 1 - 0.95^12 = 46%, so at least one is likely
noise. Bonferroni and Benjamini-Hochberg disagree about how many of the four
the analyst should believe.

Run: uv run python examples/15_multiple_comparisons.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from multiple_comparisons import benjamini_hochberg, bonferroni, naive_reject  # noqa: E402
from scipy import stats  # noqa: E402

markets, devices = ("US", "DE", "BR", "JP"), ("mobile", "desktop", "tablet")
cuts = [f"{m}-{d}" for d in devices for m in markets]
shift = {"US-mobile": 0.22, "BR-mobile": 0.26, "DE-desktop": 0.17}  # true lift, log scale

rng = np.random.default_rng(6)
p_values = []
for cut in cuts:
    a = rng.lognormal(2.2, 0.9, 900)
    b = rng.lognormal(2.2 + shift.get(cut, 0.0), 0.9, 900)
    p_values.append(stats.ttest_ind(b, a, equal_var=False).pvalue)
p_values = np.array(p_values)

naive = naive_reject(p_values)
_, bonf = bonferroni(p_values)
_, bh = benjamini_hochberg(p_values)

print("=== 15 - 12 segment cuts, 3 correction rules ===\n")
print(f"alpha = 0.05, m = {len(cuts)} cuts, Bonferroni threshold = {0.05 / len(cuts):.5f}\n")
print(f"  {'cut':<12} {'p':>10} {'naive':>7} {'Bonferroni':>11} {'BH (FDR)':>9}")
for cut, p, nv, bf, bh_ in zip(cuts, p_values, naive, bonf, bh, strict=True):
    if nv or bf or bh_:
        print(f"  {cut:<12} {p:>10.3e} {str(bool(nv)):>7} {str(bool(bf)):>11} {str(bool(bh_)):>9}")
print(f"\n  wins claimed: naive {naive.sum()}, Bonferroni {bonf.sum()}, BH {bh.sum()}")
print(f"  Bonferroni drops {[c for c, n, b in zip(cuts, naive, bonf, strict=True) if n and not b]}")
print(f"  BH also keeps    {[c for c, b, h in zip(cuts, bonf, bh, strict=True) if h and not b]}")
print()
print("VERDICT: report US-mobile (p=1.7e-09) and BR-mobile (p=9.7e-05) -- both survive")
print("         even Bonferroni, so they are safe to act on. DE-desktop (p=4.5e-03) is")
print("         real under FDR but not FWER: report it as a secondary finding needing")
print("         a confirmatory test. BR-desktop (p=2.4e-02) is the false positive to drop.")
