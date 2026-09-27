# -*- coding: utf-8 -*-
"""Multiple-testing correction for A/B analyses.

When you inspect several metrics or segments at once (the HTE use case), the
chance of at least one false positive grows with the number of tests. Two
standard controls:

  * Bonferroni: compare every p-value to alpha / m. Guarantees family-wise
    error (FWER) control but is conservative — it burns power as m grows.

  * Benjamini-Hochberg (BH / FDR): control the expected *proportion* of false
    discoveries among the rejected tests. Less conservative; the industry
    default when hunting for segments or secondary metrics.

This module implements both plus the naive no-correction baseline, and
calibrates the difference empirically on two designs: an all-null design,
where the naive family-wise error is ~1 - (1 - alpha)^m and Bonferroni holds
it at alpha, and a mixed design with true alternatives, which is the only one
that can show BH's guarantee -- the expected false-discovery proportion stays
at or below alpha while it still rejects more true alternatives than
Bonferroni.
"""

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def naive_reject(pvalues, alpha=0.05) -> np.ndarray:
    """Baseline: reject each test whose p-value < alpha (no correction)."""
    return np.asarray(pvalues) < alpha


def bonferroni(pvalues, alpha=0.05):
    """Bonferroni-corrected rejection + adjusted p-values (alpha / m each)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adjusted = np.minimum(p * m, 1.0)
    return adjusted, adjusted < alpha


def benjamini_hochberg(pvalues, alpha=0.05):
    """BH-corrected rejection + adjusted p-values (controls FDR at alpha)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest rank down
    adjusted = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(ranked[i], running)
        adjusted[order[i]] = running
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted, adjusted < alpha


def _fw_fdp(method_fn, pvalues, alpha, null_mask):
    """One draw: family-wise error (any false rejection) and FDP = V / max(R, 1).

    `null_mask` marks the truly null hypotheses, so false discoveries can be
    told apart from true ones. Under an all-null design V == R on every draw
    that rejects at all, so the FDP collapses to 1 -- which is why BH's FDR
    claim is invisible there and needs `_calibrate_mixed`.
    """
    _, rejected = method_fn(pvalues, alpha)
    false_hits = int(np.sum(rejected & null_mask))
    total = int(rejected.sum())
    fdp = false_hits / total if total else 0.0
    return (1.0 if false_hits else 0.0), fdp


_METHODS = {
    "naive": lambda p, a: (None, naive_reject(p, a)),
    "bonferroni": bonferroni,
    "bh": benjamini_hochberg,
}


def _calibrate(m, n_trials=2000, alpha=0.05, seed=0):
    """All-null design: FWER for naive / Bonferroni / BH over m tests.

    Every hypothesis is null here, so this design measures the family-wise
    rate only -- the FDP column is 1 by construction whenever anything rejects.
    """
    rng = np.random.default_rng(seed)
    null_mask = np.ones(m, dtype=bool)
    acc = {name: {"fwer": [], "fdp": []} for name in _METHODS}
    for _ in range(n_trials):
        p = rng.uniform(size=m)  # all nulls
        for name, fn in _METHODS.items():
            fw, fdp = _fw_fdp(fn, p, alpha, null_mask)
            acc[name]["fwer"].append(fw)
            acc[name]["fdp"].append(fdp)
    return {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in acc.items()}


def _calibrate_mixed(m=20, n_alt=5, effect=2.0, n_trials=2000, alpha=0.05, seed=0):
    """Mixed design: `n_alt` true alternatives and `m - n_alt` nulls.

    Returns FWER, the realised FDR (mean false-discovery proportion) and the
    power (share of true alternatives rejected) per method. This is the design
    that can show BH controlling FDR at alpha while Bonferroni, holding FWER
    at the same level, rejects fewer true alternatives.
    """
    rng = np.random.default_rng(seed)
    null_mask = np.zeros(m, dtype=bool)
    null_mask[n_alt:] = True
    acc = {name: {"fwer": [], "fdp": [], "power": []} for name in _METHODS}
    for _ in range(n_trials):
        z = rng.normal(0.0, 1.0, size=m)
        z[:n_alt] += effect
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        for name, fn in _METHODS.items():
            _, rejected = fn(p, alpha)
            false_hits = int(np.sum(rejected & null_mask))
            total = int(rejected.sum())
            acc[name]["fwer"].append(1.0 if false_hits else 0.0)
            acc[name]["fdp"].append(false_hits / total if total else 0.0)
            acc[name]["power"].append(int(np.sum(rejected[:n_alt])) / n_alt)
    return {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in acc.items()}


def main():
    print("=== Multiple-Testing Correction ===\n")

    # Worked example: 3 segment p-values from the HTE module
    p = np.array([0.001, 0.04, 0.20])
    print("[3 segment tests, p =", p, "]")
    _, rej = bonferroni(p)
    print(f"  naive reject:      {list(naive_reject(p))}")
    print(f"  Bonferroni reject: {list(rej)}   (alpha/m = {0.05 / 3:.4f})")
    _, rej = benjamini_hochberg(p)
    print(f"  BH reject:         {list(rej)}")

    print("\n[All-null calibration, m=10 tests, 2000 draws, alpha=0.05]")
    print(f"  {'method':<12} {'FWER (any false +)':>20} {'FDP (1 if any reject)':>24}")
    print(f"  {'naive':<12}  1-(1-a)^m = {1 - (0.95) ** 10:.3f} (theory)")
    for name, d in _calibrate(m=10).items():
        print(f"  {name:<12} {d['fwer']:>20.3f} {d['fdp']:>24.3f}")
    print("  all-null: every rejection is a false discovery, so FDP is 1 by")
    print("  construction -- this design can only check FWER.")

    print("\n[Mixed calibration, m=20 tests (5 true), 2000 draws, alpha=0.05]")
    print(f"  {'method':<12} {'FWER':>8} {'FDR (avg false share)':>24} {'power':>8}")
    for name, d in _calibrate_mixed().items():
        print(f"  {name:<12} {d['fwer']:>8.3f} {d['fdp']:>24.3f} {d['power']:>8.3f}")
    print("  naive rejection inflates FWER; Bonferroni caps FWER but burns")
    print("  power; BH holds FDR at alpha while rejecting more true alternatives.")


if __name__ == "__main__":
    main()
