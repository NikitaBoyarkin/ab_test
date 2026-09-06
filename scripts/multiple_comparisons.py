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
calibrates the difference empirically: under the all-null setting the naive
family-wise error is ~1 - (1 - alpha)^m, Bonferroni holds it at alpha, and BH
keeps the false-discovery proportion at alpha.
"""

import warnings

import numpy as np

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


def _fw_fdp(method_fn, pvalues, alpha):
    """One draw: family-wise error (any reject) and false-discovery proportion."""
    _, rejected = method_fn(pvalues, alpha)
    any_reject = bool(rejected.any())
    if not any_reject:
        return 0.0, 0.0  # FDP undefined -> 0
    # all-null setup: every rejection is a false discovery
    return 1.0, rejected.sum() / rejected.sum()


def _calibrate(m, n_trials=2000, alpha=0.05, seed=0):
    """Under all-null, FWER and FDR for naive / Bonferroni / BH over m tests."""
    rng = np.random.default_rng(seed)
    stats = {name: {"fwer": [], "fdp": []} for name in ("naive", "bonferroni", "bh")}
    methods = {
        "naive": lambda p, a: (None, naive_reject(p, a)),
        "bonferroni": bonferroni,
        "bh": benjamini_hochberg,
    }
    for _ in range(n_trials):
        p = rng.uniform(size=m)  # all nulls
        for name, fn in methods.items():
            fw, fdp = _fw_fdp(fn, p, alpha)
            stats[name]["fwer"].append(fw)
            stats[name]["fdp"].append(fdp)
    return {name: {k: float(np.mean(v)) for k, v in d.items()} for name, d in stats.items()}


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
    print(f"  {'method':<12} {'FWER (any false +)':>20} {'FDR (avg false share)':>24}")
    print(f"  {'naive':<12}  1-(1-a)^m = {1 - (0.95) ** 10:.3f} (theory)")
    for name, d in _calibrate(m=10).items():
        print(f"  {name:<12} {d['fwer']:>20.3f} {d['fdp']:>24.3f}")
    print("  naive inflates family-wise error; Bonferroni caps it at alpha;")
    print("  BH keeps the false-discovery proportion at alpha with more power.")


if __name__ == "__main__":
    main()
