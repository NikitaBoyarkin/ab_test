# -*- coding: utf-8 -*-
"""Group-sequential testing with alpha-spending (O'Brien-Fleming & Pocock).

Peeking at a fixed-sample z-test inflates the Type I error: checking |Z| > 1.96
at every interim look can push false-positive rate well above 5%. Group-
sequential designs spend the alpha budget across K interim looks so the overall
crossing probability under the null stays at the nominal level.

Two classical boundary shapes:
  Pocock: constant boundary at every look (early stops are easy).
  OBF:    strict early, lenient late (conservative on early looks).

The test statistic Z_k follows a random walk with Var(Z_k) = information time
t_k = k/K. Boundaries b_k = c * shape(k) are calibrated by simulating the null
random walk and binary-searching the scalar c so overall crossing = alpha.

Reference: Lan & DeMets (1983); O'Brien & Fleming (1979); Pocock (1977).
"""

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def simulate_paths(n_paths, K, drift=0.0, seed=0):
    """Standardized z-statistic at each interim look, Var(Z_k) = 1.

    S_k is the cumulative score with Var(S_k) = information time t_k = k/K;
    the properly standardized z-stat is Z_k = S_k / sqrt(t_k), which has Var 1
    at every look. Under the alternative with full-look effect `drift` (the
    z the final look would see), E[Z_k] = drift * sqrt(t_k).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, K + 1) / K
    steps = rng.normal(0, np.sqrt(1 / K), size=(n_paths, K))
    S = np.cumsum(steps, axis=1)  # score, Var(S_k) = k/K = t_k
    Z = S / np.sqrt(t)  # standardized z, Var(Z_k) = 1
    if drift != 0.0:
        Z = Z + drift * np.sqrt(t)
    return Z


def crossing_rate(paths, bounds):
    """Fraction of paths that cross |Z_k| > b_k at any look."""
    return float(np.mean(np.any(np.abs(paths) > bounds, axis=1)))


def pocock_shape(K):
    return np.ones(K)


def obf_shape(K):
    k = np.arange(1, K + 1)
    return np.sqrt(K / k)  # b_k = c * sqrt(K/k): strict early, ~c at final look


def calibrate(shape_fn, K, alpha=0.05, n_paths=40000, seed=0):
    """Find scalar c so that crossing b_k = c*shape(k) under null = alpha."""
    base = shape_fn(K)
    paths = simulate_paths(n_paths, K, drift=0.0, seed=seed)
    lo, hi = 0.5, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if crossing_rate(paths, mid * base) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, base


def main():
    print("=== Group-Sequential Testing (alpha-spending) ===\n")
    K = 5
    alpha = 0.05
    n_paths = 40000

    c_pocock, b_pocock = calibrate(pocock_shape, K, alpha, n_paths, seed=1)
    c_obf, b_obf = calibrate(obf_shape, K, alpha, n_paths, seed=2)
    b_naive = np.full(K, stats.norm.ppf(1 - alpha / 2))  # 1.96 at every look

    print(f"[K={K} interim looks, target alpha={alpha}]")
    print(f"  {'design':<14} {'c':>7}  boundaries (z) at each look")
    print(f"  {'Pocock':<14} {c_pocock:>7.3f}  {[f'{v:.2f}' for v in c_pocock * b_pocock]}")
    print(f"  {'OBF':<14} {c_obf:>7.3f}  {[f'{v:.2f}' for v in c_obf * b_obf]}")
    print(f"  {'naive 1.96':<14} {1.96:>7.3f}  {[f'{v:.2f}' for v in b_naive]}")
    print()

    # Verify Type I control at the calibrated boundaries
    null_paths = simulate_paths(n_paths, K, drift=0.0, seed=99)
    print("[Type I error under null]")
    print(f"  {'design':<14} {'empirical alpha':>16}")
    print(f"  {'naive peeking':<14} {crossing_rate(null_paths, b_naive):>16.3f}  <- inflated")
    print(f"  {'Pocock':<14} {crossing_rate(null_paths, c_pocock * b_pocock):>16.3f}")
    print(f"  {'OBF':<14} {crossing_rate(null_paths, c_obf * b_obf):>16.3f}")
    print()

    # Power under a fixed alternative (full-info effect = 2.8 SD at final look)
    drift = 2.8
    alt_paths = simulate_paths(n_paths, K, drift=drift, seed=123)
    stop_look = {}
    for name, bounds in [
        ("Pocock", c_pocock * b_pocock),
        ("OBF", c_obf * b_obf),
        ("naive", b_naive),
    ]:
        crossed = np.abs(alt_paths) > bounds
        any_cross = crossed.any(axis=1)
        first = np.argmax(crossed, axis=1) + 1
        stop_look[name] = (
            float(any_cross.mean()),
            float(first[any_cross].mean()) if any_cross.any() else float("nan"),
        )
    print(f"[Power & avg stopping look, full-info effect = {drift}]")
    print(f"  {'design':<14} {'power':>8} {'avg stop look':>14}")
    for name, (pw, al) in stop_look.items():
        print(f"  {name:<14} {pw:>8.3f} {al:>14.2f}")
    print("  Early stopping: group-sequential often concludes before the final look.")


if __name__ == "__main__":
    main()
