# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals for A/B tests on arbitrary metrics.

The bootstrap makes no distributional assumption, so it works for skewed
metrics (revenue, clicks, session length) and arbitrary statistics (median,
quantile, conversion) where the normal-approximation t-test is misleading.

Two flavours:
  - percentile: take empirical quantiles of the bootstrap distribution.
  - BCa: bias-corrected and accelerated — better coverage when the bootstrap
    distribution is skewed (the default here for the difference of means).

For the difference of group statistics, resample each group independently and
take the difference of bootstrap replicates.
"""
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def bootstrap_diff(a, b, statistic=np.mean, n_boot=2000, alpha=0.05,
                   method="bca", seed=0) -> dict:
    """Bootstrap CI for statistic(b) - statistic(a).

    Parameters
    ----------
    a, b : array-like
        Per-unit observations in each arm.
    statistic : callable
        Function applied to a 1-D sample, e.g. np.mean, np.median.
    method : "percentile" | "bca"
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ba = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        diffs[i] = statistic(bb) - statistic(ba)
    point = statistic(b) - statistic(a)
    if method == "percentile":
        lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    else:  # BCa
        lo, hi = _bca_interval(diffs, a, b, statistic, point, alpha)
    se = float(np.std(diffs, ddof=1))
    return {"point": float(point), "se": se,
            "ci_low": float(lo), "ci_high": float(hi),
            "method": method, "n_boot": n_boot, "significant": not (lo <= 0 <= hi)}


def _bca_interval(boot, a, b, statistic, point, alpha):
    """Bias-corrected accelerated bootstrap interval for a difference."""
    # bias-correction z0 from proportion of boot replicates < point
    z0 = stats.norm.ppf((boot < point).mean())
    # acceleration via jackknife of the statistic over the combined influence
    # (use jackknife on each group's statistic, combine as difference)
    ja = np.array([statistic(np.delete(a, i)) for i in range(len(a))])
    jb = np.array([statistic(np.delete(b, i)) for i in range(len(b))])
    # influence of each obs on the difference; acceleration on pooled jackknife
    jk = np.concatenate([ja - ja.mean(), jb - jb.mean()])
    denom = np.sum(jk**3) / np.sum(jk**2) ** 1.5
    acc = denom
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - acc * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - acc * (z0 + z_hi)))
    return np.quantile(boot, [a1, a2])


def t_test(a, b) -> dict:
    res = stats.ttest_ind(b, a, equal_var=False)
    diff = np.mean(b) - np.mean(a)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return {"point": diff, "se": se,
            "ci_low": diff - stats.norm.ppf(0.975) * se,
            "ci_high": diff + stats.norm.ppf(0.975) * se,
            "p_value": float(res.pvalue)}


def main():
    print("=== Bootstrap Confidence Intervals ===\n")

    # skewed revenue: lognormal. t-test CI for the mean is OK-ish by CLT, but
    # the bootstrap matches it and also handles the MEDIAN (no normal theory).
    rng = np.random.default_rng(1)
    a = rng.lognormal(mean=2.0, sigma=1.0, size=1000)
    b = rng.lognormal(mean=2.1, sigma=1.0, size=1000)  # +0.1 on log scale

    print("[Skewed revenue (lognormal), +0.1 log-shift]")
    t = t_test(a, b)
    bm = bootstrap_diff(a, b, statistic=np.mean, n_boot=2000, seed=1)
    print(f"  {'statistic':<14} {'method':<10} {'estimate':>10} {'SE':>9} {'95% CI':>22}")
    print(f"  {'mean diff':<14} {'t-test':<10} {t['point']:>10.3f} {t['se']:>9.3f} "
          f"[{t['ci_low']:>8.3f}, {t['ci_high']:>8.3f}]")
    print(f"  {'mean diff':<14} {'boot-BCa':<10} {bm['point']:>10.3f} {bm['se']:>9.3f} "
          f"[{bm['ci_low']:>8.3f}, {bm['ci_high']:>8.3f}]")

    bd_med = bootstrap_diff(a, b, statistic=np.median, n_boot=2000, seed=2)
    bd_q90 = bootstrap_diff(a, b, statistic=lambda x: np.quantile(x, 0.9),
                            n_boot=2000, seed=3)
    print(f"  {'median diff':<14} {'boot-BCa':<10} {bd_med['point']:>10.3f} {bd_med['se']:>9.3f} "
          f"[{bd_med['ci_low']:>8.3f}, {bd_med['ci_high']:>8.3f}]")
    print(f"  {'p90 diff':<14} {'boot-BCa':<10} {bd_q90['point']:>10.3f} {bd_q90['se']:>9.3f} "
          f"[{bd_q90['ci_low']:>8.3f}, {bd_q90['ci_high']:>8.3f}]")
    print("  (t-test has no direct analogue for median/p90 — bootstrap does.)")
    print()

    # calibration: does the bootstrap mean-diff CI cover at 95% under the null?
    n_sims = 400
    cover = 0
    for s in range(n_sims):
        r = np.random.default_rng(10_000 + s)
        x = r.lognormal(2.0, 1.0, 500)
        y = r.lognormal(2.0, 1.0, 500)  # same distribution -> true diff = 0
        ci = bootstrap_diff(x, y, statistic=np.mean, n_boot=499, seed=s)
        cover += ci["ci_low"] <= 0 <= ci["ci_high"]
    print(f"[Coverage check, {n_sims} sims, true mean diff = 0]")
    print(f"  BCa 95% CI covers 0: {cover/n_sims*100:.1f}%  (expect ~95%)")


if __name__ == "__main__":
    main()
