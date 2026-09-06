# -*- coding: utf-8 -*-
"""Delta method for ratio metrics.

Ratio metrics (CTR = clicks/impressions, revenue per session, RPC) are ratios
of two random variables. A naive t-test on the *per-unit ratio* is biased when
the denominator varies across units, and a naive t-test on the aggregate ratio
ignores the variance of the denominator. The delta method gives the correct
asymptotic SE for R = sum(Y) / sum(X).

For two independent groups, the SE of the difference R_B - R_A is
sqrt(Var(R_A) + Var(R_B)), where per group:

    Var(R) = (1/mean(X)^2) * [Var(Y) + R^2*Var(X) - 2*R*Cov(X,Y)] / n

Reference: Deng, Knoblich, Lu (2018), "Applying the Delta Method in Metrics
Experiments".
"""

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

CONF_Z = stats.norm.ppf(0.975)


def ratio_delta_se(x: np.ndarray, y: np.ndarray) -> float:
    """Asymptotic SE of R = sum(y)/sum(x) = mean(y)/mean(x) via delta method.

    x, y are per-unit (e.g. per-user) observations of denominator and numerator.
    """
    n = len(x)
    mean_x, mean_y = x.mean(), y.mean()
    ratio = mean_y / mean_x
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    var_ratio = (var_y + ratio**2 * var_x - 2 * ratio * cov_xy) / (mean_x**2 * n)
    return float(np.sqrt(var_ratio))


def delta_method_test(x_a, y_a, x_b, y_b, alpha: float = 0.05) -> dict:
    """Two-sample delta-method test for difference of ratios R_B - R_A."""
    r_a = y_a.mean() / x_a.mean()
    r_b = y_b.mean() / x_b.mean()
    se = np.sqrt(ratio_delta_se(x_a, y_a) ** 2 + ratio_delta_se(x_b, y_b) ** 2)
    diff = r_b - r_a
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci = (diff - CONF_Z * se, diff + CONF_Z * se)
    return {
        "ratio_a": r_a,
        "ratio_b": r_b,
        "diff": diff,
        "se": se,
        "z": z,
        "p_value": p,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "significant": p < alpha,
    }


def naive_per_unit_ratio_test(x_a, y_a, x_b, y_b, alpha: float = 0.05) -> dict:
    """Naive: compute per-unit ratio r_i = y_i/x_i, then Welch t-test. Biased."""
    ra = y_a / x_a
    rb = y_b / x_b
    res = stats.ttest_ind(rb, ra, equal_var=False, alternative="two-sided")
    diff = rb.mean() - ra.mean()
    se = np.sqrt(np.var(rb, ddof=1) / len(rb) + np.var(ra, ddof=1) / len(ra))
    return {
        "ratio_a": ra.mean(),
        "ratio_b": rb.mean(),
        "diff": diff,
        "se": se,
        "p_value": float(res.pvalue),
        "significant": res.pvalue < alpha,
    }


def simulate_ratio_metric(n=2000, true_ratio=0.05, rel_lift=0.0, seed=0):
    """Generate per-user (impressions, clicks) with heterogeneous impressions.

    Heterogeneous impressions are what makes the naive per-unit ratio test
    break down (it over-weights low-impression users).
    """
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mean=3.5, sigma=0.6, size=n).astype(int) + 1
    rate = true_ratio * (1 + rel_lift)
    y = rng.binomial(x, rate)
    return x, y


def main():
    print("=== Delta Method for Ratio Metrics ===\n")

    # A/A calibration check
    aa_reject = naive_reject = 0
    n_sims = 500
    for s in range(n_sims):
        x_a, y_a = simulate_ratio_metric(seed=10_000 + s, rel_lift=0.0)
        x_b, y_b = simulate_ratio_metric(seed=20_000 + s, rel_lift=0.0)
        aa_reject += delta_method_test(x_a, y_a, x_b, y_b)["significant"]
        naive_reject += naive_per_unit_ratio_test(x_a, y_a, x_b, y_b)["significant"]
    print(f"[A/A calibration, {n_sims} sims, true lift = 0]")
    print(f"  delta method rejection rate: {aa_reject / n_sims * 100:.1f}%  (expect ~5%)")
    print(
        f"  naive per-unit t-test reject:  {naive_reject / n_sims * 100:.1f}%  "
        "(inflated when impressions vary)"
    )
    print()

    # A/B: +10% relative lift on CTR
    x_a, y_a = simulate_ratio_metric(seed=1, rel_lift=0.0)
    x_b, y_b = simulate_ratio_metric(seed=2, rel_lift=0.10)
    d = delta_method_test(x_a, y_a, x_b, y_b)
    n = naive_per_unit_ratio_test(x_a, y_a, x_b, y_b)
    print("[A/B, +10% relative lift on CTR]")
    print(
        f"  {'method':<22} {'CTR_A':>8} {'CTR_B':>8} {'diff':>10} {'SE':>10} "
        f"{'p_value':>10} {'sig':>5}"
    )
    print(
        f"  {'delta (correct)':<22} {d['ratio_a']:>8.4f} {d['ratio_b']:>8.4f} "
        f"{d['diff']:>10.5f} {d['se']:>10.5f} {d['p_value']:>10.4f} {str(d['significant']):>5}"
    )
    print(
        f"  {'naive per-unit':<22} {n['ratio_a']:>8.4f} {n['ratio_b']:>8.4f} "
        f"{n['diff']:>10.5f} {n['se']:>10.5f} {n['p_value']:>10.4f} {str(n['significant']):>5}"
    )
    print(f"\n  delta-method 95% CI for diff: [{d['ci_low']:.5f}, {d['ci_high']:.5f}]")


if __name__ == "__main__":
    main()
