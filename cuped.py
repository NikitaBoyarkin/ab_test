# -*- coding: utf-8 -*-
"""CUPED — Controlled-Experiment Using Pre-Experiment Data.

If a pre-experiment covariate X (typically the pre-period value of the same
metric) is correlated with the outcome Y, adjust Y by its linear projection
on X and test on the residual. Variance drops by ~(1 - rho^2).

    theta = Cov(Y, X) / Var(X)      # from pooled pre+post, unaffected by treatment
    Y_adj = Y - theta * (X - mean(X))

Reference: Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of
Online Controlled Experiments by Utilizing Pre-Experiment Data".
"""

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def cuped_theta(y, x) -> float:
    """theta = Cov(Y, X) / Var(X) from the pre-period covariate x and outcome y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cov = np.cov(x, y, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    return float(cov / var_x)


def cuped_adjust(y, x, theta) -> np.ndarray:
    """Y_adj = Y - theta * (X - mean(X)). mean(X) from the pooled covariate."""
    x = np.asarray(x, dtype=float)
    return np.asarray(y, dtype=float) - theta * (x - x.mean())


def simulate_cuped(n=2000, effect=0.0, rho=0.6, seed=0):
    """Each user has a pre-period outcome X and post-period outcome Y, linked
    by a shared latent factor with correlation rho. Treatment B adds `effect`
    to the post-period outcome only (pre-period is unaffected)."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, 2 * n)
    x_pre = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, 2 * n)  # pre-period
    y_post = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, 2 * n)  # post-period
    y_post[n:] += effect  # treatment applied to group B in post-period only
    return x_pre[:n], y_post[:n], x_pre[n:], y_post[n:]  # (x_a, y_a, x_b, y_b)


def t_test(a, b) -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = b.mean() - a.mean()
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return {
        "diff": diff,
        "se": se,
        "p_value": float(p),
        "ci_half": stats.norm.ppf(0.975) * se,
        "var_a": np.var(a, ddof=1),
        "var_b": np.var(b, ddof=1),
    }


def main():
    print("=== CUPED Variance Reduction ===\n")

    x_a, y_a, x_b, y_b = simulate_cuped(n=2000, effect=0.05, rho=0.6, seed=1)
    pre = t_test(y_a, y_b)
    # theta from pooled pre-period covariate vs post outcome (unaffected by treatment)
    theta = cuped_theta(np.concatenate([y_a, y_b]), np.concatenate([x_a, x_b]))
    y_a_adj = cuped_adjust(y_a, x_a, theta)
    y_b_adj = cuped_adjust(y_b, x_b, theta)
    post = t_test(y_a_adj, y_b_adj)

    var_reduction = 1 - (post["var_a"] + post["var_b"]) / (pre["var_a"] + pre["var_b"])
    xy_corr = float(np.corrcoef(np.concatenate([x_a, x_b]), np.concatenate([y_a, y_b]))[0, 1])
    print("[Single A/B, effect=0.05, rho=0.6]")
    print(f"  {'method':<14} {'diff':>8} {'SE':>8} {'95% CI half-width':>20} {'p_value':>10}")
    print(
        f"  {'naive':<14} {pre['diff']:>8.4f} {pre['se']:>8.4f} {pre['ci_half']:>20.4f} {pre['p_value']:>10.4f}"
    )
    print(
        f"  {'CUPED':<14} {post['diff']:>8.4f} {post['se']:>8.4f} {post['ci_half']:>20.4f} {post['p_value']:>10.4f}"
    )
    print(
        f"  variance reduction: {var_reduction * 100:.1f}%  (theory = corr(X,Y)^2 = {xy_corr**2 * 100:.1f}%, corr(X,Y)={xy_corr:.2f})"
    )
    print()

    def trial(effect, seed, use_cuped):
        x_a, y_a, x_b, y_b = simulate_cuped(n=2000, effect=effect, rho=0.6, seed=seed)
        if not use_cuped:
            return t_test(y_a, y_b)["p_value"]
        theta = cuped_theta(np.concatenate([y_a, y_b]), np.concatenate([x_a, x_b]))
        a = cuped_adjust(y_a, x_a, theta)
        b = cuped_adjust(y_b, x_b, theta)
        return t_test(a, b)["p_value"]

    n_sims = 400
    alpha = 0.05
    t1_naive = t1_cuped = pow_naive = pow_cuped = 0
    for s in range(n_sims):
        t1_naive += trial(0.0, s, False) < alpha
        t1_cuped += trial(0.0, s, True) < alpha
        pow_naive += trial(0.03, s, False) < alpha
        pow_cuped += trial(0.03, s, True) < alpha
    print(f"[Calibration, {n_sims} sims, rho=0.6]")
    print(f"  {'metric':<22} {'naive':>8} {'CUPED':>8}")
    print(
        f"  {'Type I error':<22} {t1_naive / n_sims * 100:>7.1f}% {t1_cuped / n_sims * 100:>7.1f}%"
    )
    print(
        f"  {'power (effect=0.03)':<22} {pow_naive / n_sims * 100:>7.1f}% {pow_cuped / n_sims * 100:>7.1f}%"
    )
    print("  CUPED keeps Type I calibrated and lifts power at the same sample size.")


if __name__ == "__main__":
    main()
