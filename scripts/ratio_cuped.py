# -*- coding: utf-8 -*-
"""CUPED variance reduction for ratio metrics (CTR, RPC).

Ratio metrics R = sum(Y)/sum(X) (e.g. CTR = clicks/impressions) are ratios of
two random variables. The delta method gives the correct SE; CUPED reduces it
further. The standard trick (Deng et al. 2013) is to *linearize* the ratio:

    Z_i = Y_i - R * X_i,   R = pooled ratio = sum(Y)/sum(X)

so that R_B - R_A ~= (mean(Z_B) - mean(Z_A)) / mean(X). CUPED is then applied
to the single outcome Z using its pre-period analog Z_pre = Y_pre - R*X_pre:

    Z_adj = Z - theta * (Z_pre - mean(Z_pre))

theta is the OLS slope of Z on Z_pre (pooled); centering by the pooled
pre-period mean (an estimate of E[Z_pre]) keeps the variance reduction of the
estimator real. The delta-method scale 1/mean(X) maps the Z-difference back to
a ratio difference.

Reference: Deng, Xu, Kohavi, Walker (2013) "Improving the sensitivity of
online controlled experiments by utilizing pre-experiment data" (CUPED);
Deng, Knoblich, Lu (2018) "Applying the Delta Method in Metrics Experiments".
"""

import warnings

import delta_method_ratio as dmr
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

CONF_Z = stats.norm.ppf(0.975)


def ratio_cuped_test(
    x_a,
    y_a,
    x_b,
    y_b,
    x_pre_a,
    y_pre_a,
    x_pre_b,
    y_pre_b,
    alpha: float = 0.05,
) -> dict:
    """CUPED-adjusted delta-method test for difference of ratios.

    x/y are post-period denominator/numerator; x_pre/y_pre are pre-period.
    The ratio is linearized to Z = Y - R*X, CUPED is applied to Z using Z_pre,
    and the result is scaled by 1/mean(X) (delta method).
    """
    x_all = np.concatenate([x_a, x_b])
    y_all = np.concatenate([y_a, y_b])
    R = y_all.mean() / x_all.mean()

    z_a = np.asarray(y_a, dtype=float) - R * np.asarray(x_a, dtype=float)
    z_b = np.asarray(y_b, dtype=float) - R * np.asarray(x_b, dtype=float)
    z_pre_a = np.asarray(y_pre_a, dtype=float) - R * np.asarray(x_pre_a, dtype=float)
    z_pre_b = np.asarray(y_pre_b, dtype=float) - R * np.asarray(x_pre_b, dtype=float)

    z_all = np.concatenate([z_a, z_b])
    z_pre_all = np.concatenate([z_pre_a, z_pre_b])
    theta = np.cov(z_all, z_pre_all, ddof=1)[0, 1] / np.var(z_pre_all, ddof=1)
    z_pre_mean = z_pre_all.mean()

    za_adj = z_a - theta * (z_pre_a - z_pre_mean)
    zb_adj = z_b - theta * (z_pre_b - z_pre_mean)

    scale = 1.0 / x_all.mean()
    diff = (zb_adj.mean() - za_adj.mean()) * scale
    se = (
        np.sqrt(np.var(za_adj, ddof=1) / len(za_adj) + np.var(zb_adj, ddof=1) / len(zb_adj)) * scale
    )
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci = (diff - CONF_Z * se, diff + CONF_Z * se)
    return {
        "ratio_a": y_a.mean() / x_a.mean(),
        "ratio_b": y_b.mean() / x_b.mean(),
        "diff": diff,
        "se": se,
        "z": z,
        "p_value": p,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "significant": p < alpha,
        "theta": theta,
        "r_squared": float(np.corrcoef(z_all, z_pre_all)[0, 1] ** 2),
    }


def simulate_ratio_cuped(n=2000, true_ratio=0.05, rel_lift=0.0, seed=0):
    """Per-user pre/post impressions and clicks with heterogeneous click rates.

    Latent activity u drives both pre- and post-period impressions; per-user
    click rate r varies around true_ratio. Both make pre-period data predictive
    of post-period — the signal CUPED exploits. Returns (x_pre, y_pre, x_post, y_post).
    """
    rng = np.random.default_rng(seed)
    u = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    rate = true_ratio * rng.lognormal(mean=0.0, sigma=0.5, size=n)
    x_pre = (u * rng.lognormal(mean=3.5, sigma=0.3, size=n)).astype(int) + 1
    x_post = (u * rng.lognormal(mean=3.5, sigma=0.3, size=n)).astype(int) + 1
    y_pre = rng.binomial(x_pre, rate)
    y_post = rng.binomial(x_post, rate * (1 + rel_lift))
    return x_pre, y_pre, x_post, y_post


def main():
    print("=== CUPED for Ratio Metrics ===\n")

    # A/A calibration + coverage
    n_sims = 300
    rejects = 0
    covered = 0
    for s in range(n_sims):
        x_pre_a, y_pre_a, x_a, y_a = simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, x_b, y_b = simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        res = ratio_cuped_test(x_a, y_a, x_b, y_b, x_pre_a, y_pre_a, x_pre_b, y_pre_b)
        rejects += res["significant"]
        covered += res["ci_low"] <= 0 <= res["ci_high"]
    print(f"[A/A calibration, {n_sims} sims]")
    print(f"  CUPED+delta rejection rate: {rejects / n_sims * 100:.1f}%  (expect ~5%)")
    print(f"  CI coverage: {covered / n_sims * 100:.1f}%  (expect ~95%)\n")

    # Variance reduction: delta-only vs CUPED+delta
    n_sims = 100
    se_delta, se_cuped = [], []
    for s in range(n_sims):
        x_pre_a, y_pre_a, x_a, y_a = simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, x_b, y_b = simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        se_delta.append(dmr.delta_method_test(x_a, y_a, x_b, y_b)["se"])
        se_cuped.append(
            ratio_cuped_test(x_a, y_a, x_b, y_b, x_pre_a, y_pre_a, x_pre_b, y_pre_b)["se"]
        )
    reduction = 1 - np.mean(se_cuped) ** 2 / np.mean(se_delta) ** 2
    print(f"[Variance reduction, {n_sims} sims]")
    print(f"  SE delta-only:  {np.mean(se_delta):.6f}")
    print(f"  SE CUPED+delta: {np.mean(se_cuped):.6f}")
    print(f"  variance reduction: {reduction * 100:.1f}%\n")

    # A/B: +10% relative lift
    x_pre_a, y_pre_a, x_a, y_a = simulate_ratio_cuped(seed=1, rel_lift=0.0)
    x_pre_b, y_pre_b, x_b, y_b = simulate_ratio_cuped(seed=2, rel_lift=0.10)
    d = dmr.delta_method_test(x_a, y_a, x_b, y_b)
    c = ratio_cuped_test(x_a, y_a, x_b, y_b, x_pre_a, y_pre_a, x_pre_b, y_pre_b)
    print("[A/B, +10% relative lift on CTR]")
    print(f"  {'method':<16} {'diff':>10} {'SE':>10} {'p_value':>10} {'sig':>5}")
    print(
        f"  {'delta only':<16} {d['diff']:>10.5f} {d['se']:>10.5f} {d['p_value']:>10.4f} {str(d['significant']):>5}"
    )
    print(
        f"  {'CUPED+delta':<16} {c['diff']:>10.5f} {c['se']:>10.5f} {c['p_value']:>10.4f} {str(c['significant']):>5}"
    )


if __name__ == "__main__":
    main()
