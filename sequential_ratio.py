# -*- coding: utf-8 -*-
"""Sequential (always-valid) testing for ratio metrics.

Combines the delta method (correct SE for a ratio R = sum(Y)/sum(X), see
delta_method_ratio.py) with the mixture SPRT (always-valid p-values, see
msprt_always_valid.py) so you can monitor a ratio metric continuously and stop
the moment it crosses alpha without inflating Type I error.

At each interim (n users per group):
    D_hat = R_B - R_A                          (point estimate of the lift)
    V      = V_A + V_b                         (per-unit delta-method variance)
    I      = n / V                             (information)
    z      = D_hat * sqrt(I)                   (delta-method z)
    Lambda = (1 + tau^2 I)^(-1/2) * exp( tau^2 z^2 I / (2 (1 + tau^2 I)) )
    p      = 1 / Lambda                        (always-valid)

`tau` is the prior SD of the effect D in natural ratio-difference units — set it
near the MDE you care about (e.g. for CTR, the smallest lift you'd act on).
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def ratio_delta_var_perunit(x, y) -> float:
    """Per-unit delta-method variance of R = sum(y)/sum(x) = mean(y)/mean(x)."""
    mean_x = x.mean()
    ratio = y.mean() / mean_x
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    return float((var_y + ratio**2 * var_x - 2 * ratio * cov) / mean_x**2)


def ratio_interim(x_a, y_a, x_b, y_b):
    """Delta-method D_hat, z, and information I for the difference of ratios."""
    R_a = y_a.mean() / x_a.mean()
    R_b = y_b.mean() / x_b.mean()
    D = R_b - R_a
    V = ratio_delta_var_perunit(x_a, y_a) + ratio_delta_var_perunit(x_b, y_b)
    n = len(x_a)
    I = n / V
    z = D * np.sqrt(I)
    return {"D": float(D), "z": float(z), "I": float(I), "R_a": float(R_a),
            "R_b": float(R_b), "se": float(np.sqrt(V / n))}


def msprt_ratio(D, z, I, tau):
    lam = (1 + tau**2 * I) ** (-0.5) * np.exp(tau**2 * z**2 * I / (2 * (1 + tau**2 * I)))
    return float(lam), float(1.0 / lam)


def gen_batch(n, base_rate, rel_lift, seed):
    """One batch of per-user (impressions, clicks) per group, heterogeneous X."""
    rng = np.random.default_rng(seed)
    def group(rate):
        x = rng.lognormal(mean=3.5, sigma=0.6, size=n).astype(int) + 1
        y = rng.binomial(x, rate)
        return x, y
    a = group(base_rate)
    b = group(base_rate * (1 + rel_lift))
    return a, b


def run_stream(n_batches, batch_size, base_rate, rel_lift, tau, seed=0):
    """Stream batches; at each interim recompute always-valid p and naive p."""
    xa_all = ya_all = xb_all = yb_all = np.array([], dtype=float)
    records = []
    for k in range(1, n_batches + 1):
        (xa, ya), (xb, yb) = gen_batch(batch_size, base_rate, rel_lift,
                                      seed=seed * 1000 + k)
        xa_all = np.concatenate([xa_all, xa])
        ya_all = np.concatenate([ya_all, ya])
        xb_all = np.concatenate([xb_all, xb])
        yb_all = np.concatenate([yb_all, yb])
        r = ratio_interim(xa_all, ya_all, xb_all, yb_all)
        lam_av, p_av = msprt_ratio(r["D"], r["z"], r["I"], tau)
        p_naive = _two_sided_p(r["z"])
        records.append({"n": k * batch_size, **r, "p_av": p_av, "p_naive": p_naive})
    return records


def _two_sided_p(z):
    from scipy import stats
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def main():
    print("=== Sequential (Always-Valid) Testing for Ratio Metrics ===\n")
    tau = 0.01   # prior SD of the lift in natural CTR-difference units (MDE-ish)
    alpha = 0.05
    n_sims = 400
    n_batches = 20
    batch_size = 500
    base_rate = 0.05

    # --- Null calibration (A/A) ---
    av_false = naive_false = 0
    for s in range(n_sims):
        recs = run_stream(n_batches, batch_size, base_rate, rel_lift=0.0, tau=tau, seed=s)
        if min(r["p_av"] for r in recs) <= alpha:
            av_false += 1
        if min(r["p_naive"] for r in recs) <= alpha:
            naive_false += 1
    print(f"[A/A calibration, {n_sims} streams, {n_batches} interims, alpha={alpha}]")
    print(f"  {'method':<16} {'P(ever p<=alpha)':>18}")
    print(f"  {'always-valid':<16} {av_false/n_sims*100:>17.1f}%   (expect ~{alpha*100:.0f}%)")
    print(f"  {'naive delta-z':<16} {naive_false/n_sims*100:>17.1f}%   (inflated by peeking)")
    print()

    # --- Alternative: +20% relative lift, time to detection ---
    lift = 0.20
    av_detect = 0
    av_first = []
    for s in range(n_sims):
        recs = run_stream(n_batches, batch_size, base_rate, rel_lift=lift, tau=tau,
                          seed=10_000 + s)
        p_av = [r["p_av"] for r in recs]
        first = next((r["n"] for r in recs if r["p_av"] <= alpha), None)
        if first is not None:
            av_detect += 1
            av_first.append(first)
    print(f"[A/B, +{int(lift*100)}% relative lift on CTR]")
    print(f"  always-valid detects in {av_detect/n_sims*100:.1f}% of streams")
    if av_first:
        print(f"  median users/group at detection: {int(np.median(av_first))}")
    print(f"  (with batch_size={batch_size}, max n = {n_batches*batch_size}/group)")


if __name__ == "__main__":
    main()
