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
    p      = 1 / Lambda                        (always-valid)

Each mixture component is a normal prior on the lift in NATURAL units, with its
own fixed scale tau_j (a plausible effect size, not a multiple of the current
standard error):

    delta ~ N(0, tau_j^2),  j = 1..J
    Lambda_j = (1 + tau_j^2 I)^(-1/2) * exp( tau_j^2 I z^2 / (2 (1 + tau_j^2 I)) )
    Lambda   = mean_j Lambda_j

Why a mixture of fixed scales: one fixed `tau` is valid but wasteful — it is
tuned for a single information level, so its spend is well below alpha whenever
the observed I sits elsewhere (0.7-1.3% at alpha=5% with tau=0.01). A mixture
with fixed weights is still a nonnegative martingale with expectation 1 under
H0 (log-average of supermartingales is a supermartingale), so Ville's
inequality still holds, while the mixture picks up whichever scale matches the
information actually reached: the A/A spend grows toward alpha as monitoring
lengthens instead of decaying.

What must NOT be done — and was, in an earlier revision — is scaling the prior
with the information (a "local prior" at +-c/sqrt(I)). That cancels I out of
Lambda, so the rejection boundary becomes a fixed curve in z (~2.7 SE) instead
of growing as sqrt(2 log(1 + tau^2 I)); the result is a repeated fixed-sample
test whose spend grows without bound (measured 9.2% at alpha=5% over 600 looks).
tests/test_sequential_ratio.py pins spend <= alpha at several horizons so that
regression cannot come back.

Pass a scalar `tau` for a single fixed scale (legacy path); `tau=None` (default)
mixes PRIOR_TAU_GRID.

Caveat: I = n / V carries the DATA-estimated delta variance V, so the
components are exact martingales only if V were known in advance. With V
estimated this is a pseudo-martingale; the validity tests bounds the realized
A/A spend empirically rather than proving the inequality.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Prior scales for the mixture, in NATURAL ratio-difference units (CTR deltas).
# Spread over the effect sizes worth powering for on a 5%-CTR metric: 0.1% to 0.5%
# absolute lift (2-10% relative). Fixed in natural units on purpose — see the
# module docstring. Chosen by sweeping the A/A spend: this grid reaches ~2% of the
# 5% budget at 250 looks and ~3.7% at 600, while a single scale stays at <=2%.
PRIOR_TAU_GRID = (0.001, 0.002, 0.005)


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
    info = n / V
    z = D * np.sqrt(info)
    return {
        "D": float(D),
        "z": float(z),
        "I": float(info),
        "R_a": float(R_a),
        "R_b": float(R_b),
        "se": float(np.sqrt(V / n)),
    }


def _log_normal_component(z, info, tau):
    """log of the fixed-scale normal component (1 + tau^2 I)^(-1/2) exp(...)."""
    u = float(tau) ** 2 * info
    return float(-0.5 * np.log1p(u) + u * z * z / (2 * (1 + u)))


def _log_lambda(z, info, tau):
    """log of the mixture Lambda: PRIOR_TAU_GRID, or the given fixed scale(s)."""
    if tau is None:
        taus = PRIOR_TAU_GRID
    elif np.ndim(tau) == 0:
        taus = (tau,)
    else:
        taus = tuple(tau)
    logs = [_log_normal_component(z, info, t) for t in taus]
    return float(np.logaddexp.reduce(logs) - np.log(len(logs)))


def msprt_ratio(D, z, info, tau=None):
    """Mixture-SPRT Lambda and its always-valid p-value for a ratio metric.

    `tau=None` (default) mixes the fixed scales in PRIOR_TAU_GRID. A scalar
    `tau` is a single fixed-scale normal prior; a sequence mixes those scales.
    """
    log_lam = _log_lambda(z, info, tau)
    lam = float(np.exp(min(log_lam, 700.0)))
    p = float(np.exp(min(0.0, -log_lam)))
    return lam, p


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


def run_stream(n_batches, batch_size, base_rate, rel_lift, tau=None, seed=0):
    """Stream batches; at each interim recompute always-valid p and naive p.

    `tau=None` mixes PRIOR_TAU_GRID (see msprt_ratio).
    """
    xa_all = ya_all = xb_all = yb_all = np.array([], dtype=float)
    records = []
    for k in range(1, n_batches + 1):
        (xa, ya), (xb, yb) = gen_batch(batch_size, base_rate, rel_lift, seed=seed * 1000 + k)
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
    tau = None  # mixture over PRIOR_TAU_GRID (fixed natural-unit scales)
    alpha = 0.05
    n_sims = 300
    n_batches = 60
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
    print(
        f"  {'always-valid':<16} {av_false / n_sims * 100:>17.1f}%   "
        f"(<= {alpha * 100:.0f}% at every horizon; single fixed tau spends ~1%)"
    )
    print(f"  {'naive delta-z':<16} {naive_false / n_sims * 100:>17.1f}%   (inflated by peeking)")
    print()

    # --- Alternative: +20% relative lift, time to detection ---
    lift = 0.20
    av_detect = 0
    av_first = []
    for s in range(n_sims):
        recs = run_stream(n_batches, batch_size, base_rate, rel_lift=lift, tau=tau, seed=10_000 + s)
        first = next((r["n"] for r in recs if r["p_av"] <= alpha), None)
        if first is not None:
            av_detect += 1
            av_first.append(first)
    print(f"[A/B, +{int(lift * 100)}% relative lift on CTR]")
    print(f"  always-valid detects in {av_detect / n_sims * 100:.1f}% of streams")
    if av_first:
        print(f"  median users/group at detection: {int(np.median(av_first))}")
    print(f"  (with batch_size={batch_size}, max n = {n_batches * batch_size}/group)")


if __name__ == "__main__":
    main()
