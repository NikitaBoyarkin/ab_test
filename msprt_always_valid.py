# -*- coding: utf-8 -*-
"""mSPRT — mixture Sequential Probability Ratio Test, always-valid p-values.

A z-test p-value is only valid at one pre-specified sample size. Recompute it
while data streams in ("peeking") and the Type I error inflates (see
group_sequential.py for the boundary-fix alternative). The *mixture* SPRT gives
an always-valid p-value: for any stopping time tau,

    P( exists t <= T : p_t <= alpha ) <= alpha

so you may stop the instant p_t crosses alpha. Under H0 (mean = 0, unit-variance
observations) with a Gaussian mixing prior on the effect ~ N(0, rho^2):

    Lambda_t = (1 + rho^2 t)^(-1/2) * exp( rho^2 S_t^2 / (2 (1 + rho^2 t)) )
    reject when Lambda_t >= 1/alpha    <=>    p_t = 1 / Lambda_t <= alpha

where S_t = sum of the first t observations. Lambda is a nonnegative
supermartingale under H0, so Ville's inequality gives the always-valid guarantee.

`rho` (prior SD of the effect, in observation-SD units) trades sensitivity:
small rho favours detecting small effects, large rho favours large effects.
Pick rho near the smallest effect you care about (the MDE in SD units).

Reference: Johari, Pekelis, Walsh (2015), "Always Valid Inference: Bringing
Sequential Analysis to A/B Testing" (this is Optimizely's Stats Engine).
"""

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def msprt_lambda(S, t, rho):
    """Mixture-SPRT statistic Lambda_t for cumulative sum S at time t."""
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    return (1 + rho**2 * t) ** (-0.5) * np.exp(rho**2 * S**2 / (2 * (1 + rho**2 * t)))


def always_valid_pvalue(S, t, rho):
    """Always-valid p-value = 1 / Lambda_t (valid at any stopping time)."""
    return 1.0 / msprt_lambda(S, t, rho)


def naive_z_pvalue(S, t):
    """Standard z-test p-value at time t (valid only if t is pre-planned)."""
    z = np.asarray(S, dtype=float) / np.sqrt(np.asarray(t, dtype=float))
    return 2 * (1 - stats.norm.cdf(np.abs(z)))


def _median_hit(hits, n_cap):
    detected = hits[hits < n_cap]
    return float(np.median(detected)) if detected.size else float("nan")


def main():
    print("=== mSPRT / Always-Valid p-values ===\n")
    rho = 0.5  # prior SD of the effect (in obs-SD units)
    alpha = 0.05
    n = 500
    n_streams = 2000

    # --- Null calibration: does any stopping time trigger a false positive? ---
    av_false = naive_false = 0
    av_hits, naive_hits = [], []
    for s in range(n_streams):
        rng = np.random.default_rng(s)
        x = rng.normal(0.0, 1.0, n)
        S = np.cumsum(x)
        t = np.arange(1, n + 1)
        p_av = always_valid_pvalue(S, t, rho)
        p_naive = naive_z_pvalue(S, t)
        # first crossing time (n if never)
        i_av = int(np.argmax(p_av <= alpha)) if (p_av <= alpha).any() else n
        i_naive = int(np.argmax(p_naive <= alpha)) if (p_naive <= alpha).any() else n
        av_hits.append(i_av)
        naive_hits.append(i_naive)
        av_false += i_av < n
        naive_false += i_naive < n
    av_hits = np.array(av_hits)
    naive_hits = np.array(naive_hits)

    print(f"[Null calibration, {n_streams} streams, peek over t=1..{n}, alpha={alpha}]")
    print(f"  {'method':<16} {'P(ever p<=alpha)':>18}")
    print(
        f"  {'always-valid':<16} {av_false / n_streams * 100:>17.1f}%   (expect ~{alpha * 100:.0f}%)"
    )
    print(f"  {'naive z-test':<16} {naive_false / n_streams * 100:>17.1f}%   (inflated by peeking)")
    print()

    # --- Alternative: time to detection ---
    mu = 0.1  # effect in SD units
    av_t, naive_t = [], []
    for s in range(n_streams):
        rng = np.random.default_rng(10_000 + s)
        x = rng.normal(mu, 1.0, n)
        S = np.cumsum(x)
        t = np.arange(1, n + 1)
        p_av = always_valid_pvalue(S, t, rho)
        p_naive = naive_z_pvalue(S, t)
        i_av = int(np.argmax(p_av <= alpha)) if (p_av <= alpha).any() else n
        i_naive = int(np.argmax(p_naive <= alpha)) if (p_naive <= alpha).any() else n
        av_t.append(i_av)
        naive_t.append(i_naive)
    av_t = np.array(av_t)
    naive_t = np.array(naive_t)

    print(f"[Alternative, effect={mu} SD, capped at n={n}]")
    print(f"  {'method':<16} {'detect rate':>12} {'median n to detect':>20}")
    print(f"  {'always-valid':<16} {np.mean(av_t < n) * 100:>11.1f}% {_median_hit(av_t, n):>19.0f}")
    print(
        f"  {'naive z-test':<16} {np.mean(naive_t < n) * 100:>11.1f}% "
        f"{_median_hit(naive_t, n):>19.0f}"
    )
    print("  naive detects earlier, but ~the extra detections are the false positives")
    print("  seen in the null block above. always-valid buys validity, not free power.")


if __name__ == "__main__":
    main()
