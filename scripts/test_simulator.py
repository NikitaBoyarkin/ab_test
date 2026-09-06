# -*- coding: utf-8 -*-
"""Generic test simulator — empirical calibration of any A/B test.

A test is "valid" if, under the null (no effect), its rejection rate equals the
nominal alpha (Type I error), and "powered" if it rejects often enough under a
real effect. Rather than trust asymptotic promises, simulate the whole pipeline
end-to-end.

Plug in:
  dgp_fn(effect) -> (group_a, group_b)   # data-generating process
  test_fn(a, b)  -> dict with key 'reject' (bool)   # the test under audit

Returns empirical Type I error (effect=0) and power (effect>0), plus a
calibration curve over a range of true effects.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def simulate(dgp_fn, test_fn, effect=0.0, n_trials=1000, alpha=0.05, seed=0):
    """Return empirical rejection rate for a given true effect."""
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_trials):
        # give the dgp a deterministic sub-seed so trials are independent
        a, b = dgp_fn(effect, seed=int(rng.integers(1 << 31)))
        if test_fn(a, b)["reject"]:
            rejects += 1
    rate = rejects / n_trials
    return {
        "effect": effect,
        "n_trials": n_trials,
        "alpha": alpha,
        "rejection_rate": rate,
        "is_type1": effect == 0.0,
        "label": "Type I error" if effect == 0.0 else "power",
    }


def calibration_curve(dgp_fn, test_fn, effects, n_trials=500, alpha=0.05, seed=0):
    """Rejection rate across a range of true effects (power curve)."""
    return [simulate(dgp_fn, test_fn, e, n_trials, alpha, seed + k) for k, e in enumerate(effects)]


# --- example DGP + tests -----------------------------------------------------
def dgp_normal(effect, n=1000, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, sd, n)
    b = rng.normal(effect, sd, n)
    return a, b


def test_welch(a, b, alpha=0.05):
    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"reject": p < alpha, "p_value": float(p)}


def test_mannwhitney(a, b, alpha=0.05):
    from scipy import stats

    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"reject": p < alpha, "p_value": float(p)}


def test_bootstrap_mean(a, b, alpha=0.05, n_boot=499, seed=123):
    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    for i in range(n_boot):
        ba = rng.choice(a_arr, len(a_arr), replace=True).mean()
        bb = rng.choice(b_arr, len(b_arr), replace=True).mean()
        boot_diffs[i] = bb - ba
    lo, hi = np.quantile(boot_diffs, [alpha / 2, 1 - alpha / 2])
    return {"reject": not (lo <= 0 <= hi), "ci": (lo, hi)}


def _fmt(row):
    tag = "Type I" if row["is_type1"] else "power"
    return (
        f"  effect={row['effect']:+.3f}  reject_rate={row['rejection_rate']:.3f}  "
        f"({tag}, alpha={row['alpha']})"
    )


def main():
    print("=== Generic Test Simulator ===\n")

    print("[Welch t-test audit]")
    r0 = simulate(dgp_normal, test_welch, effect=0.0, n_trials=1000, seed=1)
    r1 = simulate(dgp_normal, test_welch, effect=0.1, n_trials=1000, seed=2)
    print(_fmt(r0), "-> 5% Type I expected")
    print(_fmt(r1), "-> high power expected")

    print("\n[Power curve: Welch t-test across effects]")
    for row in calibration_curve(
        dgp_normal, test_welch, np.linspace(0.0, 0.2, 5), n_trials=400, seed=10
    ):
        print(_fmt(row))

    print("\n[Mann-Whitney on skewed (lognormal) data] — Welch would be miscalibrated")

    def dgp_lognormal(effect, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.lognormal(0, 1, n)
        b = rng.lognormal(np.log1p(effect), 1, n)  # multiplicative effect
        return a, b

    r0 = simulate(dgp_lognormal, test_mannwhitney, effect=0.0, n_trials=500, seed=1)
    print(_fmt(r0), "-> ~5% Type I expected")


if __name__ == "__main__":
    main()
