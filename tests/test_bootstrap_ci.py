# -*- coding: utf-8 -*-
import bootstrap_ci as bc
import numpy as np


def test_bca_mean_coverage():
    n_sims = 200
    cover = 0
    for s in range(n_sims):
        r = np.random.default_rng(10_000 + s)
        x = r.lognormal(2.0, 1.0, 300)
        y = r.lognormal(2.0, 1.0, 300)
        ci = bc.bootstrap_diff(x, y, statistic=np.mean, n_boot=399, seed=s)
        cover += ci["ci_low"] <= 0 <= ci["ci_high"]
    assert 0.90 <= cover / n_sims <= 0.99, f"coverage = {cover / n_sims:.3f}"


def test_percentile_and_bca_agree_on_mean():
    rng = np.random.default_rng(1)
    a = rng.lognormal(2.0, 1.0, 1000)
    b = rng.lognormal(2.1, 1.0, 1000)
    pct = bc.bootstrap_diff(a, b, statistic=np.mean, n_boot=1000, method="percentile", seed=1)
    bca = bc.bootstrap_diff(a, b, statistic=np.mean, n_boot=1000, method="bca", seed=1)
    assert pct["point"] == bca["point"]  # same point estimate either way
    assert pct["ci_low"] < pct["ci_high"] and bca["ci_low"] < bca["ci_high"]
    # for right-skewed data BCa pulls the CI down on both ends (bias correction)
    assert bca["ci_high"] <= pct["ci_high"]
    assert bca["ci_low"] <= pct["ci_low"]


def test_median_and_quantile_no_normal_theory():
    rng = np.random.default_rng(2)
    a = rng.lognormal(2.0, 1.0, 1000)
    b = rng.lognormal(2.1, 1.0, 1000)
    med = bc.bootstrap_diff(a, b, statistic=np.median, n_boot=500, seed=2)
    q90 = bc.bootstrap_diff(a, b, statistic=lambda x: np.quantile(x, 0.9), n_boot=500, seed=3)
    assert med["ci_low"] < med["ci_high"]
    assert q90["ci_low"] < q90["ci_high"]


def test_bca_matches_scipy_golden():
    """BCa interval must agree with the authoritative reference (scipy) on both ends.

    Regression guard: the acceleration constant is easy to get subtly wrong
    (wrong jackknife signs, missing 1/6), which shifts the whole interval.
    Tolerance is relative to the CI width because the two implementations use
    different bootstrap resamples, so exact equality is not expected.
    """
    from scipy import stats

    def diff_stat(a, b, axis=0):
        return np.mean(b, axis=axis) - np.mean(a, axis=axis)

    for seed in (1, 7, 13):
        r = np.random.default_rng(seed)
        a = r.lognormal(2.0, 1.0, 150)
        b = r.lognormal(2.1, 1.0, 150)
        mine = bc.bootstrap_diff(a, b, np.mean, n_boot=2000, method="bca", seed=seed)
        ref = stats.bootstrap(
            (a, b),
            diff_stat,
            n_resamples=2000,
            method="bca",
            confidence_level=0.95,
            random_state=np.random.default_rng(seed),
            vectorized=True,
        )
        lo, hi = ref.confidence_interval
        width = mine["ci_high"] - mine["ci_low"]
        assert abs(mine["ci_low"] - lo) < 0.10 * width, f"seed={seed} lower end off"
        assert abs(mine["ci_high"] - hi) < 0.10 * width, f"seed={seed} upper end off"
