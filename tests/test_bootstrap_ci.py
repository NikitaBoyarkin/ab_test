# -*- coding: utf-8 -*-
import numpy as np

import bootstrap_ci as bc


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
