# -*- coding: utf-8 -*-
import delta_method_ratio as dmr
import numpy as np
import ratio_cuped as rc


def test_type1_calibrated():
    n_sims = 300
    rejects = 0
    for s in range(n_sims):
        x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        rejects += rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)[
            "significant"
        ]
    rate = rejects / n_sims
    assert 0.015 <= rate <= 0.09, f"Type I = {rate:.3f}, expected ~0.05"


def test_coverage():
    n_sims = 300
    covered = 0
    for s in range(n_sims):
        x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        res = rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)
        covered += res["ci_low"] <= 0 <= res["ci_high"]
    rate = covered / n_sims
    assert 0.90 <= rate <= 0.99, f"coverage = {rate:.3f}, expected ~0.95"


def test_variance_reduction_vs_delta_method():
    n_sims = 100
    se_delta, se_cuped = [], []
    for s in range(n_sims):
        x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        se_delta.append(dmr.delta_method_test(xa, ya, xb, yb)["se"])
        se_cuped.append(
            rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)["se"]
        )
    reduction = 1 - np.mean(se_cuped) ** 2 / np.mean(se_delta) ** 2
    assert reduction > 0.05, f"variance reduction = {reduction:.3f}"
