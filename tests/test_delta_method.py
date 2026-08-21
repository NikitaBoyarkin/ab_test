# -*- coding: utf-8 -*-
import numpy as np

import delta_method_ratio as dmr


def test_aa_calibration_type1_about_5pct():
    n_sims = 300
    rejects = 0
    for s in range(n_sims):
        xa, ya = dmr.simulate_ratio_metric(seed=10_000 + s, rel_lift=0.0)
        xb, yb = dmr.simulate_ratio_metric(seed=20_000 + s, rel_lift=0.0)
        rejects += dmr.delta_method_test(xa, ya, xb, yb)["significant"]
    rate = rejects / n_sims
    assert 0.015 <= rate <= 0.09, f"Type I = {rate:.3f}, expected ~0.05"


def test_delta_method_se_accurate():
    # Across A/A sims, the reported SE should match the empirical spread of the
    # ratio-difference estimator (this is what "correct SE" means in practice).
    n_sims = 300
    diffs, ses = [], []
    for s in range(n_sims):
        xa, ya = dmr.simulate_ratio_metric(seed=10_000 + s, rel_lift=0.0)
        xb, yb = dmr.simulate_ratio_metric(seed=20_000 + s, rel_lift=0.0)
        d = dmr.delta_method_test(xa, ya, xb, yb)
        diffs.append(d["diff"])
        ses.append(d["se"])
    emp_se = np.std(diffs, ddof=1)
    mean_se = np.mean(ses)
    ratio = mean_se / emp_se
    assert 0.7 <= ratio <= 1.3, f"reported SE / empirical SE = {ratio:.2f}"


def test_detects_ten_percent_lift():
    xa, ya = dmr.simulate_ratio_metric(seed=1, rel_lift=0.0)
    xb, yb = dmr.simulate_ratio_metric(seed=2, rel_lift=0.10)
    d = dmr.delta_method_test(xa, ya, xb, yb)
    assert d["significant"]
    assert d["diff"] > 0
    assert d["ci_low"] < d["diff"] < d["ci_high"]
