# -*- coding: utf-8 -*-
import cupac
import cuped
import numpy as np


def test_type1_calibrated():
    n_sims = 200
    rejects = 0
    for s in range(n_sims):
        Xa, ya, Xb, yb = cupac.simulate_cupac(seed=s)
        rejects += cupac.cupac_test(ya, Xa, yb, Xb)["significant"]
    rate = rejects / n_sims
    assert 0.02 <= rate <= 0.08, f"Type I = {rate:.3f}, expected ~0.05"


def test_unbiased_under_null():
    n_sims = 200
    diffs = []
    for s in range(n_sims):
        Xa, ya, Xb, yb = cupac.simulate_cupac(seed=s)
        diffs.append(cupac.cupac_test(ya, Xa, yb, Xb)["diff"])
    assert abs(np.mean(diffs)) < 0.02, f"mean diff = {np.mean(diffs):.4f}"


def test_variance_reduction_better_than_single_covariate():
    n_sims = 100
    se_naive, se_cuped, se_cupac = [], [], []
    for s in range(n_sims):
        Xa, ya, Xb, yb = cupac.simulate_cupac(n=3000, effect=0.05, n_features=3, seed=s)
        se_naive.append(np.sqrt(np.var(ya, ddof=1) / len(ya) + np.var(yb, ddof=1) / len(yb)))
        theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([Xa[:, 2], Xb[:, 2]]))
        a = cuped.cuped_adjust(ya, Xa[:, 2], theta)
        b = cuped.cuped_adjust(yb, Xb[:, 2], theta)
        se_cuped.append(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
        se_cupac.append(cupac.cupac_test(ya, Xa, yb, Xb)["se"])
    red_cuped = 1 - np.mean(se_cuped) ** 2 / np.mean(se_naive) ** 2
    red_cupac = 1 - np.mean(se_cupac) ** 2 / np.mean(se_naive) ** 2
    assert red_cuped > 0.05, f"CUPED reduction = {red_cuped:.3f}"
    assert red_cupac > red_cuped + 0.02, f"CUPAC {red_cupac:.3f} vs CUPED {red_cuped:.3f}"
