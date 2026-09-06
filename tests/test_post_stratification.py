# -*- coding: utf-8 -*-
import numpy as np
import post_stratification as ps


def test_type1_calibrated():
    n_sims = 300
    rejects = 0
    for s in range(n_sims):
        t, y, st = ps.simulate_stratified(seed=s)
        rejects += ps.stratified_ate(t, y, st)["significant"]
    rate = rejects / n_sims
    assert 0.015 <= rate <= 0.09, f"Type I = {rate:.3f}, expected ~0.05"


def test_corrects_imbalance_bias():
    # effect=0 but treatment is imbalanced toward the high-baseline stratum:
    # the naive estimator is biased, the stratified one is not.
    n_sims = 200
    naive_errors, strat_errors = [], []
    for s in range(n_sims):
        t, y, st = ps.simulate_stratified(n=3000, effect=0.0, imbalance=0.3, seed=s)
        naive_errors.append(ps.naive_ate(t, y)["diff"])
        strat_errors.append(ps.stratified_ate(t, y, st)["ate"])
    assert abs(np.mean(naive_errors)) > 0.005, "naive should be visibly biased"
    assert abs(np.mean(strat_errors)) < 0.002, (
        f"stratified mean error = {np.mean(strat_errors):.4f}"
    )


def test_variance_reduction_with_predictive_strata():
    n_sims = 100
    se_naive, se_strat = [], []
    for s in range(n_sims):
        t, y, st = ps.simulate_stratified(n=3000, effect=0.02, imbalance=0.0, seed=s)
        se_naive.append(ps.naive_ate(t, y)["se"])
        se_strat.append(ps.stratified_ate(t, y, st)["se"])
    mean_reduction = 1 - np.mean(se_strat) ** 2 / np.mean(se_naive) ** 2
    assert mean_reduction > 0.05, f"variance reduction = {mean_reduction:.3f}"
