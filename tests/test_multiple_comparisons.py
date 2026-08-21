# -*- coding: utf-8 -*-
import numpy as np

import multiple_comparisons as mc


def test_bonferroni_worked_example():
    p = np.array([0.001, 0.04, 0.20])
    adj, rej = mc.bonferroni(p)
    assert rej[0] and not rej[1] and not rej[2]
    assert adj[1] == 0.12  # 0.04 * 3


def test_bh_adjusted_pvalues_sorted_like_p():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=30)
    adj, _ = mc.benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_naive_inflates_fwer_bonferroni_and_bh_control():
    stats = mc._calibrate(m=10, n_trials=1500, seed=1)
    assert stats["naive"]["fwer"] > 0.2
    assert abs(stats["bonferroni"]["fwer"] - 0.05) < 0.03
    assert abs(stats["bh"]["fdp"] - 0.05) < 0.03


def test_bh_less_conservative_than_bonferroni_with_real_signal():
    # 10 tests, one strong true positive + a few borderline
    p = np.array([0.0001, 0.01, 0.02, 0.03, 0.05, 0.06, 0.1, 0.2, 0.3, 0.4])
    _, rej_b = mc.bonferroni(p)
    _, rej_bh = mc.benjamini_hochberg(p)
    assert rej_bh.sum() >= rej_b.sum()
