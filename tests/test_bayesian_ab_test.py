# -*- coding: utf-8 -*-
import numpy as np

import bayesian_ab_test as b


def test_b_correctly_favored_when_clearly_better():
    pa, pb = b.beta_binomial(s=(120, 150), f=(1880, 1850), n_samples=20_000, seed=1)
    d = b.decide(pa, pb, rope=0.005)
    assert d["p_b_better"] > 0.95
    assert d["decision"] in ("ship B", "lean B (keep testing)")


def test_aa_not_favoring_either():
    pa, pb = b.beta_binomial(s=(100, 102), f=(1900, 1898), n_samples=20_000, seed=2)
    d = b.decide(pa, pb, rope=0.005)
    assert 0.3 < d["p_b_better"] < 0.7
    assert d["decision"] not in ("ship B", "keep A")


def test_calibration_p_better_uniform():
    rng = np.random.default_rng(0)
    ps = []
    for s in range(500):
        sa = rng.binomial(2000, 0.10)
        sb = rng.binomial(2000, 0.10)
        pa, pb = b.beta_binomial(s=(sa, sb), f=(2000 - sa, 2000 - sb), n_samples=400, seed=s)
        ps.append(np.mean(pb > pa))
    ps = np.array(ps)
    assert abs(ps.mean() - 0.5) < 0.04, f"mean P(B>A) = {ps.mean():.3f}"
    assert abs(np.mean(ps <= 0.05) - 0.05) < 0.03, f"frac P(B>A)<=0.05 = {np.mean(ps <= 0.05):.3f}"


def test_normal_normal_detects_lift():
    pa, pb = b.normal_normal(2.10, 9.0, 5000, 2.25, 9.0, 5000, seed=3)
    d = b.decide(pa, pb, rope=0.05)
    assert d["p_b_better"] > 0.9
    assert d["lift_mean"] > 0
