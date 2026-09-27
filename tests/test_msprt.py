# -*- coding: utf-8 -*-
import msprt_always_valid as m
import numpy as np


def test_naive_inflated_always_valid_calibrated():
    rho, alpha, n = 0.5, 0.05, 300
    n_streams = 400
    av_false = naive_false = 0
    for s in range(n_streams):
        rng = np.random.default_rng(s)
        x = rng.normal(0.0, 1.0, n)
        S = np.cumsum(x)
        t = np.arange(1, n + 1)
        av_false += (m.always_valid_pvalue(S, t, rho) <= alpha).any()
        naive_false += (m.naive_z_pvalue(S, t) <= alpha).any()
    assert 0.02 <= av_false / n_streams <= 0.08, (
        f"always-valid FWER = {av_false / n_streams:.3f}, expect ~5%"
    )
    assert naive_false / n_streams > 0.15, (
        f"naive peeking FWER = {naive_false / n_streams:.3f}, should be inflated"
    )


def test_always_valid_detects_effect():
    rho, alpha, n = 0.5, 0.05, 800
    n_streams = 200
    detected = 0
    for s in range(n_streams):
        rng = np.random.default_rng(10_000 + s)
        x = rng.normal(0.1, 1.0, n)
        S = np.cumsum(x)
        t = np.arange(1, n + 1)
        detected += (m.always_valid_pvalue(S, t, rho) <= alpha).any()
    assert detected / n_streams > 0.35


def test_always_valid_pvalue_is_a_pvalue():
    """p must lie in (0, 1].

    Regression guard: the raw reciprocal 1/Lambda exceeds 1 wherever Lambda < 1
    (data favouring the null), which is not a p-value. S = 0 is the extreme case:
    it is the weakest possible evidence, so p must clamp to exactly 1.
    """
    S = np.array([0.0, 3.0, -2.0, 10.0, 0.5])
    t = np.arange(1, 6)
    p = m.always_valid_pvalue(S, t, rho=0.5)
    assert np.all(p > 0.0)
    assert np.all(p <= 1.0), "p-value above 1 means the clamp is missing"
    # no evidence at all -> p == 1 exactly
    assert float(m.always_valid_pvalue(0.0, 25, 0.5)) == 1.0
    # more evidence (larger |S|) at the same t must give a smaller p
    assert m.always_valid_pvalue(10.0, 25, 0.5) < m.always_valid_pvalue(1.0, 25, 0.5)
