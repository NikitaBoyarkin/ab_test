# -*- coding: utf-8 -*-
import numpy as np

import msprt_always_valid as m


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
