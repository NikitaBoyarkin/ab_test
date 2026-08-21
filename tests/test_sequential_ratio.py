# -*- coding: utf-8 -*-

import sequential_ratio as sr


def test_aa_always_valid_calibrated():
    tau, alpha = 0.01, 0.05
    n_sims = 200
    av_false = naive_false = 0
    for s in range(n_sims):
        recs = sr.run_stream(20, 500, 0.05, 0.0, tau, seed=s)
        av_false += min(r["p_av"] for r in recs) <= alpha
        naive_false += min(r["p_naive"] for r in recs) <= alpha
    # the always-valid guarantee is an upper bound (never exceed alpha); being
    # conservative in finite horizons is expected. Naive peeking must inflate.
    assert av_false / n_sims <= 0.09, f"always-valid FWER = {av_false / n_sims:.3f}"
    assert naive_false / n_sims > 0.12, f"naive FWER = {naive_false / n_sims:.3f}"


def test_detects_twenty_pct_lift():
    recs = sr.run_stream(20, 500, 0.05, 0.20, 0.01, seed=5)
    assert min(r["p_av"] for r in recs) <= 0.05


def test_information_grows_with_sample():
    recs = sr.run_stream(5, 500, 0.05, 0.0, 0.01, seed=0)
    infos = [r["I"] for r in recs]
    assert all(infos[i] < infos[i + 1] for i in range(len(infos) - 1))
