# -*- coding: utf-8 -*-
import multiple_comparisons as mc
import numpy as np


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
    # all-null design: every rejection is a false discovery, so FDP == FWER
    # by construction for every method -- BH's FDR guarantee is invisible here
    assert abs(stats["bh"]["fdp"] - stats["bh"]["fwer"]) < 1e-9


def test_bh_controls_fdr_and_beats_bonferroni_power():
    """FDR control is only observable with true alternatives present.

    On a mixed design, the naive baseline inflates FWER badly; Bonferroni holds
    FWER at alpha but burns power; BH holds FDR at alpha while rejecting more of
    the true alternatives -- which is the whole reason to prefer it.
    """
    s = mc._calibrate_mixed(m=20, n_alt=5, effect=2.0, n_trials=2000, seed=0)
    assert s["naive"]["fwer"] > 0.3, "uncorrected FWER should be badly inflated"
    assert s["naive"]["fdp"] > s["bh"]["fdp"], "BH must cut false discoveries vs naive"
    assert s["bonferroni"]["fwer"] <= 0.06, "Bonferroni must hold FWER at alpha"
    assert s["bh"]["fdp"] <= 0.06, "BH must hold FDR at alpha"
    assert s["bh"]["power"] > s["bonferroni"]["power"], "BH must keep more power at the same alpha"


def test_bh_less_conservative_than_bonferroni_with_real_signal():
    # 10 tests, one strong true positive + a few borderline
    p = np.array([0.0001, 0.01, 0.02, 0.03, 0.05, 0.06, 0.1, 0.2, 0.3, 0.4])
    _, rej_b = mc.bonferroni(p)
    _, rej_bh = mc.benjamini_hochberg(p)
    assert rej_bh.sum() >= rej_b.sum()
