# -*- coding: utf-8 -*-
import inspect

import bayesian_ab_test as b
import numpy as np
import pytest


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
    # the planted lift is +0.15: checking only "lift_mean > 0" accepts a
    # posterior centred anywhere above zero, including the wrong arm's
    assert d["lift_mean"] == pytest.approx(0.15, abs=0.05)
    assert d["ci95"][0] < d["lift_mean"] < d["ci95"][1]


def test_loss_threshold_is_in_metric_units():
    """The ship/keep-testing cut must be passed in, not hardcoded.

    `loss` is in the metric's own units, so a universal 1e-4 says "ship" for a
    conversion rate and "keep testing" for the same effect expressed in cents.
    """
    pa, pb = b.beta_binomial(s=(120, 150), f=(1880, 1850), n_samples=20_000, seed=1)
    default = b.decide(pa, pb, rope=0.005)
    loss_b = default["expected_loss_choose_b"]
    # the default band is a documented contract, not a number to re-derive
    assert inspect.signature(b.decide).parameters["loss_threshold"].default == 1e-4

    # drive the threshold explicitly rather than sitting on the 1e-4 boundary:
    # this fixture's loss is 9.1e-05, close enough that a Monte-Carlo wobble would
    # flip the verdict and read as a regression
    assert b.decide(pa, pb, rope=0.005, loss_threshold=0.5 * loss_b)["decision"] == (
        "lean B (keep testing)"
    )
    assert b.decide(pa, pb, rope=0.005, loss_threshold=2 * loss_b)["decision"] == "ship B"

    # the same experiment read in cents: the posterior scales, the band must too
    cents = b.decide(pa * 100, pb * 100, rope=0.5)
    assert cents["expected_loss_choose_b"] == pytest.approx(100 * loss_b, rel=0.01)
    assert cents["decision"] == "lean B (keep testing)"
