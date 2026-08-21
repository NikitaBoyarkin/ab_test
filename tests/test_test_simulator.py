# -*- coding: utf-8 -*-
import numpy as np

import test_simulator as ts


def test_welch_type1_calibrated():
    r0 = ts.simulate(ts.dgp_normal, ts.test_welch, effect=0.0, n_trials=300, seed=1)
    assert 0.02 <= r0["rejection_rate"] <= 0.08


def test_welch_power_high():
    r1 = ts.simulate(ts.dgp_normal, ts.test_welch, effect=0.15, n_trials=300, seed=2)
    assert r1["rejection_rate"] > 0.85


def test_power_curve_monotonic():
    curve = ts.calibration_curve(
        ts.dgp_normal, ts.test_welch, np.linspace(0.0, 0.2, 5), n_trials=150, seed=10
    )
    rates = [r["rejection_rate"] for r in curve]
    assert rates == sorted(rates)


def test_mannwhitney_type1_on_lognormal():
    def dgp_lognormal(effect, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.lognormal(0, 1, n)
        b = rng.lognormal(np.log1p(effect), 1, n)
        return a, b

    r0 = ts.simulate(dgp_lognormal, ts.test_mannwhitney, effect=0.0, n_trials=200, seed=1)
    assert 0.01 <= r0["rejection_rate"] <= 0.09
