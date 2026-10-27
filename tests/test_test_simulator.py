# -*- coding: utf-8 -*-
import bootstrap_ci
import cuped
import numpy as np
import pytest
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


# --- B86: the plugin contract is enforced, not assumed -----------------------
def _renamed_key(a, b, alpha=0.05):
    """A test fn that reports its verdict under another name."""
    return {"significant": True, "p_value": 0.001}


def _no_verdict(a, b, alpha=0.05):
    return {"p_value": 0.5}


def _not_a_dict(a, b, alpha=0.05):
    return True


@pytest.mark.parametrize("test_fn", [_renamed_key, _no_verdict, _not_a_dict])
def test_missing_reject_key_raises(test_fn):
    """A result without 'reject' used to be read as "did not reject", so the
    bench reported a perfectly calibrated 5% for a test that never ran."""
    with pytest.raises(TypeError, match="'reject' key"):
        ts.simulate(ts.dgp_normal, test_fn, effect=0.0, n_trials=3, seed=1)


def test_alpha_is_forwarded_to_the_test_fn():
    seen = []

    def spy(a, b, alpha):
        seen.append(alpha)
        return {"reject": False}

    ts.simulate(ts.dgp_normal, spy, effect=0.0, n_trials=5, alpha=0.123, seed=1)
    assert seen == [0.123] * 5


@pytest.mark.parametrize(
    "test_fn",
    [
        ts.test_welch,
        ts.test_mannwhitney,
        ts.test_bootstrap_mean,
        cuped.welch_mean_test,
        bootstrap_ci.t_test,
    ],
)
def test_module_test_fns_can_be_plugged_in(test_fn):
    """The docstring's "plug in any module's test fn" is the claim under audit:
    every fn here must accept alpha and return 'reject' -- and, once plugged in,
    actually reject at a rate the null can explain. A rate of 0 or 1 means the
    verdict was never read from the function at all."""
    row = ts.simulate(ts.dgp_normal, test_fn, effect=0.0, n_trials=200, alpha=0.05, seed=3)
    # the lower bound is strict on purpose: a function whose verdict is never read
    # rejects zero times out of 200 under a null with a real 5% level
    assert 0.0 < row["rejection_rate"] <= 0.13, (
        f"{test_fn.__name__}: rejection rate {row['rejection_rate']:.3f} on a null DGP"
    )


def test_welch_mean_test_reject_tracks_alpha():
    """p = 0.002 on this fixture: reject at 0.05, keep at 0.001."""
    xa, ya, xb, yb = cuped.simulate_cuped(n=2000, effect=0.1, rho=0.6, seed=1)
    assert cuped.welch_mean_test(ya, yb, alpha=0.05)["reject"] is True
    assert cuped.welch_mean_test(ya, yb, alpha=0.001)["reject"] is False
