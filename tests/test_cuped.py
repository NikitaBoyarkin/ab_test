# -*- coding: utf-8 -*-
import cuped
import numpy as np


def _run_trial(effect, seed, use_cuped):
    xa, ya, xb, yb = cuped.simulate_cuped(n=2000, effect=effect, rho=0.6, seed=seed)
    if not use_cuped:
        return cuped.welch_mean_test(ya, yb)
    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    a = cuped.cuped_adjust(ya, xa, theta)
    b = cuped.cuped_adjust(yb, xb, theta)
    return cuped.welch_mean_test(a, b)


def test_variance_reduction_matches_theory():
    xa, ya, xb, yb = cuped.simulate_cuped(n=3000, effect=0.05, rho=0.6, seed=1)
    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    pre = cuped.welch_mean_test(ya, yb)
    post = cuped.welch_mean_test(
        cuped.cuped_adjust(ya, xa, theta), cuped.cuped_adjust(yb, xb, theta)
    )
    reduction = 1 - (post["var_a"] + post["var_b"]) / (pre["var_a"] + pre["var_b"])
    # DGP links X and Y through a shared latent with corr(X,Y) = rho^2, so the
    # theory reduction is corr(X,Y)^2 = rho^4 = 0.1296.
    assert 0.08 <= reduction <= 0.20, f"reduction={reduction:.2f}, theory ~ rho^4=0.13"


def test_constant_covariate_gives_no_adjustment():
    """A covariate with zero variance carries no information: theta = 0, Y unadjusted.

    Without the guard theta is 0/0 -> nan, and the nan silently propagates into
    the adjusted outcomes and every downstream SE.
    """
    y = np.random.default_rng(0).normal(1.0, 1.0, 200)
    x = np.ones(200)
    theta = cuped.cuped_theta(y, x)
    assert theta == 0.0
    np.testing.assert_allclose(cuped.cuped_adjust(y, x, theta), y)


def test_type1_stays_calibrated():
    n_sims = 200
    t1_naive = t1_cuped = 0
    for s in range(n_sims):
        t1_naive += _run_trial(0.0, s, False)["p_value"] < 0.05
        t1_cuped += _run_trial(0.0, s, True)["p_value"] < 0.05
    assert 0.02 <= t1_naive / n_sims <= 0.08
    assert 0.02 <= t1_cuped / n_sims <= 0.08


def test_cuped_boosts_power():
    n_sims = 300
    pow_naive = pow_cuped = 0
    for s in range(n_sims):
        pow_naive += _run_trial(0.04, s, False)["p_value"] < 0.05
        pow_cuped += _run_trial(0.04, s, True)["p_value"] < 0.05
    # variance reduction is modest (rho^4 ~ 13%), so the power edge is small but
    # consistent; the mechanism itself is pinned by the SE-reduction test above.
    assert pow_cuped >= pow_naive - 0.02
    assert pow_cuped / n_sims > 0.15  # clearly above the 5% null rate
