# -*- coding: utf-8 -*-
import cuped
import numpy as np
import pytest


def _run_trial(effect, seed, use_cuped):
    xa, ya, xb, yb = cuped.simulate_cuped(n=2000, effect=effect, rho=0.6, seed=seed)
    if not use_cuped:
        return cuped.welch_mean_test(ya, yb)
    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    x_mean = cuped.pooled_covariate_mean(xa, xb)
    a = cuped.cuped_adjust(ya, xa, theta, x_mean)
    b = cuped.cuped_adjust(yb, xb, theta, x_mean)
    return cuped.welch_mean_test(a, b)


def test_variance_reduction_matches_theory():
    xa, ya, xb, yb = cuped.simulate_cuped(n=3000, effect=0.05, rho=0.6, seed=1)
    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    x_mean = cuped.pooled_covariate_mean(xa, xb)
    pre = cuped.welch_mean_test(ya, yb)
    post = cuped.welch_mean_test(
        cuped.cuped_adjust(ya, xa, theta, x_mean),
        cuped.cuped_adjust(yb, xb, theta, x_mean),
    )
    reduction = 1 - (post["var_a"] + post["var_b"]) / (pre["var_a"] + pre["var_b"])
    # DGP links X and Y through a shared latent with corr(X,Y) = rho^2, so the
    # theory reduction is corr(X,Y)^2 = rho^4 = 0.1296.
    assert 0.08 <= reduction <= 0.20, f"reduction={reduction:.2f}, theory ~ rho^4=0.13"


def test_pooled_centering_keeps_the_estimate_off_naive():
    """Regression (B85): per-arm centering collapses CUPED back to the naive diff.

    With Y_adj = Y - theta * (X - mean(X_arm)), the adjusted mean inside each arm
    is exactly the raw mean (mean(X - mean(X_arm)) == 0), so the point estimate is
    bit-for-bit the naive one while welch_mean_test still reports the reduced
    residual SE. Point estimate and SE then come from different estimators: the
    CI is too narrow and Type I error inflates (~6.6% at nominal 5%, measured
    over 4000 sims). Pooled centering yields the textbook estimator
    diff_naive - theta * (mean(X_b) - mean(X_a)).
    """
    xa, ya, xb, yb = cuped.simulate_cuped(n=2000, effect=0.05, rho=0.6, seed=1)
    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    x_mean = cuped.pooled_covariate_mean(xa, xb)
    naive = cuped.welch_mean_test(ya, yb)
    adjusted = cuped.welch_mean_test(
        cuped.cuped_adjust(ya, xa, theta, x_mean),
        cuped.cuped_adjust(yb, xb, theta, x_mean),
    )
    expected = naive["point"] - theta * (xb.mean() - xa.mean())
    assert adjusted["point"] == pytest.approx(expected, abs=1e-12)
    # the defect: centred per arm, the adjustment vanishes from the point estimate
    assert abs(adjusted["point"] - naive["point"]) > 1e-4
    per_arm = cuped.welch_mean_test(ya - theta * (xa - xa.mean()), yb - theta * (xb - xb.mean()))
    assert per_arm["point"] == pytest.approx(naive["point"], abs=1e-12)
    # the reduced residual SE is only valid paired with the pooled-centred estimate
    assert adjusted["se"] < naive["se"]


def test_known_lift_has_positive_point():
    """B87: pin the sign and the definition of `point`.

    The DGP adds `effect` to arm B only, so mean(b) - mean(a) must be positive.
    Nothing else in the suite reads the sign: p_value, se and the CI width are
    all invariant to `point -> -point`, so a reversed effect would pass.
    """
    xa, ya, xb, yb = cuped.simulate_cuped(n=2000, effect=0.1, rho=0.6, seed=1)
    assert yb.mean() > ya.mean(), "fixture degenerate: B is not above A"
    naive = cuped.welch_mean_test(ya, yb)
    assert naive["point"] > 0
    # pins the ordering: point is mean(b) - mean(a), not the reverse
    assert naive["point"] == pytest.approx(yb.mean() - ya.mean(), abs=1e-12)
    # CUPED must preserve the sign of the effect it removes variance from
    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
    x_mean = cuped.pooled_covariate_mean(xa, xb)
    adjusted = cuped.welch_mean_test(
        cuped.cuped_adjust(ya, xa, theta, x_mean),
        cuped.cuped_adjust(yb, xb, theta, x_mean),
    )
    assert adjusted["point"] > 0
    assert adjusted["point"] == pytest.approx(
        naive["point"] - theta * (xb.mean() - xa.mean()), abs=1e-12
    )


def test_constant_covariate_gives_no_adjustment():
    """A covariate with zero variance carries no information: theta = 0, Y unadjusted.

    Without the guard theta is 0/0 -> nan, and the nan silently propagates into
    the adjusted outcomes and every downstream SE.
    """
    y = np.random.default_rng(0).normal(1.0, 1.0, 200)
    x = np.ones(200)
    theta = cuped.cuped_theta(y, x)
    assert theta == 0.0
    np.testing.assert_allclose(cuped.cuped_adjust(y, x, theta, cuped.pooled_covariate_mean(x)), y)


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
