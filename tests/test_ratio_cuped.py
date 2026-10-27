# -*- coding: utf-8 -*-
import delta_method_ratio as dmr
import numpy as np
import ratio_cuped as rc


def test_type1_calibrated():
    n_sims = 300
    rejects = 0
    for s in range(n_sims):
        x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        rejects += rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)[
            "significant"
        ]
    rate = rejects / n_sims
    assert 0.015 <= rate <= 0.09, f"Type I = {rate:.3f}, expected ~0.05"


def test_coverage():
    n_sims = 300
    covered = 0
    for s in range(n_sims):
        x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        res = rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)
        covered += res["ci_low"] <= 0 <= res["ci_high"]
    rate = covered / n_sims
    assert 0.90 <= rate <= 0.99, f"coverage = {rate:.3f}, expected ~0.95"


def test_known_lift_is_positive():
    """B87: pin the sign of the effect.

    `significant`, `ci` and `se` are all invariant to flipping `diff`: the CI is
    centred on diff, so it still straddles or excludes zero exactly as before.
    Only asserting the sign catches a test that reports the effect backwards.
    """
    x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=1, rel_lift=0.0)
    x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=2, rel_lift=0.10)
    res = rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)
    # unadjusted B - A: the fixture must show the lift before the test can pin it
    raw = yb.mean() / xb.mean() - ya.mean() / xa.mean()
    assert raw > 0, f"fixture degenerate: raw ratio lift = {raw}"
    assert res["diff"] > 0, f"diff = {res['diff']}, expected +10% lift in B"
    assert res["z"] > 0
    assert np.sign(res["diff"]) == np.sign(raw)
    # the estimator is roughly the raw difference scaled by 1/mean(X); a sign
    # flip keeps the magnitude, so bound it too
    assert 0.5 * raw < res["diff"] < 2.0 * raw, f"diff={res['diff']:.5f}, raw={raw:.5f}"
    assert res["ci_low"] > 0  # a +10% lift is far from the null here


def test_constant_pre_period_falls_back_to_no_adjustment():
    """A pre-period with zero variance is 0/0 for theta.

    Users with no pre-period activity (zero clicks on constant impressions) give
    z_pre identical for every row, so var(z_pre) == 0 and the ratio was nan --
    silently, because the modules swallow warnings on import. The guard must fall
    back to the unadjusted delta method.
    """
    rng = np.random.default_rng(5)
    n = 400
    x_a = np.full(n, 10.0)
    y_a = rng.binomial(10, 0.10, n).astype(float)
    x_b = np.full(n, 10.0)
    y_b = rng.binomial(10, 0.10, n).astype(float)
    res = rc.ratio_cuped_test(
        x_a,
        y_a,
        x_b,
        y_b,
        np.full(n, 10.0),
        np.zeros(n),
        np.full(n, 10.0),
        np.zeros(n),
    )
    assert res["theta"] == 0.0
    assert res["r_squared"] == 0.0
    for key in ("diff", "se", "p_value", "ci_low", "ci_high"):
        assert np.isfinite(res[key]), f"{key} = {res[key]}"


def test_variance_reduction_vs_delta_method():
    n_sims = 100
    se_delta, se_cuped = [], []
    for s in range(n_sims):
        x_pre_a, y_pre_a, xa, ya = rc.simulate_ratio_cuped(seed=10_000 + s, rel_lift=0.0)
        x_pre_b, y_pre_b, xb, yb = rc.simulate_ratio_cuped(seed=20_000 + s, rel_lift=0.0)
        se_delta.append(dmr.delta_method_test(xa, ya, xb, yb)["se"])
        se_cuped.append(
            rc.ratio_cuped_test(xa, ya, xb, yb, x_pre_a, y_pre_a, x_pre_b, y_pre_b)["se"]
        )
    reduction = 1 - np.mean(se_cuped) ** 2 / np.mean(se_delta) ** 2
    assert reduction > 0.05, f"variance reduction = {reduction:.3f}"
