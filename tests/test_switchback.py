# -*- coding: utf-8 -*-
import numpy as np
import switchback as sb


def test_cluster_randomized_naive_overrejects():
    rn, rc = sb._calibrate(
        sb.simulate_cluster_randomized,
        sb.ols_effect,
        sb.ols_effect_cluster_robust,
        effect=0.0,
        n_sims=250,
    )
    assert 0.02 <= rc <= 0.09, f"cluster-robust Type I = {rc:.3f}"
    assert rn > 0.15, f"naive Type I = {rn:.3f}, should over-reject"


def test_switchback_cluster_robust_type1_calibrated():
    _, rc = sb._calibrate(
        sb.simulate_switchback, sb.ols_effect, sb.ols_effect_cluster_robust, effect=0.0, n_sims=250
    )
    assert 0.02 <= rc <= 0.09, f"cluster-robust Type I = {rc:.3f}"


def test_switchback_se_estimators_agree_under_period_iid():
    """Under i.i.d. period assignment neither estimator is systematically
    conservative.

    Regression guard: this used to assert `se_cluster <= se_naive` on a single
    seed and read the single-seed result as "within-cluster correlation cancels".
    Across seeds the ratio straddles 1 (0.81-1.22 in the 15 seeds measured), so the
    seed was doing the work, not the design.
    """
    ratios = [
        sb.ols_effect_cluster_robust(sb.simulate_switchback(effect=0.30, seed=s))["se"]
        / sb.ols_effect(sb.simulate_switchback(effect=0.30, seed=s))["se"]
        for s in range(1, 11)
    ]
    median = float(np.median(ratios))
    assert 0.85 <= median <= 1.25, f"median cluster/naive SE ratio = {median:.3f}"
    assert 0.7 <= min(ratios) and max(ratios) <= 1.4, ratios


def test_carryover_biases_point_estimate():
    ests = []
    for s in range(150):
        d = sb.simulate_switchback(effect=0.0, carryover=0.5, block_size=2, seed=s)
        ests.append(sb.ols_effect_cluster_robust(d)["coef"])
    assert abs(np.mean(ests)) > 0.05, (
        "carryover leaks the prior period's treatment into the outcome -> biased estimate"
    )
