# -*- coding: utf-8 -*-
import numpy as np

import switchback as sb


def test_cluster_randomized_naive_overrejects():
    rn, rc = sb._calibrate(
        sb.simulate_cluster_randomized, sb.cr_naive, sb.cr_cluster_robust, effect=0.0, n_sims=250
    )
    assert 0.02 <= rc <= 0.09, f"cluster-robust Type I = {rc:.3f}"
    assert rn > 0.15, f"naive Type I = {rn:.3f}, should over-reject"


def test_switchback_naive_conservative():
    rn, rc = sb._calibrate(
        sb.simulate_switchback, sb.sw_naive, sb.sw_cluster_robust, effect=0.0, n_sims=250
    )
    assert 0.02 <= rc <= 0.09, f"cluster-robust Type I = {rc:.3f}"
    assert rn <= rc + 0.02, f"naive Type I = {rn:.3f}, should be <= {rc:.3f}"


def test_switchback_cluster_robust_se_recovers_power():
    df = sb.simulate_switchback(effect=0.30, seed=1)
    _, se_naive, _ = sb.sw_naive(df)
    _, se_cluster, _ = sb.sw_cluster_robust(df)
    assert se_cluster <= se_naive, (
        "within-cluster correlation cancels in the contrast -> cluster-robust SE "
        "should be no larger than naive"
    )


def test_carryover_biases_point_estimate():
    ests = []
    for s in range(150):
        d = sb.simulate_switchback(effect=0.0, carryover=0.5, block_size=2, seed=s)
        est, _, _ = sb.sw_cluster_robust(d)
        ests.append(est)
    assert abs(np.mean(ests)) > 0.05, (
        "carryover leaks the prior period's treatment into the outcome -> biased estimate"
    )
