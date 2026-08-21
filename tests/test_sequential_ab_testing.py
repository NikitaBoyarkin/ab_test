# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import sequential_ab_testing as sat


def _run_trials(lift, n_trials=400, seed=42):
    np.random.seed(seed)
    rows = [
        sat.seq_ab_testing(
            base_rate=0.01, true_relative_lift_effect=lift, n_total_success=808, n_success_ahead=56
        )
        for _ in range(n_trials)
    ]
    return pd.DataFrame([r for r in rows if r is not None])


def test_seq_rule_type1_about_5pct():
    aa = _run_trials(0.0)
    t1 = aa["if_b_win"].mean()
    assert 0.01 <= t1 <= 0.10, f"Type I = {t1:.3f}, expect ~5%"


def test_seq_rule_power_about_80pct():
    ab = _run_trials(0.2)
    power = ab["if_b_win"].mean()
    assert power > 0.7, f"power = {power:.3f}, expect ~80%"


def test_sample_size_table_monotonic():
    assert len(sat.SEQ_SIZE_TABLE) == 5
    assert sat.SEQ_SIZE_TABLE["n_total_success"].is_monotonic_decreasing
    assert sat.SEQ_SIZE_TABLE["n_success_ahead"].is_monotonic_decreasing


def test_fixed_sample_benchmark_matches_module():
    n = sat.fixed_sample_size(0.01, 0.002)
    assert n > 0
    assert isinstance(n, int)
