# -*- coding: utf-8 -*-
import pytest
import srm_test


def test_clean_50_50_not_flagged():
    r = srm_test.srm_test([5012, 4988], [0.5, 0.5])
    assert not r["srm_detected"]


def test_52_48_mismatch_detected():
    r = srm_test.srm_test([5200, 4800], [0.5, 0.5])
    assert r["srm_detected"]


def test_clean_70_30_not_flagged():
    r = srm_test.srm_test([7000, 3000], [0.7, 0.3])
    assert not r["srm_detected"]


def test_large_sample_tiny_leak_detected():
    r = srm_test.srm_test([505000, 495000], [0.5, 0.5])
    assert r["srm_detected"]


def test_verdict_follows_shipped_alpha_not_five_percent():
    # p = 0.0124 sits between the conventional 0.05 and the shipped 0.001; every
    # other fixture is far enough out to pass with SRM_ALPHA mutated to 0.05.
    r = srm_test.srm_test([5125, 4875], [0.5, 0.5])
    assert r["p_value"] < 0.05
    assert r["alpha"] == srm_test.SRM_ALPHA
    assert not r["srm_detected"]

    assert srm_test.srm_test([5125, 4875], [0.5, 0.5], alpha=0.05)["srm_detected"]


def test_expected_ratio_must_sum_to_one():
    with pytest.raises(ValueError):
        srm_test.srm_test([5000, 5000], [0.4, 0.5])


def test_three_arm_clean_split():
    r = srm_test.srm_test([5000, 5000, 5000], [1 / 3, 1 / 3, 1 / 3])
    assert not r["srm_detected"]
