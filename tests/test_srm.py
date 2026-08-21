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


def test_expected_ratio_must_sum_to_one():
    with pytest.raises(ValueError):
        srm_test.srm_test([5000, 5000], [0.4, 0.5])


def test_three_arm_clean_split():
    r = srm_test.srm_test([5000, 5000, 5000], [1 / 3, 1 / 3, 1 / 3])
    assert not r["srm_detected"]
