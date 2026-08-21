# -*- coding: utf-8 -*-
import sample_size as ss


def test_larger_mde_needs_less_sample():
    assert ss.two_proportion(0.10, 0.02) < ss.two_proportion(0.10, 0.01)


def test_tighter_alpha_needs_more_sample():
    assert ss.two_proportion(0.10, 0.02, alpha=0.01) > ss.two_proportion(0.10, 0.02, alpha=0.05)


def test_two_mean_sample_size():
    assert ss.two_mean(effect_size=0.2, sd=1.0) > ss.two_mean(effect_size=0.5, sd=1.0)


def test_power_rises_with_effect_size():
    p_small = ss.power_two_proportion(0.10, 0.01, 4000)
    p_big = ss.power_two_proportion(0.10, 0.05, 4000)
    assert p_big > p_small


def test_power_at_own_sample_size_about_80pct():
    n = ss.two_proportion(0.10, 0.02)
    p = ss.power_two_proportion(0.10, 0.02, n)
    assert 0.75 <= p <= 0.85
