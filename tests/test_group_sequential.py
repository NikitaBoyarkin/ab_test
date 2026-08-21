# -*- coding: utf-8 -*-
import numpy as np

import group_sequential as gs


def test_naive_peeking_inflates_type1():
    paths = gs.simulate_paths(20_000, 5, seed=99)
    rate = gs.crossing_rate(paths, np.full(5, 1.96))
    assert rate > 0.07, f"naive peeking FWER = {rate:.3f}, should be inflated"


def test_pocock_and_obf_control_type1():
    for shape in (gs.pocock_shape, gs.obf_shape):
        c, base = gs.calibrate(shape, 5, n_paths=8000, seed=1)
        rate = gs.crossing_rate(gs.simulate_paths(8000, 5, seed=2), c * base)
        assert 0.04 <= rate <= 0.06, f"empirical alpha = {rate:.3f}"


def test_obf_early_boundary_stricter_than_pocock():
    c_p, _ = gs.calibrate(gs.pocock_shape, 5, n_paths=8000, seed=1)
    c_o, base_o = gs.calibrate(gs.obf_shape, 5, n_paths=8000, seed=2)
    assert c_o * base_o[0] > c_p  # OBF first-look boundary is the strictest
    assert abs(c_o * base_o[-1] - 1.96) < 0.15  # final look ~ standard z


def test_power_grows_with_drift():
    paths_lo = gs.simulate_paths(5000, 5, drift=1.5, seed=1)
    paths_hi = gs.simulate_paths(5000, 5, drift=3.0, seed=2)
    c, base = gs.calibrate(gs.obf_shape, 5, n_paths=8000, seed=3)
    assert gs.crossing_rate(paths_hi, c * base) > gs.crossing_rate(paths_lo, c * base)
