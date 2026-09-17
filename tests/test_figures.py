# -*- coding: utf-8 -*-
from pathlib import Path

import make_figures as mf
import matplotlib.pyplot as plt
import plotting
import sequential_ab_testing as sat


def test_new_axes_single_and_grid():
    fig, ax = plotting.new_axes()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)

    fig, ax = plotting.new_axes(1, 2, figsize=(6, 3))
    assert len(ax) == 2
    plt.close(fig)


def test_save_fig_writes_and_closes(tmp_path):
    fig, ax = plotting.new_axes()
    ax.plot([0, 1], [1, 0])
    number = fig.number
    path = plotting.save_fig(fig, "smoke.png", out_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert not plt.fignum_exists(number)


def test_gallery_is_wired():
    assert len(mf.FIGURE_FUNCTIONS) == 17
    assert all(callable(fn) for fn in mf.FIGURE_FUNCTIONS)


def test_figure_srm(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "PLOTS_DIR", tmp_path)
    path = mf.figure_srm()
    assert path.name == "srm_test.png"
    assert path.exists() and path.stat().st_size > 0


def test_figure_multiple_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "PLOTS_DIR", tmp_path)
    path = mf.figure_multiple_comparisons()
    assert path.exists() and path.stat().st_size > 0


def test_sequential_plots_dir_is_repo_plots():
    assert sat.PLOTS_DIR == Path(sat.__file__).resolve().parent.parent / "plots"
