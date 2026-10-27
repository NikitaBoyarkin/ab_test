# -*- coding: utf-8 -*-
import inspect

import make_figures as mf
import matplotlib.pyplot as plt
import plotting


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


def test_gallery_registry_covers_every_figure_function():
    """No figure defined in the module may be missing from the gallery, and no
    registry entry may point at a function that no longer exists.

    Regression guard: this used to assert `len(FIGURE_FUNCTIONS) == 17`, which red
    on every legitimate new figure and green on an orphaned one.
    """
    defined = {
        name
        for name, obj in vars(mf).items()
        if name.startswith("figure_")
        and callable(obj)
        and getattr(obj, "__module__", None) == mf.__name__
        and not inspect.signature(obj).parameters
    }
    registered = {fn.__name__ for fn in mf.FIGURE_FUNCTIONS}
    assert registered == defined, (
        f"unregistered: {sorted(defined - registered)}; stale: {sorted(registered - defined)}"
    )


def test_figure_srm(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "PLOTS_DIR", tmp_path)
    path = mf.figure_srm()
    assert path.name == "srm_test.png"
    assert path.exists() and path.stat().st_size > 0


def test_figure_multiple_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "PLOTS_DIR", tmp_path)
    path = mf.figure_multiple_comparisons()
    assert path.exists() and path.stat().st_size > 0
