# -*- coding: utf-8 -*-
import novelty_primacy as npmod


def test_detects_novelty():
    df, _ = npmod.simulate_daily(kind="novelty", seed=1)
    res, _ = npmod.fit_trend(df)
    assert res["inter_coef"] < 0
    assert res["p_value"] < 0.05
    assert "novelty" in npmod.diagnose(res)


def test_detects_primacy():
    df, _ = npmod.simulate_daily(kind="primacy", seed=1)
    res, _ = npmod.fit_trend(df)
    assert res["inter_coef"] > 0
    assert res["p_value"] < 0.05
    assert "primacy" in npmod.diagnose(res)


def test_no_trend_for_stable_and_null():
    for kind in ("stable", "none"):
        df, _ = npmod.simulate_daily(kind=kind, seed=1)
        res, _ = npmod.fit_trend(df)
        assert res["p_value"] >= 0.05, f"{kind}: p={res['p_value']:.3f}"


def test_daily_uplift_trajectory_shape():
    df, true_effect = npmod.simulate_daily(kind="novelty", seed=1)
    _, daily = npmod.fit_trend(df)
    assert daily["uplift"].iloc[0] > daily["uplift"].iloc[-1]  # decaying
