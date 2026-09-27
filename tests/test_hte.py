# -*- coding: utf-8 -*-
import heterogeneous_treatment_effects as hte
import numpy as np
import pandas as pd


def test_per_segment_ate_sign_detected():
    df = hte.simulate_het(seed=1)
    _, ate = hte.fit_het(df, ref_segment="desktop")
    assert ate.loc["mobile", "ci_low"] > 0  # mobile works
    assert ate.loc["tablet", "ci_high"] < 0  # tablet backfires
    assert ate.loc["desktop", "ci_low"] <= 0 <= ate.loc["desktop", "ci_high"]


def test_interactions_significant():
    df = hte.simulate_het(seed=1)
    model, _ = hte.fit_het(df, ref_segment="desktop")
    terms = [t for t in model.params.index if "treat:segment" in t]
    assert len(terms) == 2
    # the mobile interaction (works: +0.5 vs desktop 0) is unambiguous; the
    # tablet interaction is noisier because the reference desktop effect is
    # itself estimated with error (see per-segment CI test below).
    mobile = [t for t in terms if "mobile" in t][0]
    assert model.pvalues[mobile] < 0.05
    assert model.params[mobile] > 0
    assert min(model.pvalues[t] for t in terms) < 0.05


def test_pooled_effect_hides_heterogeneity():
    df = hte.simulate_het(seed=1)
    overall = df[df.treat == 1].y.mean() - df[df.treat == 0].y.mean()
    assert abs(overall) < 0.2, "opposite-signed segments cancel in the pooled average"


def _null_het(seed, n_per_seg=600, sd=2.0):
    """Zero-effect three-segment frame: every per-segment CI should cover 0 at 95%."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in ("mobile", "desktop", "tablet"):
        for treat in (0, 1):
            n = n_per_seg // 2
            rows.append(
                pd.DataFrame({"segment": g, "treat": treat, "y": 10.0 + rng.normal(0, sd, n)})
            )
    return pd.concat(rows, ignore_index=True)


def test_per_segment_ci_type1_calibrated():
    """Under a true null the per-segment 95% CI must exclude 0 about 5% of the time.

    This is the guard on the SE source in `fit_het`: a per-cell or otherwise
    mis-scaled SE would over-reject here while every effect-detection test still
    passed.
    """
    n_sims = 600
    excluded = total = 0
    for s in range(n_sims):
        _, ate = hte.fit_het(_null_het(seed=s), ref_segment="desktop")
        for _, r in ate.iterrows():
            total += 1
            excluded += (r["ci_low"] > 0) or (r["ci_high"] < 0)
    rate = excluded / total
    assert 0.02 <= rate <= 0.09, f"per-segment Type I = {rate:.3f}, expect ~0.05"


def test_per_segment_se_comes_from_one_model():
    """Equal-size, homoskedastic cells -> one common model SE for all segments.

    A per-cell SE would not be equal across segments, so this pins that the three
    SEs all come from the single fitted model. Compared with a tolerance because
    the three values agree only to floating-point round-off (~1e-15); an exact
    comparison would catch nothing but that round-off.
    """
    df = hte.simulate_het(seed=1)
    _, ate = hte.fit_het(df, ref_segment="desktop")
    ses = ate["se"].to_numpy()
    assert np.ptp(ses) < 1e-12, f"SEs differ across segments: {list(ate['se'])}"
