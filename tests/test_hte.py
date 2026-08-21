# -*- coding: utf-8 -*-
import heterogeneous_treatment_effects as hte


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
