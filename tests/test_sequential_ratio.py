# -*- coding: utf-8 -*-

import sequential_ratio as sr


def test_aa_always_valid_calibrated():
    tau, alpha = 0.01, 0.05
    n_sims = 200
    av_false = naive_false = 0
    for s in range(n_sims):
        recs = sr.run_stream(20, 500, 0.05, 0.0, tau, seed=s)
        av_false += min(r["p_av"] for r in recs) <= alpha
        naive_false += min(r["p_naive"] for r in recs) <= alpha
    # the always-valid guarantee is an upper bound (never exceed alpha); being
    # conservative in finite horizons is expected. Naive peeking must inflate.
    assert av_false / n_sims <= 0.09, f"always-valid FWER = {av_false / n_sims:.3f}"
    assert naive_false / n_sims > 0.12, f"naive FWER = {naive_false / n_sims:.3f}"


def test_aa_spend_stays_under_alpha_at_two_horizons():
    """The default prior must not spend more alpha than it is given, at any look.

    Regression guard against two opposite failures, both seen in this file's
    history:

    * a fixed prior scale `tau` in natural units is valid but wasteful — its
      spend sits far below alpha (0.7% at 20 looks);
    * a prior scaled with the information (a point mass at +-c/sqrt(I)) uses the
      alpha but is NOT always-valid: the rejection boundary becomes a constant
      |z| ~ 2.7 curve, so the spend grows with the horizon without bound
      (measured 6.7% at 60 looks and 11.3% at 600, at alpha=5%).

    The default mixture of fixed scales keeps the growing boundary and lands in
    between: ~1.3% at 20 looks, ~2% at 60.
    """
    n_sims, n_batches, batch_size, alpha = 150, 60, 500, 0.05
    spend = {20: 0, 60: 0}
    for s in range(n_sims):
        recs = sr.run_stream(n_batches, batch_size, 0.05, 0.0, seed=s)
        for h in spend:
            if min(r["p_av"] for r in recs[:h]) <= alpha:
                spend[h] += 1
    for h, count in spend.items():
        rate = count / n_sims
        assert rate <= 0.05, f"A/A spend at {h} looks = {rate:.3f} > alpha; prior is not valid"
    assert spend[60] / n_sims >= 0.005, (
        f"A/A spend at 60 looks collapsed to {spend[60] / n_sims:.3f}; "
        "the mixture stopped using the alpha it is allowed"
    )


def test_threshold_in_z_grows_with_information():
    """Always-validity means the rejection boundary must grow with information.

    At fixed |z| the p-value is a decreasing function of information once the
    horizon is long, and the |z| needed to cross alpha rises from ~3.3 SE to
    ~4.3 SE between I=1e7 and I=1e10. A constant boundary of ~2.72 SE is the
    signature of the invalid information-scaled prior and must not come back.
    """

    def crossing(info, alpha=0.05):
        lo, hi = 0.0, 30.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if sr.msprt_ratio(0.0, mid, info)[1] <= alpha:
                hi = mid
            else:
                lo = mid
        return hi

    c7, c8, c10 = crossing(1e7), crossing(1e8), crossing(1e10)
    assert c7 < c8 < c10, f"threshold must grow with information: {c7:.3f}, {c8:.3f}, {c10:.3f}"
    assert c10 > 4.0, f"threshold at I=1e10 is {c10:.3f}; a fixed ~2.7 boundary is not always-valid"

    # The mixture must also beat a single fixed scale at the horizon it is tuned
    # for: otherwise it would be a pointless wrapper around tau=0.01.
    def crossing_fixed(info, tau, alpha=0.05):
        lo, hi = 0.0, 30.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if sr.msprt_ratio(0.0, mid, info, tau)[1] <= alpha:
                hi = mid
            else:
                lo = mid
        return hi

    assert c8 < crossing_fixed(1e8, 0.01), "mixture is less sensitive than the legacy single scale"


def test_p_value_depends_on_information():
    """Same z at three information levels must not give the same p.

    The information-scaled prior this replaces cancelled I out of Lambda, so the
    statistic was a constant-z boundary. With fixed natural-unit scales the p
    value is a function of (z, I), which is what makes it a martingale.
    """
    infos = (1e4, 1e5, 1e7)
    ps = [sr.msprt_ratio(0.0, 3.0, info)[1] for info in infos]
    assert len({round(p, 12) for p in ps}) == 3, f"p must depend on I, got {ps}"
    assert all(0.0 <= p <= 1.0 for p in ps)


def test_detects_twenty_pct_lift():
    recs = sr.run_stream(20, 500, 0.05, 0.20, 0.01, seed=5)
    assert min(r["p_av"] for r in recs) <= 0.05


def test_default_prior_detects_twenty_pct_lift():
    recs = sr.run_stream(20, 500, 0.05, 0.20, seed=5)
    assert min(r["p_av"] for r in recs) <= 0.05


def test_information_grows_with_sample():
    recs = sr.run_stream(5, 500, 0.05, 0.0, 0.01, seed=0)
    infos = [r["I"] for r in recs]
    assert all(infos[i] < infos[i + 1] for i in range(len(infos) - 1))
