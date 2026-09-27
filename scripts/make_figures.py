# -*- coding: utf-8 -*-
"""Generate the full matplotlib figure gallery for the A/B toolkit.

One figure per method module, so the calibration story each module proves in
the terminal also exists as a chart. Run the whole gallery with:

    uv run python scripts/make_figures.py

figures land in `plots/` (gitignored). Every figure uses the shared style and
`save_fig` helper from `plotting.py`, so dpi, layout and the colorblind-safe
palette stay consistent. The two plotnine figures in `sequential_ab_testing.py`
are separate and keep their ggplot theme.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bayesian_ab_test  # noqa: E402
import bootstrap_ci  # noqa: E402
import cupac  # noqa: E402
import cuped  # noqa: E402
import delta_method_ratio  # noqa: E402
import group_sequential  # noqa: E402
import heterogeneous_treatment_effects  # noqa: E402
import msprt_always_valid  # noqa: E402
import multiple_comparisons  # noqa: E402
import novelty_primacy  # noqa: E402
import plotting  # noqa: E402
import post_stratification  # noqa: E402
import ratio_cuped  # noqa: E402
import sample_size  # noqa: E402
import sequential_ratio  # noqa: E402
import srm_test  # noqa: E402
import switchback  # noqa: E402
import test_simulator  # noqa: E402
from plotting import (  # noqa: E402
    CB_PALETTE,
    CONTROL_COLOR,
    NEUTRAL_COLOR,
    TREAT_COLOR,
    apply_style,
    new_axes,
    save_fig,
)


def figure_srm() -> Path:
    """Observed vs expected split, and the chi-square null with the statistic."""
    res = srm_test.srm_test([5200, 4800], [0.5, 0.5])
    obs = np.array(res["observed"])
    exp = np.array(res["expected_ratio"]) * obs.sum()

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    x = np.arange(len(obs))
    ax[0].bar(x - 0.2, obs, 0.4, label="observed", color=CONTROL_COLOR)
    ax[0].bar(x + 0.2, exp, 0.4, label="expected", color=NEUTRAL_COLOR)
    ax[0].set(xticks=x, xticklabels=["A", "B"], ylabel="users", title="Observed vs expected split")
    ax[0].legend()

    xs = np.linspace(0, 30, 400)
    ax[1].plot(xs, stats.chi2.pdf(xs, len(obs) - 1), color=CONTROL_COLOR)
    ax[1].axvline(
        res["chi2"],
        color=TREAT_COLOR,
        ls="--",
        label=f"observed χ²={res['chi2']:.0f}\np={res['p_value']:.1e}",
    )
    ax[1].set(xlabel="χ²", ylabel="density", title="SRM chi-square null")
    ax[1].legend()
    return save_fig(fig, "srm_test.png")


def figure_sample_size() -> Path:
    """Power curve and the sample size required for a given MDE."""
    base = 0.10
    ns = np.arange(200, 21001, 200)

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    for i, diff in enumerate((0.01, 0.02, 0.05)):
        power = [sample_size.power_two_proportion(base, diff, int(n)) for n in ns]
        ax[0].plot(ns, power, color=CB_PALETTE[i], label=f"MDE {diff * 100:.0f}pp")
    ax[0].axhline(0.8, color=NEUTRAL_COLOR, ls=":", label="80% power")
    ax[0].set(
        xlabel="n per arm",
        ylabel="power",
        title=f"Power curve (base rate {base:.0%})",
        ylim=(0, 1.02),
    )
    ax[0].legend()

    diffs = np.linspace(0.005, 0.06, 40)
    n_req = [sample_size.two_proportion(base, float(d)) for d in diffs]
    ax[1].plot(diffs * 100, n_req, color=CONTROL_COLOR, marker="o", ms=3)
    ax[1].set(
        xlabel="MDE (pp)",
        ylabel="n per arm",
        title="Required sample size",
        yscale="log",
    )
    return save_fig(fig, "sample_size_power.png")


def figure_cuped() -> Path:
    """Pre/post correlation CUPED exploits, and the SE it shrinks."""
    x_a, y_a, x_b, y_b = cuped.simulate_cuped(n=2000, effect=0.05, rho=0.6, seed=1)
    theta = cuped.cuped_theta(np.concatenate([y_a, y_b]), np.concatenate([x_a, x_b]))
    x_all = np.concatenate([x_a, x_b])
    y_all = np.concatenate([y_a, y_b])
    pre = cuped.welch_mean_test(y_a, y_b)
    post = cuped.welch_mean_test(
        cuped.cuped_adjust(y_a, x_a, theta), cuped.cuped_adjust(y_b, x_b, theta)
    )

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    ax[0].scatter(x_a, y_a, s=6, alpha=0.25, color=CONTROL_COLOR, label="control")
    ax[0].scatter(x_b, y_b, s=6, alpha=0.25, color=TREAT_COLOR, label="treatment")
    grid = np.linspace(x_all.min(), x_all.max(), 50)
    ax[0].plot(
        grid, y_all.mean() + theta * (grid - x_all.mean()), color=NEUTRAL_COLOR, label="CUPED fit"
    )
    ax[0].set(
        xlabel="pre-period outcome X", ylabel="post-period outcome Y", title="CUPED covariate"
    )
    ax[0].legend()

    ax[1].bar([0, 1], [pre["se"], post["se"]], color=[NEUTRAL_COLOR, CONTROL_COLOR])
    ax[1].set(
        xticks=[0, 1],
        xticklabels=["naive", "CUPED"],
        ylabel="SE of the effect",
        title="Standard error shrink",
    )
    reduction = 1 - post["se"] ** 2 / pre["se"] ** 2
    ax[1].text(0.5, post["se"], f"−{reduction:.0%} variance", ha="center", va="bottom")
    return save_fig(fig, "cuped_variance.png")


def figure_cupac() -> Path:
    """SE across naive / single-covariate CUPED / multi-feature CUPAC."""
    x_a, y_a, x_b, y_b = cupac.simulate_cupac(n=3000, effect=0.05, n_features=3, seed=1)
    se_naive = np.sqrt(np.var(y_a, ddof=1) / len(y_a) + np.var(y_b, ddof=1) / len(y_b))

    theta = cuped.cuped_theta(np.concatenate([y_a, y_b]), np.concatenate([x_a[:, 2], x_b[:, 2]]))
    a_cuped = cuped.cuped_adjust(y_a, x_a[:, 2], theta)
    b_cuped = cuped.cuped_adjust(y_b, x_b[:, 2], theta)
    se_cuped = np.sqrt(
        np.var(a_cuped, ddof=1) / len(a_cuped) + np.var(b_cuped, ddof=1) / len(b_cuped)
    )

    res = cupac.cupac_test(y_a, x_a, y_b, x_b)
    labels = ["naive", "CUPED\n1 feature", "CUPAC\n3 features"]
    ses = [se_naive, se_cuped, res["se"]]

    fig, ax = new_axes(figsize=(8, 5))
    bars = ax.bar(labels, ses, color=[NEUTRAL_COLOR, CONTROL_COLOR, TREAT_COLOR])
    ax.bar_label(bars, fmt="%.4f", padding=2)
    ax.set(
        ylabel="SE of the effect",
        title=f"Variance reduction ≈ model R² = {res['r_squared']:.2f}",
        ylim=(0, max(ses) * 1.15),
    )
    return save_fig(fig, "cupac_variance.png")


def figure_delta_method() -> Path:
    """Naive per-unit ratio test over-rejects under the null; delta method does not."""
    n_sims = 500
    delta_rej = naive_rej = 0
    for s in range(n_sims):
        x_a, y_a = delta_method_ratio.simulate_ratio_metric(seed=10_000 + s, rel_lift=0.0)
        x_b, y_b = delta_method_ratio.simulate_ratio_metric(seed=20_000 + s, rel_lift=0.0)
        delta_rej += delta_method_ratio.delta_method_test(x_a, y_a, x_b, y_b)["significant"]
        naive_rej += delta_method_ratio.naive_per_unit_ratio_test(x_a, y_a, x_b, y_b)["significant"]

    rates = [delta_rej / n_sims, naive_rej / n_sims]
    fig, ax = new_axes(figsize=(7.5, 5))
    bars = ax.bar(["delta method", "naive per-unit"], rates, color=[CONTROL_COLOR, TREAT_COLOR])
    ax.bar_label(bars, fmt="{:.1%}", padding=2)
    ax.axhline(0.05, color=NEUTRAL_COLOR, ls="--", label="nominal α = 5%")
    ax.set(
        ylabel="A/A rejection rate",
        title=f"Type I error under the null ({n_sims} sims)",
        ylim=(0, 0.6),
    )
    ax.legend()
    return save_fig(fig, "delta_method.png")


def figure_ratio_cuped() -> Path:
    """Delta-method SE with and without CUPED on the linearized ratio."""
    n_sims = 100
    se_delta, se_cuped = [], []
    for s in range(n_sims):
        x_pre_a, y_pre_a, x_a, y_a = ratio_cuped.simulate_ratio_cuped(seed=10_000 + s)
        x_pre_b, y_pre_b, x_b, y_b = ratio_cuped.simulate_ratio_cuped(seed=20_000 + s)
        se_delta.append(delta_method_ratio.delta_method_test(x_a, y_a, x_b, y_b)["se"])
        se_cuped.append(
            ratio_cuped.ratio_cuped_test(x_a, y_a, x_b, y_b, x_pre_a, y_pre_a, x_pre_b, y_pre_b)[
                "se"
            ]
        )
    mean_delta, mean_cuped = float(np.mean(se_delta)), float(np.mean(se_cuped))
    reduction = 1 - mean_cuped**2 / mean_delta**2

    fig, ax = new_axes(figsize=(7.5, 5))
    bars = ax.bar(
        ["delta only", "CUPED + delta"],
        [mean_delta, mean_cuped],
        color=[NEUTRAL_COLOR, CONTROL_COLOR],
    )
    ax.bar_label(bars, fmt="%.5f", padding=2)
    ax.set(
        ylabel="mean SE of the CTR lift",
        title=f"CUPED on the linearized ratio (−{reduction:.0%} variance)",
        ylim=(0, mean_delta * 1.15),
    )
    return save_fig(fig, "ratio_cuped_se.png")


def figure_post_stratification() -> Path:
    """Naive ATE is biased under strata imbalance; post-stratification removes it."""
    n_sims = 300
    naive, strat = [], []
    for s in range(n_sims):
        t, y, st = post_stratification.simulate_stratified(
            n=3000, effect=0.0, imbalance=0.3, seed=s
        )
        naive.append(post_stratification.naive_ate(t, y)["diff"])
        strat.append(post_stratification.stratified_ate(t, y, st)["ate"])

    fig, ax = new_axes(figsize=(8, 5))
    bins = np.linspace(-0.08, 0.08, 40)
    ax.hist(
        naive,
        bins=bins,
        alpha=0.6,
        color=NEUTRAL_COLOR,
        label=f"naive (mean {np.mean(naive):+.4f})",
    )
    ax.hist(
        strat,
        bins=bins,
        alpha=0.6,
        color=CONTROL_COLOR,
        label=f"stratified (mean {np.mean(strat):+.4f})",
    )
    ax.axvline(0, color=TREAT_COLOR, ls="--", label="true effect = 0")
    ax.set(
        xlabel="estimated ATE",
        ylabel="simulations",
        title="Imbalance bias (effect = 0, imbalance = 0.3)",
    )
    ax.legend()
    return save_fig(fig, "post_stratification.png")


def figure_group_sequential() -> Path:
    """Alpha-spending boundaries by look and the power they preserve."""
    K, alpha, n_paths = 5, 0.05, 20000
    c_pocock, b_pocock = group_sequential.calibrate(
        group_sequential.pocock_shape, K, alpha, n_paths, seed=1
    )
    c_obf, b_obf = group_sequential.calibrate(group_sequential.obf_shape, K, alpha, n_paths, seed=2)
    b_naive = np.full(K, stats.norm.ppf(1 - alpha / 2))
    looks = np.arange(1, K + 1)

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    ax[0].plot(looks, c_pocock * b_pocock, marker="o", color=CONTROL_COLOR, label="Pocock")
    ax[0].plot(looks, c_obf * b_obf, marker="s", color=TREAT_COLOR, label="O'Brien-Fleming")
    ax[0].plot(looks, b_naive, marker="^", ls=":", color=NEUTRAL_COLOR, label="naive 1.96")
    ax[0].set(
        xlabel="interim look", ylabel="|Z| boundary", title="Stopping boundaries", xticks=looks
    )
    ax[0].legend()

    drift = 2.8
    alt = group_sequential.simulate_paths(n_paths, K, drift=drift, seed=123)
    names = ["Pocock", "OBF", "naive"]
    bounds = [c_pocock * b_pocock, c_obf * b_obf, b_naive]
    power = [group_sequential.crossing_rate(alt, b) for b in bounds]
    bars = ax[1].bar(names, power, color=[CONTROL_COLOR, TREAT_COLOR, NEUTRAL_COLOR])
    ax[1].bar_label(bars, fmt="{:.2f}", padding=2)
    ax[1].set(ylabel="power", title=f"Power at effect = {drift} (full info)", ylim=(0, 1.05))
    return save_fig(fig, "group_sequential.png")


def figure_msprt() -> Path:
    """Always-valid p-value vs the peeking-inflated z p-value."""
    rho, alpha, n = 0.5, 0.05, 500
    rng = np.random.default_rng(7)
    x = rng.normal(0.12, 1.0, n)
    S = np.cumsum(x)
    t = np.arange(1, n + 1)
    p_av = msprt_always_valid.always_valid_pvalue(S, t, rho)
    p_naive = msprt_always_valid.naive_z_pvalue(S, t)

    n_streams = 500
    av_false = naive_false = 0
    for s in range(n_streams):
        r = np.random.default_rng(s)
        xs = r.normal(0.0, 1.0, n)
        Ss = np.cumsum(xs)
        ts = np.arange(1, n + 1)
        av_false += (msprt_always_valid.always_valid_pvalue(Ss, ts, rho) <= alpha).any()
        naive_false += (msprt_always_valid.naive_z_pvalue(Ss, ts) <= alpha).any()

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    ax[0].plot(t, p_av, color=CONTROL_COLOR, label="always-valid")
    ax[0].plot(t, p_naive, color=TREAT_COLOR, label="naive z (peeking)")
    ax[0].axhline(alpha, color=NEUTRAL_COLOR, ls="--", label="α = 5%")
    ax[0].set(
        xlabel="observations", ylabel="p-value", title="One stream, effect = 0.12 SD", yscale="log"
    )
    ax[0].legend()

    rates = [av_false / n_streams, naive_false / n_streams]
    bars = ax[1].bar(["always-valid", "naive z"], rates, color=[CONTROL_COLOR, TREAT_COLOR])
    ax[1].bar_label(bars, fmt="{:.1%}", padding=2)
    ax[1].axhline(alpha, color=NEUTRAL_COLOR, ls="--", label="α = 5%")
    ax[1].set(
        ylabel="P(ever p ≤ α)", title=f"Null calibration ({n_streams} streams)", ylim=(0, 0.45)
    )
    ax[1].legend()
    return save_fig(fig, "msprt_always_valid.png")


def figure_sequential_ratio() -> Path:
    """Always-valid vs naive p-value while a CTR stream accumulates."""
    recs = sequential_ratio.run_stream(
        n_batches=20, batch_size=500, base_rate=0.05, rel_lift=0.20, tau=0.01, seed=3
    )
    n = [r["n"] for r in recs]
    p_av = [r["p_av"] for r in recs]
    p_naive = [r["p_naive"] for r in recs]

    fig, ax = new_axes(figsize=(9, 5))
    ax.plot(n, p_av, marker="o", color=CONTROL_COLOR, label="always-valid (mSPRT)")
    ax.plot(n, p_naive, marker="s", color=TREAT_COLOR, label="naive delta-z")
    ax.axhline(0.05, color=NEUTRAL_COLOR, ls="--", label="α = 5%")
    ax.set(
        xlabel="users per group",
        ylabel="p-value",
        title="Sequential ratio metric (+20% CTR lift)",
        yscale="log",
    )
    ax.legend()
    return save_fig(fig, "sequential_ratio.png")


def figure_bayesian() -> Path:
    """Posterior densities of both arms and of the lift with the ROPE."""
    post_a, post_b = bayesian_ab_test.beta_binomial(
        s=(120, 150), f=(1880, 1850), prior=(1, 1), seed=1
    )
    d = bayesian_ab_test.decide(post_a, post_b, rope=0.005)

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    grid = np.linspace(0.03, 0.10, 400)
    ax[0].plot(
        grid, stats.beta.pdf(grid, 1 + 120, 1 + 1880), color=CONTROL_COLOR, label="A 120/2000"
    )
    ax[0].plot(grid, stats.beta.pdf(grid, 1 + 150, 1 + 1850), color=TREAT_COLOR, label="B 150/2000")
    ax[0].set(xlabel="conversion rate θ", ylabel="posterior density", title="Arm posteriors")
    ax[0].legend()

    diff = post_b - post_a
    ax[1].hist(diff, bins=80, color=CONTROL_COLOR, alpha=0.8)
    ax[1].axvspan(-d["rope"], d["rope"], color=NEUTRAL_COLOR, alpha=0.25, label="ROPE ±0.005")
    ax[1].axvline(0, color=TREAT_COLOR, ls="--")
    ax[1].set(
        xlabel="θ_B − θ_A",
        ylabel="posterior samples",
        title=f"Lift: P(B>A) = {d['p_b_better']:.2f}, decision: {d['decision']}",
    )
    ax[1].legend()
    return save_fig(fig, "bayesian_posterior.png")


def figure_bootstrap() -> Path:
    """BCa bootstrap CIs for a skewed metric across statistics."""
    rng = np.random.default_rng(1)
    a = rng.lognormal(mean=2.0, sigma=1.0, size=1000)
    b = rng.lognormal(mean=2.1, sigma=1.0, size=1000)
    stats_map = {
        "mean": np.mean,
        "median": np.median,
        "p90": lambda x: np.quantile(x, 0.9),
    }
    points, los, his = [], [], []
    for fn in stats_map.values():
        r = bootstrap_ci.bootstrap_diff(a, b, statistic=fn, n_boot=2000, seed=1)
        points.append(r["point"])
        los.append(r["ci_low"])
        his.append(r["ci_high"])

    fig, ax = new_axes(figsize=(8, 5))
    y = np.arange(len(points))
    ax.errorbar(
        points,
        y,
        xerr=[np.array(points) - np.array(los), np.array(his) - np.array(points)],
        fmt="o",
        color=CONTROL_COLOR,
        capsize=6,
        ms=8,
    )
    ax.axvline(0, color=NEUTRAL_COLOR, ls="--")
    ax.set(
        yticks=y,
        yticklabels=list(stats_map),
        xlabel="B − A (95% BCa CI)",
        title="Bootstrap CIs for a skewed (lognormal) revenue metric",
    )
    return save_fig(fig, "bootstrap_ci.png")


def figure_hte() -> Path:
    """Forest plot: per-segment ATE with CIs vs the pooled effect."""
    df = heterogeneous_treatment_effects.simulate_het(seed=1)
    _, ate = heterogeneous_treatment_effects.fit_het(df, ref_segment="desktop")
    pooled = df[df.treat == 1].y.mean() - df[df.treat == 0].y.mean()

    fig, ax = new_axes(figsize=(8.5, 5))
    y = np.arange(len(ate))[::-1]
    ax.errorbar(
        ate["ate"],
        y,
        xerr=[ate["ate"] - ate["ci_low"], ate["ci_high"] - ate["ate"]],
        fmt="o",
        color=CONTROL_COLOR,
        capsize=6,
        ms=8,
    )
    ax.axvline(0, color=NEUTRAL_COLOR, ls="--")
    ax.axvline(pooled, color=TREAT_COLOR, ls=":", label=f"pooled ATE = {pooled:+.2f}")
    ax.set(
        yticks=y,
        yticklabels=ate.index,
        xlabel="treatment effect (95% CI)",
        title="Heterogeneous effects by segment",
    )
    ax.legend()
    return save_fig(fig, "hte_forest.png")


def figure_multiple_comparisons() -> Path:
    """FWER under the all-null across naive / Bonferroni / BH."""
    m, n_trials, alpha = 10, 2000, 0.05
    rng = np.random.default_rng(0)
    fwer = {"naive": 0, "bonferroni": 0, "benjamini-hochberg": 0}
    for _ in range(n_trials):
        p = rng.uniform(size=m)
        fwer["naive"] += multiple_comparisons.naive_reject(p, alpha).any()
        fwer["bonferroni"] += multiple_comparisons.bonferroni(p, alpha)[1].any()
        fwer["benjamini-hochberg"] += multiple_comparisons.benjamini_hochberg(p, alpha)[1].any()
    rates = {k: v / n_trials for k, v in fwer.items()}

    fig, ax = new_axes(figsize=(8, 5))
    bars = ax.bar(
        list(rates), list(rates.values()), color=[TREAT_COLOR, CONTROL_COLOR, CB_PALETTE[2]]
    )
    ax.bar_label(bars, fmt="{:.2f}", padding=2)
    ax.axhline(alpha, color=NEUTRAL_COLOR, ls="--", label="α = 0.05")
    ax.set(
        ylabel="family-wise error rate",
        title=f"All-null FWER across {m} tests ({n_trials} draws)",
        ylim=(0, 1.0),
    )
    ax.legend()
    return save_fig(fig, "multiple_comparisons.png")


def figure_novelty_primacy() -> Path:
    """Daily uplift trajectory under novelty, primacy and a stable effect."""
    fig, ax = new_axes(1, 3, figsize=(13, 4.2), sharey=True)
    for i, kind in enumerate(("novelty", "primacy", "stable")):
        df, _ = novelty_primacy.simulate_daily(kind=kind, seed=1)
        res, daily = novelty_primacy.fit_trend(df)
        ax[i].plot(daily["day"], daily["uplift"], marker="o", ms=3, color=CONTROL_COLOR)
        ax[i].fill_between(
            daily["day"], daily["ci_low"], daily["ci_high"], alpha=0.2, color=CONTROL_COLOR
        )
        ax[i].axhline(0, color=NEUTRAL_COLOR, ls="--")
        ax[i].set(
            xlabel="day",
            title=f"{kind}\ntreat×day = {res['inter_coef']:+.3f} (p={res['p_value']:.3f})",
        )
    ax[0].set_ylabel("daily uplift")
    return save_fig(fig, "novelty_primacy.png")


def figure_switchback() -> Path:
    """Naive vs cluster-robust SE, and the carryover bias SE cannot fix."""
    fig, ax = new_axes(1, 2, figsize=(11.5, 4.5))

    cr_df = switchback.simulate_cluster_randomized(effect=0.30, seed=1)
    se_cr_naive = switchback.ols_effect(cr_df)["se"]
    se_cr_robust = switchback.ols_effect_cluster_robust(cr_df)["se"]
    sb_df = switchback.simulate_switchback(effect=0.30, seed=1)
    se_sb_naive = switchback.ols_effect(sb_df)["se"]
    se_sb_robust = switchback.ols_effect_cluster_robust(sb_df)["se"]

    x = np.arange(2)
    ax[0].bar(x - 0.2, [se_cr_naive, se_sb_naive], 0.4, label="naive SE", color=NEUTRAL_COLOR)
    ax[0].bar(
        x + 0.2, [se_cr_robust, se_sb_robust], 0.4, label="cluster-robust SE", color=CONTROL_COLOR
    )
    ax[0].set(
        xticks=x,
        xticklabels=["cluster-randomized", "switchback"],
        ylabel="SE of the effect",
        title="Clustering changes the SE",
    )
    ax[0].legend()

    carries = (0.0, 0.2, 0.5)
    means = []
    for carry in carries:
        ests = [
            switchback.ols_effect_cluster_robust(
                switchback.simulate_switchback(effect=0.0, carryover=carry, block_size=2, seed=s)
            )[0]
            for s in range(120)
        ]
        means.append(float(np.mean(ests)))
    bars = ax[1].bar([str(c) for c in carries], means, color=TREAT_COLOR)
    ax[1].bar_label(bars, fmt="%+.3f", padding=2)
    ax[1].axhline(0, color=NEUTRAL_COLOR, ls="--", label="true effect = 0")
    ax[1].set(
        xlabel="carryover",
        ylabel="mean effect estimate",
        title="Carryover biases the point estimate",
    )
    ax[1].legend()
    return save_fig(fig, "switchback.png")


def figure_test_simulator() -> Path:
    """Welch power curve and Type I comparison vs Mann-Whitney on skewed data."""
    effects = np.linspace(0.0, 0.2, 9)
    rows = test_simulator.calibration_curve(
        test_simulator.dgp_normal, test_simulator.test_welch, effects, n_trials=400, seed=10
    )

    def dgp_lognormal(effect, n=1000, seed=0):
        r = np.random.default_rng(seed)
        return r.lognormal(0, 1, n), r.lognormal(np.log1p(effect), 1, n)

    t1_welch = test_simulator.simulate(
        test_simulator.dgp_normal, test_simulator.test_welch, effect=0.0, n_trials=1000, seed=1
    )
    t1_mw = test_simulator.simulate(
        dgp_lognormal, test_simulator.test_mannwhitney, effect=0.0, n_trials=1000, seed=1
    )

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    ax[0].plot(
        [r["effect"] for r in rows],
        [r["rejection_rate"] for r in rows],
        marker="o",
        color=CONTROL_COLOR,
    )
    ax[0].axhline(0.8, color=NEUTRAL_COLOR, ls=":", label="80% power")
    ax[0].set(xlabel="true effect (SD)", ylabel="rejection rate", title="Welch power curve")
    ax[0].legend()

    bars = ax[1].bar(
        ["Welch\nnormal", "Mann-Whitney\nlognormal"],
        [t1_welch["rejection_rate"], t1_mw["rejection_rate"]],
        color=[CONTROL_COLOR, TREAT_COLOR],
    )
    ax[1].bar_label(bars, fmt="{:.1%}", padding=2)
    ax[1].axhline(0.05, color=NEUTRAL_COLOR, ls="--", label="α = 5%")
    ax[1].set(ylabel="Type I error", title="Null calibration", ylim=(0, 0.2))
    ax[1].legend()
    return save_fig(fig, "test_simulator.png")


def figure_pipeline(df: pd.DataFrame | None = None):
    """Three figures for the end-to-end pipeline report.

    Returns `{name: Path}` so `run_full_pipeline.py` can link them in the
    markdown. Kept separate from the gallery figures because they depend on the
    pipeline's synthetic dataset.
    """
    import run_full_pipeline as rfp

    if df is None:
        df = rfp.generate_data()
    res = rfp.run_pipeline(df)
    ratio, seg, trend = res["ratio"], res["segments"], res["trend"]

    out = {}

    fig, ax = new_axes(1, 2, figsize=(11, 4.5))
    bars = ax[0].bar(
        ["A", "B"], [ratio["ratio_a"], ratio["ratio_b"]], color=[CONTROL_COLOR, TREAT_COLOR]
    )
    ax[0].bar_label(bars, fmt="%.4f", padding=2)
    ax[0].set(ylabel="CTR", title="Click-through rate by arm", ylim=(0, ratio["ratio_b"] * 1.15))

    half = (ratio["ci_high"] - ratio["ci_low"]) / 2
    ax[1].errorbar(
        ratio["diff"],
        0,
        xerr=half,
        fmt="o",
        color=TREAT_COLOR,
        capsize=8,
        ms=10,
    )
    ax[1].axvline(0, color=NEUTRAL_COLOR, ls="--", label="no effect")
    ax[1].set(
        yticks=[],
        xlabel="CTR_B − CTR_A (95% CI)",
        title=f"Lift: {ratio['diff'] * 100:+.2f}pp (p={ratio['p_value']:.4f})",
    )
    ax[1].legend()
    out["pipeline_ctr_ci.png"] = save_fig(fig, "pipeline_ctr_ci.png")

    fig, ax = new_axes(figsize=(8, 4.5))
    y = np.arange(len(seg))[::-1]
    ax.errorbar(seg["ate"], y, xerr=1.96 * seg["se"], fmt="o", color=CONTROL_COLOR, capsize=6, ms=8)
    ax.axvline(0, color=NEUTRAL_COLOR, ls="--")
    ax.set(
        yticks=y,
        yticklabels=[
            f"{s}{'  *' if sig else ''}" for s, sig in zip(seg["segment"], seg["sig"], strict=True)
        ],
        xlabel="segment ATE on conversion (95% CI)",
        title="Segment effects (BH-corrected, * = significant)",
    )
    out["pipeline_segments.png"] = save_fig(fig, "pipeline_segments.png")

    _, daily = novelty_primacy.fit_trend(df.assign(y=df["conv"]))
    fig, ax = new_axes(figsize=(8, 4.5))
    ax.plot(daily["day"], daily["uplift"], marker="o", ms=4, color=CONTROL_COLOR)
    ax.fill_between(daily["day"], daily["ci_low"], daily["ci_high"], alpha=0.2, color=CONTROL_COLOR)
    ax.axhline(0, color=NEUTRAL_COLOR, ls="--")
    ax.set(
        xlabel="day",
        ylabel="daily uplift (conv)",
        title=f"Novelty check: treat×day = {trend['inter_coef']:+.5f} (p={trend['p_value']:.3f})",
    )
    out["pipeline_novelty.png"] = save_fig(fig, "pipeline_novelty.png")
    return out


FIGURE_FUNCTIONS = (
    figure_srm,
    figure_sample_size,
    figure_cuped,
    figure_cupac,
    figure_delta_method,
    figure_ratio_cuped,
    figure_post_stratification,
    figure_group_sequential,
    figure_msprt,
    figure_sequential_ratio,
    figure_bayesian,
    figure_bootstrap,
    figure_hte,
    figure_multiple_comparisons,
    figure_novelty_primacy,
    figure_switchback,
    figure_test_simulator,
)


def main():
    apply_style("light")
    print("=== Generating matplotlib figure gallery ===\n")
    paths = [fn() for fn in FIGURE_FUNCTIONS]
    paths.extend(figure_pipeline().values())
    for path in paths:
        print(f"  wrote {path.name}")
    print(f"\n  {len(paths)} figures written to {plotting.PLOTS_DIR}")


if __name__ == "__main__":
    main()
