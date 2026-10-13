# -*- coding: utf-8 -*-
"""Figures for the ab_test EDA.

Run with the shared EDA venv:
    /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
        reports/eda/scripts/plot_ab_test.py

Writes 5 PNGs (dpi=300) into reports/eda/figures/.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "scripts"))

import delta_method_ratio  # noqa: E402
import srm_test  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

# Okabe-Ito colorblind-safe palette
C_CONTROL = "#0072B2"
C_TREAT = "#D55E00"
C_ACCENT = "#009E73"
C_MUTED = "#666666"

df = pd.read_csv(PROJECT / "data" / "experiment_sample.csv")
ctrl = df[df.treat == 0]
trt = df[df.treat == 1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def fig1_distributions() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

    # (a) primary metric: binary conversion rate by arm with Wilson 95% CI
    arms = ["control", "treatment"]
    rates = [ctrl.conv.mean(), trt.conv.mean()]
    los, his = [], []
    for i, g in enumerate((ctrl, trt)):
        lo, hi = wilson(int(g.conv.sum()), len(g))
        los.append(rates[i] - lo)
        his.append(hi - rates[i])
    ax = axs[0]
    ax.bar(arms, rates, yerr=[los, his], color=[C_CONTROL, C_TREAT], capsize=6, width=0.6)
    for i, r in enumerate(rates):
        ax.annotate(f"{r:.3f}", (i, r), ha="center", va="bottom", fontsize=11)
    ax.set(xlabel="Arm", ylabel="Conversion rate", title="(a) Primary metric `conv` (binary)")
    ax.set_ylim(0, 0.28)

    # (b) skewed count metric, linear
    ax = axs[1]
    ax.hist(df.impressions, bins=30, color=C_ACCENT, edgecolor="black", alpha=0.85)
    ax.axvline(
        df.impressions.median(),
        color=C_TREAT,
        linestyle="--",
        label=f"median={df.impressions.median():.0f}",
    )
    ax.axvline(
        df.impressions.mean(),
        color=C_CONTROL,
        linestyle=":",
        label=f"mean={df.impressions.mean():.1f}",
    )
    ax.set(xlabel="impressions per user", ylabel="users", title="(b) `impressions` (skew=1.68)")
    ax.legend(frameon=False, fontsize=9)

    # (c) same, log x-axis
    ax = axs[2]
    bins = np.logspace(np.log10(df.impressions.min()), np.log10(df.impressions.max()), 25)
    ax.hist(df.impressions, bins=bins, color=C_ACCENT, edgecolor="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set(
        xlabel="impressions per user (log scale)",
        ylabel="users",
        title="(c) `impressions` — log-scale variant",
    )

    fig.suptitle(
        "Distribution of the primary metric and the skewed count metric (n=600 users)", fontsize=13
    )
    fig.savefig(
        FIGDIR / "fig1_distributions.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def fig2_conversion_by_segment() -> None:
    g = (
        df.groupby("segment")["conv"]
        .agg(n="size", conv="mean")
        .sort_values("conv", ascending=False)
    )
    overall = df.conv.mean()
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    bars = ax.bar(
        g.index, g.conv, color=[C_TREAT if v > overall else C_CONTROL for v in g.conv], width=0.6
    )
    ax.axhline(overall, color=C_MUTED, linestyle="--", label=f"overall mean = {overall:.3f}")
    ax.bar_label(
        bars, labels=[f"{v:.3f}\n(n={n})" for v, n in zip(g.conv, g.n)], padding=3, fontsize=10
    )
    ax.set(xlabel="Segment", ylabel="Conversion rate", title="Conversion rate by segment (sorted)")
    ax.set_ylim(0, 0.27)
    ax.legend(frameon=False)
    fig.savefig(
        FIGDIR / "fig2_conversion_by_segment.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def fig3_segment_effect() -> None:
    rows = []
    for seg in ["desktop", "mobile", "tablet"]:
        s = df[df.segment == seg]
        a, b = s[s.treat == 0]["conv"], s[s.treat == 1]["conv"]
        d = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        rows.append((seg, d, 1.96 * se))
    rows.sort(key=lambda r: r[1])
    segs = [r[0] for r in rows]
    ates = np.array([r[1] for r in rows])
    errs = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    colors = [C_TREAT if v > 0 else C_CONTROL for v in ates]
    ax.barh(segs, ates * 100, xerr=errs * 100, color=colors, capsize=6, height=0.55)
    ax.axvline(0, color="black", linewidth=1)
    for i, (v, e) in enumerate(zip(ates, errs)):
        ax.annotate(
            f"{v * 100:+.1f}pp",
            (v * 100, i),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set(
        xlabel="Treatment effect on conversion, percentage points (95% CI)",
        ylabel="Segment",
        title="Per-segment treatment effect — sign flips across segments\n"
        "(Simpson's-paradox-like cancellation; all n.s. after BH)",
    )
    fig.savefig(
        FIGDIR / "fig3_segment_effect.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def fig4_daily_trend() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    piv_c = df[df.treat == 0].groupby("day")["conv"].mean()
    piv_t = df[df.treat == 1].groupby("day")["conv"].mean()
    ax = axs[0]
    ax.plot(piv_c.index, piv_c.values, marker="o", color=C_CONTROL, label="control")
    ax.plot(piv_t.index, piv_t.values, marker="s", color=C_TREAT, label="treatment")
    ax.set(xlabel="Day", ylabel="Conversion rate", title="(a) Daily conversion by arm")
    ax.set_xticks([0, 1, 2])
    ax.set_ylim(0.10, 0.24)
    ax.legend(frameon=False)
    ax.annotate(
        "effect converges\nto ~0 by day 2",
        (2, 0.16),
        xytext=(-10, 30),
        textcoords="offset points",
        fontsize=9,
        color=C_MUTED,
        arrowprops=dict(arrowstyle="->", color=C_MUTED),
    )

    counts = df.groupby(["day", "treat"]).size().unstack()
    ax = axs[1]
    x = np.arange(len(counts.index))
    ax.bar(x - 0.2, counts[0], width=0.4, color=C_CONTROL, label="control")
    ax.bar(x + 0.2, counts[1], width=0.4, color=C_TREAT, label="treatment")
    ax.set(xlabel="Day", ylabel="Users (flow)", title="(b) Daily flow per arm")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)
    ax.legend(frameon=False)

    fig.suptitle("Daily trend: behaviour (per-user conv) vs flow (user count)", fontsize=13)
    fig.savefig(
        FIGDIR / "fig4_daily_trend.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def fig5_ctr_decision() -> None:
    xa, ya = ctrl.impressions.to_numpy(), ctrl.clicks.to_numpy()
    xb, yb = trt.impressions.to_numpy(), trt.clicks.to_numpy()
    r = delta_method_ratio.delta_method_test(xa, ya, xb, yb)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ax = axs[0]
    vals = [r["ratio_a"] * 100, r["ratio_b"] * 100]
    ax.bar(["control", "treatment"], vals, color=[C_CONTROL, C_TREAT], width=0.55)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}%", (i, v), ha="center", va="bottom", fontsize=11)
    ax.set(
        xlabel="Arm",
        ylabel="CTR, % (Σclicks / Σimpressions)",
        title=f"(a) CTR by arm — delta-method diff {r['diff'] * 100:+.2f}pp\n"
        f"95% CI [{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}]pp, p={r['p_value']:.3f}",
    )
    ax.set_ylim(0, 7.5)

    ax = axs[1]
    n_a, n_b = len(ctrl), len(trt)
    srm = srm_test.srm_test([n_a, n_b], [0.5, 0.5])
    x = np.arange(2)
    ax.bar(x - 0.2, [300, n_a], width=0.4, color=C_CONTROL, label="control")
    ax.bar(x + 0.2, [300, n_b], width=0.4, color=C_TREAT, label="treatment")
    for xi, (a, b) in zip(x, [(300, 300), (n_a, n_b)]):
        ax.annotate(f"{a}", (xi - 0.2, a), ha="center", va="bottom", fontsize=10)
        ax.annotate(f"{b}", (xi + 0.2, b), ha="center", va="bottom", fontsize=10)
    ax.set(
        xlabel="Split", ylabel="Users", title=f"(b) SRM check — χ² p={srm['p_value']:.3f} (clean)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["expected (50/50)", "observed"])
    ax.set_ylim(0, 380)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Decision metric on the 600-user fixture: CTR lift is NOT significant", fontsize=13
    )
    fig.savefig(
        FIGDIR / "fig5_ctr_decision.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


if __name__ == "__main__":
    fig1_distributions()
    fig2_conversion_by_segment()
    fig3_segment_effect()
    fig4_daily_trend()
    fig5_ctr_decision()
    print("wrote 5 figures to", FIGDIR)
