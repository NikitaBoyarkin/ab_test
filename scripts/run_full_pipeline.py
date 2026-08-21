# -*- coding: utf-8 -*-
"""End-to-end A/B test pipeline on synthetic data.

Ties together every module in the toolkit into one realistic flow:

    1. generate a synthetic experiment (users, segments, days, impressions)
    2. SRM check        -> did traffic split correctly?           (srm_test)
    3. CUPED            -> variance reduction from pre-period data (cuped)
    4. Ratio metric test-> CTR via delta method                    (delta_method_ratio)
    5. Segments         -> per-segment ATE with BH correction      (heterogeneous_treatment_effects, multiple_comparisons)
    6. Novelty check    -> is the effect decaying over time?       (novelty_primacy)
    7. write markdown report to outputs/

Run:  uv run python scripts/run_full_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cuped
import delta_method_ratio
import multiple_comparisons
import novelty_primacy
import srm_test

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"

SEGMENTS = ["mobile", "desktop", "tablet"]
SEGMENT_UPLIFT = {"mobile": 0.006, "desktop": 0.0, "tablet": -0.004}


def generate_data(n_days=21, n_per_day=2000, effect=0.008, novelty=0.6, seed=7):
    """Synthetic experiment: conversion (binomial) + CTR (ratio) + pre-period.

    `effect` is the treatment lift on conversion probability; `novelty`
    scales a linear decay of that lift over the experiment window (0 = stable,
    >0 = novelty effect that fades).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        eff = effect * (1 - novelty * d / (n_days - 1))  # novelty decay
        for _ in range(n_per_day):
            seg = rng.choice(SEGMENTS, p=[0.4, 0.4, 0.2])
            treat = int(rng.random() < 0.5)
            latent = rng.normal(0, 1)
            p_conv = 1 / (1 + np.exp(-(-3.0 + 0.8 * latent + eff * treat + SEGMENT_UPLIFT[seg])))
            conv = int(rng.random() < p_conv)
            pre_sessions = rng.poisson(3 * np.exp(0.5 * latent))  # correlates w/ conv
            impressions = int(rng.lognormal(3.5, 0.6)) + 1
            ctr = 0.05 * (1 + 0.1 * treat)
            clicks = int(rng.binomial(impressions, ctr))
            rows.append(
                {
                    "day": d,
                    "segment": seg,
                    "treat": treat,
                    "conv": conv,
                    "pre_sessions": pre_sessions,
                    "impressions": impressions,
                    "clicks": clicks,
                }
            )
    return pd.DataFrame(rows)


def _ratio_by_group(df, metric_num="clicks", metric_den="impressions"):
    a = df[df.treat == 0]
    b = df[df.treat == 1]
    return (
        a[metric_den].to_numpy(),
        a[metric_num].to_numpy(),
        b[metric_den].to_numpy(),
        b[metric_num].to_numpy(),
    )


def _segment_ate(df):
    seg_rows = []
    for seg in SEGMENTS:
        sub = df[df.segment == seg]
        a = sub[sub.treat == 0]["conv"]
        b = sub[sub.treat == 1]["conv"]
        diff = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        seg_rows.append(
            {"segment": seg, "ate": diff, "se": se, "p_value": float(_two_sided_p(diff / se))}
        )
    return pd.DataFrame(seg_rows)


def _two_sided_p(z):
    from scipy import stats

    return float(2 * (1 - stats.norm.cdf(abs(z))))


def main():
    print("=== End-to-end A/B pipeline (synthetic data) ===\n")
    df = generate_data()

    print("[1] SRM check (expected 50/50 split)")
    n_a = int((df.treat == 0).sum())
    n_b = int((df.treat == 1).sum())
    srm = srm_test.srm_test([n_a, n_b], [0.5, 0.5])
    print(f"  observed n: A={n_a}, B={n_b} -> SRM={srm['srm_detected']} (p={srm['p_value']:.4g})")

    print("\n[2] CUPED variance reduction (conv, pre-period sessions)")
    cup = cuped_calibrate(df)
    print(
        f"  naive SE={cup['se_naive']:.5f} vs CUPED SE={cup['se_cuped']:.5f} "
        f"-> variance reduction {cup['var_reduction'] * 100:.1f}%"
    )

    print("\n[3] Main decision metric: CTR via delta method")
    x_a, y_a, x_b, y_b = _ratio_by_group(df)
    ratio = delta_method_ratio.delta_method_test(x_a, y_a, x_b, y_b)
    print(
        f"  CTR_A={ratio['ratio_a']:.4f} CTR_B={ratio['ratio_b']:.4f} "
        f"diff={ratio['diff']:+.5f} p={ratio['p_value']:.4f} "
        f"sig={ratio['significant']}"
    )

    print("\n[4] Per-segment ATE on conversion (BH-corrected)")
    seg = _segment_ate(df)
    adj, rejected = multiple_comparisons.benjamini_hochberg(seg["p_value"].to_numpy())
    seg = seg.assign(p_adj=adj, sig=rejected)
    print(seg.to_string(index=False))

    print("\n[5] Novelty check (treat:day interaction)")
    trend, _ = novelty_primacy.fit_trend(df.assign(y=df["conv"]))
    print(f"  treat:day coef = {trend['inter_coef']:+.5f} p = {trend['p_value']:.4f}")
    print(f"  diagnosis: {novelty_primacy.diagnose(trend)}")

    print("\n[6] Writing report -> outputs/report.md")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    write_report(df, srm, cup, ratio, seg, trend, OUTPUTS_DIR / "report.md")


def cuped_calibrate(df):
    """CUPED on conversion using pre-period sessions as the covariate."""
    y = df["conv"].to_numpy(dtype=float)
    x = df["pre_sessions"].to_numpy(dtype=float)
    theta = cuped.cuped_theta(y, x)
    y_adj = cuped.cuped_adjust(y, x, theta)
    mask = df["treat"].to_numpy() == 0
    a, b = y[mask], y[~mask]
    a_adj, b_adj = y_adj[mask], y_adj[~mask]
    se_naive = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    se_cuped = np.sqrt(np.var(a_adj, ddof=1) / len(a_adj) + np.var(b_adj, ddof=1) / len(b_adj))
    return {
        "se_naive": se_naive,
        "se_cuped": se_cuped,
        "var_reduction": 1 - se_cuped**2 / se_naive**2,
    }


def write_report(df, srm, cup, ratio, seg, trend, path):
    conclusion = []
    if srm["srm_detected"]:
        conclusion.append("- FAIL: sample ratio mismatch — do NOT trust downstream tests.")
    else:
        conclusion.append("- PASS: traffic split matches the intended ratio.")
    if ratio["significant"]:
        conclusion.append(
            f"- CTR lift {ratio['diff'] * 100:.2f}pp is significant (p={ratio['p_value']:.3f})."
        )
    else:
        conclusion.append("- CTR difference is not significant.")
    sig_segs = seg[seg["sig"]]["segment"].tolist()
    if sig_segs:
        conclusion.append(
            f"- Heterogeneous effect: significant segments = {sig_segs}. "
            "Consider segment-targeted rollout."
        )
    else:
        conclusion.append("- No significant segment heterogeneity after BH correction.")
    if "novelty" in novelty_primacy.diagnose(trend):
        conclusion.append(
            "- Novelty effect detected: the early lift fades over time; "
            "wait for the stable tail before deciding."
        )

    md = (
        f"""# A/B Experiment Report (synthetic demo)

- **Users:** {len(df):,} ({int((df.treat == 1).sum()):,} treatment / {int((df.treat == 0).sum()):,} control)
- **Window:** {int(df.day.max()) + 1} days
- **Decision metric:** CTR (ratio), secondary: conversion

## 1. Sample Ratio Mismatch
- Observed: A={int(srm["observed"][0])}, B={int(srm["observed"][1])}, p={srm["p_value"]:.4g}
- Detected: {srm["srm_detected"]}

## 2. CUPED variance reduction
- SE naive: {cup["se_naive"]:.5f} → SE CUPED: {cup["se_cuped"]:.5f}
- Variance reduction: {cup["var_reduction"] * 100:.1f}%

## 3. Decision metric (delta-method CTR)
| | CTR_A | CTR_B | diff | p | sig |
|---|---|---|---|---|---|
| CTR | {ratio["ratio_a"]:.4f} | {ratio["ratio_b"]:.4f} | {ratio["diff"]:+.5f} | {ratio["p_value"]:.4f} | {ratio["significant"]} |

## 4. Segment heterogeneity (BH-corrected)
| segment | ATE | p_adj | significant |
|---|---|---|---|
"""
        + "\n".join(
            f"| {r.segment} | {r.ate:+.4f} | {r.p_adj:.4f} | {r.sig} |" for r in seg.itertuples()
        )
        + f"""

## 5. Novelty / primacy
- treat:day = {trend["inter_coef"]:+.5f} (p={trend["p_value"]:.4f})
- Diagnosis: {novelty_primacy.diagnose(trend)}

## 6. Conclusion
{chr(10).join(conclusion)}
"""
    )
    path.write_text(md, encoding="utf-8")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
