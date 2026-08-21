# -*- coding: utf-8 -*-
"""Heterogeneous treatment effects — does the effect differ by segment?

A single average treatment effect can hide opposite-signed effects in subgroups
(Simpson's paradox) or miss that the effect only works in one segment. Fit an OLS
with treatment x segment interaction and test each interaction term:

    y ~ treat + segment + treat:segment

The interaction coefficient for segment g is the *difference in treatment effect*
between g and the reference segment. A significant interaction means the effect
is heterogeneous. Per-segment uplift (ATE_g) and its CI come from the same model.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


def simulate_het(n_per_seg=2000, base=10.0, seed=0):
    """Three segments (mobile/desktop/tablet) with different true uplift.

    mobile:  +0.5 (works)
    desktop:  0.0 (nothing)
    tablet: -0.4 (backfires)
    """
    rng = np.random.default_rng(seed)
    segs = ["mobile", "desktop", "tablet"]
    uplift = {"mobile": 0.5, "desktop": 0.0, "tablet": -0.4}
    rows = []
    for g in segs:
        for treat in (0, 1):
            n = n_per_seg // 2
            y = base + uplift[g] * treat + rng.normal(0, 2.0, n)
            rows.append(pd.DataFrame({"segment": g, "treat": treat, "y": y}))
    return pd.concat(rows, ignore_index=True)


def fit_het(df, ref_segment="desktop"):
    """OLS with treatment x segment interaction; return results + per-segment uplift."""
    df = df.copy()
    df["segment"] = pd.Categorical(
        df["segment"],
        categories=[ref_segment] + [s for s in df["segment"].unique() if s != ref_segment],
    )
    model = smf.ols("y ~ treat * segment", data=df).fit()
    # Per-segment ATE = mean(y|treat=1,g) - mean(y|treat=0,g)
    rows = []
    for g in df["segment"].cat.categories:
        sub = df[df["segment"] == g]
        a = sub[sub["treat"] == 0]["y"]
        b = sub[sub["treat"] == 1]["y"]
        diff = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        rows.append(
            {
                "segment": g,
                "ate": diff,
                "se": se,
                "ci_low": diff - 1.96 * se,
                "ci_high": diff + 1.96 * se,
            }
        )
    ate = pd.DataFrame(rows).set_index("segment")
    return model, ate


def main():
    print("=== Heterogeneous Treatment Effects ===\n")
    df = simulate_het(seed=1)
    model, ate = fit_het(df, ref_segment="desktop")

    print("[Per-segment average treatment effect (ATE)]")
    print(f"  {'segment':<10} {'ATE':>8} {'SE':>7} {'95% CI':>20} {'sig':>5}")
    for g, r in ate.iterrows():
        sig = not (r["ci_low"] <= 0 <= r["ci_high"])
        print(
            f"  {str(g):<10} {r['ate']:>8.3f} {r['se']:>7.3f} "
            f"[{r['ci_low']:>7.3f}, {r['ci_high']:>7.3f}] {str(sig):>5}"
        )

    print("\n[OLS interaction terms (diff vs reference 'desktop')]")
    inter = [t for t in model.params.index if "treat:segment" in t]
    for term in inter:
        p = model.pvalues[term]
        star = " <--- heterogeneous" if p < 0.05 else ""
        print(f"  {term:<28} coef={model.params[term]:+.3f}  p={p:.4f}{star}")

    print("\n[Interpretation]")
    print("  Overall ATE (pooled, ignoring segments):")
    overall = df[df.treat == 1].y.mean() - df[df.treat == 0].y.mean()
    print(f"    {overall:+.3f}  <- average of +0.5, 0.0, -0.4 (roughly cancels)")
    print("  The pooled effect is near zero, but per-segment it ranges from -0.4 to +0.5.")
    print("  Significant interaction terms confirm the effect is heterogeneous ->")
    print("  report per-segment, not the average; consider targeting by segment.")


if __name__ == "__main__":
    main()
