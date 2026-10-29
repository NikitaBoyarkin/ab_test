# -*- coding: utf-8 -*-
"""Heterogeneous treatment effects — does the effect differ by segment?

A single average treatment effect can hide opposite-signed effects in subgroups
(Simpson's paradox) or miss that the effect only works in one segment. Fit an OLS
with treatment x segment interaction and test each interaction term:

    y ~ treat + segment + treat:segment

The interaction coefficient for segment g is the *difference in treatment effect*
between g and the reference segment. A significant interaction means the effect
is heterogeneous. Per-segment uplift (ATE_g) and its CI come from the same model.

Two families are tested here -- 3 per-segment ATEs and (m - 1) interactions --
so both go through `correct_pvalues`. Benjamini-Hochberg is the default; reading
raw p < 0.05 across 3 segments leaves a family-wise error rate of
1 - (1 - 0.05)^3 = 14%, three times the nominal 5%.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from multiple_comparisons import benjamini_hochberg, bonferroni, naive_reject

warnings.filterwarnings("ignore")

DEFAULT_CORRECTION = "bh"

_CORRECTIONS = {
    "bh": benjamini_hochberg,
    "bonferroni": bonferroni,
    "none": lambda p, alpha=0.05: (np.asarray(p, dtype=float), naive_reject(p, alpha)),
}


def correct_pvalues(pvalues, method=DEFAULT_CORRECTION, alpha=0.05):
    """Apply a multiple-testing correction; returns (p_adj, rejected_at_alpha).

    `bh` (Benjamini-Hochberg, FDR) is the default -- the same correction
    `run_full_pipeline.py` already applies to its segment table. `bonferroni`
    controls FWER more strictly at the cost of power; `none` is the uncorrected
    baseline, kept so the mutation control can show it is the correction, not
    the threshold, that holds FWER down.
    """
    try:
        method_fn = _CORRECTIONS[method]
    except KeyError:
        raise ValueError(
            f"unknown correction {method!r}; choose from {sorted(_CORRECTIONS)}"
        ) from None
    return method_fn(pvalues, alpha)


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


def fit_het(df, ref_segment="desktop", correction=DEFAULT_CORRECTION, alpha=0.05):
    """OLS with treatment x segment interaction; return results + per-segment uplift.

    `correction` selects the multiple-testing correction over the per-segment
    tests ("bh" default, "bonferroni", or "none"). The returned frame keeps the
    raw `p_value`/`sig` beside the adjusted `p_adj`/`sig_adj`, so a decision is
    read off `sig_adj` while the raw columns stay available for before/after.

    `alpha` drives `sig`, `sig_adj` and `p_adj` only: the `ci_low`/`ci_high`
    columns are always the model's 95% intervals. Keep alpha=0.05 when reading
    decisions and CIs together, otherwise the two answer different questions.
    """
    df = df.copy()
    df["segment"] = pd.Categorical(
        df["segment"],
        categories=[ref_segment] + [s for s in df["segment"].unique() if s != ref_segment],
    )
    model = smf.ols("y ~ treat * segment", data=df).fit()
    # The model is saturated, so every per-segment ATE below equals the raw
    # within-cell difference; the SE and CI, however, are read off the fitted
    # model's covariance -- a per-cell Welch SE would contradict the interaction
    # model those CIs are attributed to under heteroskedasticity.
    terms = list(model.params.index)
    rows = []
    for g in df["segment"].cat.categories:
        sub = df[df["segment"] == g]
        diff = sub[sub["treat"] == 1]["y"].mean() - sub[sub["treat"] == 0]["y"].mean()
        contrast = np.zeros(len(terms))
        contrast[terms.index("treat")] = 1.0
        interaction = f"treat:segment[T.{g}]"
        if interaction in terms:
            contrast[terms.index(interaction)] = 1.0
        test = model.t_test(contrast)
        lo, hi = np.squeeze(test.conf_int())
        rows.append(
            {
                "segment": g,
                "ate": diff,
                "se": float(np.squeeze(test.sd)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "p_value": float(np.squeeze(test.pvalue)),
            }
        )
    ate = pd.DataFrame(rows).set_index("segment")
    p_raw = ate["p_value"].to_numpy()
    p_adj, sig_adj = correct_pvalues(p_raw, method=correction, alpha=alpha)
    ate["sig"] = naive_reject(p_raw, alpha)
    ate["p_adj"] = p_adj
    ate["sig_adj"] = sig_adj
    return model, ate


def interaction_effects(model, correction=DEFAULT_CORRECTION, alpha=0.05):
    """Per-interaction coefficient with raw and corrected p-values.

    These are the second tested family: each term asks "does this segment's
    effect differ from the reference", so marking them at raw p < 0.05 while
    inspecting several segments is the same multiplicity problem as above.
    """
    terms = [t for t in model.params.index if "treat:segment" in t]
    p_raw = np.array([float(model.pvalues[t]) for t in terms])
    p_adj, sig_adj = correct_pvalues(p_raw, method=correction, alpha=alpha)
    return pd.DataFrame(
        {
            "coef": [float(model.params[t]) for t in terms],
            "p_value": p_raw,
            "p_adj": p_adj,
            "sig": naive_reject(p_raw, alpha),
            "sig_adj": sig_adj,
        },
        index=pd.Index(terms, name="term"),
    )


def main(correction=DEFAULT_CORRECTION):
    print("=== Heterogeneous Treatment Effects ===\n")
    print(
        f"Multiple-testing correction: {correction.upper()} "
        f"(families: 3 segment ATEs, 2 interactions)\n"
    )
    df = simulate_het(seed=1)
    model, ate = fit_het(df, ref_segment="desktop", correction=correction)

    print("[Per-segment average treatment effect (ATE)]")
    print(
        f"  {'segment':<10} {'ATE':>8} {'SE':>7} {'95% CI':>20} "
        f"{'p':>7} {'p_adj':>7} {'sig_adj':>8}"
    )
    for g, r in ate.iterrows():
        print(
            f"  {str(g):<10} {r['ate']:>8.3f} {r['se']:>7.3f} "
            f"[{r['ci_low']:>7.3f}, {r['ci_high']:>7.3f}] "
            f"{r['p_value']:>7.4f} {r['p_adj']:>7.4f} {str(bool(r['sig_adj'])):>8}"
        )

    print(f"\n[OLS interaction terms (diff vs reference 'desktop'), {correction.upper()} adjusted]")
    inter = interaction_effects(model, correction=correction)
    for term, r in inter.iterrows():
        star = " <--- heterogeneous" if r["sig_adj"] else ""
        print(
            f"  {term:<28} coef={r['coef']:+.3f}  p={r['p_value']:.4f} p_adj={r['p_adj']:.4f}{star}"
        )

    print("\n[Interpretation]")
    print("  Overall ATE (pooled, ignoring segments):")
    overall = df[df.treat == 1].y.mean() - df[df.treat == 0].y.mean()
    print(f"    {overall:+.3f}  <- average of +0.5, 0.0, -0.4 (roughly cancels)")
    print("  The pooled effect is near zero, but per-segment it ranges from -0.4 to +0.5.")
    n_het = int(inter["sig_adj"].sum())
    n_seg = int(ate["sig_adj"].sum())
    if n_het:
        print(
            f"  {n_het}/{len(inter)} interaction terms and {n_seg}/{len(ate)} segment ATEs "
            f"survive {correction.upper()} ->"
        )
        print("  the effect is heterogeneous: report per-segment, not the average;")
        print("  consider targeting by segment.")
    else:
        print(f"  No interaction term survives {correction.upper()} correction; the")
        print("  per-segment spread is consistent with noise -- do not claim heterogeneity.")


if __name__ == "__main__":
    main()
