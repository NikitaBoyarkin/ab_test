# -*- coding: utf-8 -*-
"""EDA of the committed A/B fixture + README truth-check for the ab_test project.

Run with the shared EDA venv:
    /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
        reports/eda/scripts/analyze_ab_test.py

Writes reports/eda/eda_metrics.json and prints a human-readable summary.
The README's headline numbers are produced by scripts/run_full_pipeline.py on its
own 42,000-user synthetic DGP (seed=7); reproducing them requires statsmodels and
therefore the project venv (see --truth-check). The fixture EDA below is pure
pandas/numpy/scipy and runs in the shared venv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "experiment_sample.csv"
OUT = Path(__file__).resolve().parents[1] / "eda_metrics.json"
sys.path.insert(0, str(PROJECT / "scripts"))

import cuped  # noqa: E402
import delta_method_ratio  # noqa: E402
import multiple_comparisons  # noqa: E402
import sample_size  # noqa: E402
import srm_test  # noqa: E402


def iqr_outliers(s: pd.Series) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())


def norm_p(z: float) -> float:
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def fixture_eda(df: pd.DataFrame) -> dict:
    out: dict = {}
    out["shape"] = list(df.shape)
    out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    out["nulls"] = {c: float(df[c].isna().mean()) for c in df.columns}
    out["full_row_dupes"] = int(df.duplicated().sum())
    out["day_values"] = sorted(df["day"].unique().tolist())
    out["n_days"] = int(df["day"].nunique())

    # Phase 2 — validity
    out["ranges"] = {
        c: [float(df[c].min()), float(df[c].max())]
        for c in ["day", "treat", "conv", "pre_conv", "pre_sessions", "impressions", "clicks"]
    }
    out["binary_ok"] = {c: sorted(df[c].unique().tolist()) for c in ["treat", "conv", "pre_conv"]}
    out["segment_values"] = sorted(df["segment"].unique().tolist())
    out["negative_counts"] = {
        c: int((df[c] < 0).sum()) for c in ["pre_sessions", "impressions", "clicks"]
    }
    out["clicks_le_impressions"] = bool((df["clicks"] <= df["impressions"]).all())

    # Phase 3 — univariate
    num = ["conv", "pre_conv", "pre_sessions", "impressions", "clicks"]
    out["univariate"] = {}
    for c in num:
        s = df[c]
        out["univariate"][c] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "skew": float(s.skew()),
            "iqr_outlier_share": iqr_outliers(s),
        }
    out["segment_cardinality"] = int(df["segment"].nunique())

    # Phase 3 — bivariate
    corr = df[num].corr()
    pairs = []
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            r = float(corr.iloc[i, j])
            if abs(r) >= 0.5:
                pairs.append({"a": num[i], "b": num[j], "r": r})
    out["corr_pairs_r050"] = sorted(pairs, key=lambda d: -abs(d["r"]))
    out["corr_matrix"] = {a: {b: float(corr.loc[a, b]) for b in num} for a in num}

    # Phase 4 — metric by segment
    out["arm_sizes"] = {int(k): int(v) for k, v in df["treat"].value_counts().items()}
    out["conv_by_arm"] = {int(k): float(v) for k, v in df.groupby("treat")["conv"].mean().items()}
    out["conv_by_segment"] = {
        k: {"n": int(v["n"]), "conv": float(v["conv"])}
        for k, v in df.groupby("segment")["conv"].agg(n="count", conv="mean").iterrows()
    }
    xt = df.groupby(["segment", "treat"])["conv"].agg(n="size", conv="mean").reset_index()
    out["conv_by_segment_treat"] = [
        {
            "segment": r.segment,
            "treat": int(r.treat),
            "n": int(r.n),
            "conv": float(r.conv),
        }
        for r in xt.itertuples()
    ]
    # composition check (segment shares by arm) — Simpson guard
    comp = pd.crosstab(df["segment"], df["treat"], normalize="columns")
    out["segment_share_by_arm"] = {
        seg: {"control": float(comp.loc[seg, 0]), "treat": float(comp.loc[seg, 1])}
        for seg in comp.index
    }

    # Phase 5 — daily trend (flow vs behaviour)
    daily = (
        df.groupby(["day", "treat"]).agg(n=("conv", "size"), conv=("conv", "mean")).reset_index()
    )
    out["daily"] = [
        {"day": int(r.day), "treat": int(r.treat), "n": int(r.n), "conv": float(r.conv)}
        for r in daily.itertuples()
    ]
    # novelty: treat x day interaction via OLS (no statsmodels needed)
    y = df["conv"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df["treat"], df["day"], df["treat"] * df["day"]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(df) - X.shape[1]
    s2 = float(resid @ resid) / dof
    se_b = np.sqrt(s2 * np.linalg.inv(X.T @ X)[3, 3])
    t_stat = float(beta[3] / se_b)
    out["novelty_inter_coef"] = float(beta[3])
    out["novelty_se"] = float(se_b)
    t_crit = float(stats.t.ppf(0.975, dof))
    out["novelty_ci"] = [float(beta[3] - t_crit * se_b), float(beta[3] + t_crit * se_b)]
    out["novelty_p"] = float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
    # smallest interaction this fixture could have detected at 80% power
    out["novelty_mde_80"] = float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se_b)

    # per-day Welch diff with its own CI — the reason the "novelty shape" is not identifiable
    daily_diff = []
    for d, sub in df.groupby("day"):
        a = sub.loc[sub["treat"] == 0, "conv"].to_numpy(float)
        b = sub.loc[sub["treat"] == 1, "conv"].to_numpy(float)
        diff = float(b.mean() - a.mean())
        se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
        daily_diff.append(
            {
                "day": int(d),
                "n_control": int(len(a)),
                "n_treat": int(len(b)),
                "diff": diff,
                "se": se,
                "ci": [float(diff - 1.96 * se), float(diff + 1.96 * se)],
            }
        )
    out["daily_diff"] = daily_diff

    # powered-test sizing via the repo's own closed form (two-sided, 80% power)
    base = float(df.loc[df["treat"] == 0, "conv"].mean())
    sizing = []
    for lift in (0.01, 0.0275, 0.05):
        n_arm = int(sample_size.two_proportion(base, lift, one_tail=False))
        sizing.append(
            {
                "lift": lift,
                "base_rate": base,
                "n_per_arm": n_arm,
                "total": 2 * n_arm,
                "days_at_200_per_day": round(2 * n_arm / 200, 1),
            }
        )
    out["sample_size"] = sizing

    # Formal checks reused from the project's own modules (fixture, not README DGP)
    n_a, n_b = out["arm_sizes"].get(0), out["arm_sizes"].get(1)
    srm = srm_test.srm_test([n_a, n_b], [0.5, 0.5])
    out["srm"] = {
        k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in srm.items()
    }

    yy = df["conv"].to_numpy(float)
    xx = df["pre_conv"].to_numpy(float)
    theta = cuped.cuped_theta(yy, xx)
    y_adj = cuped.cuped_adjust(yy, xx, theta)
    m = df["treat"].to_numpy() == 0
    a, b = yy[m], yy[~m]
    aa, bb = y_adj[m], y_adj[~m]
    se_n = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    se_c = float(np.sqrt(np.var(aa, ddof=1) / len(aa) + np.var(bb, ddof=1) / len(bb)))
    out["cuped"] = {
        "theta": float(theta),
        "corr_conv_preconv": float(np.corrcoef(yy, xx)[0, 1]),
        "se_naive": se_n,
        "se_cuped": se_c,
        "var_reduction": 1 - se_c**2 / se_n**2,
    }

    xa = df[df.treat == 0]["impressions"].to_numpy()
    ya = df[df.treat == 0]["clicks"].to_numpy()
    xb = df[df.treat == 1]["impressions"].to_numpy()
    yb = df[df.treat == 1]["clicks"].to_numpy()
    ratio = delta_method_ratio.delta_method_test(xa, ya, xb, yb)
    out["ctr_delta"] = {
        k: (float(v) if isinstance(v, (int, float, np.floating, np.bool_)) else v)
        for k, v in ratio.items()
    }

    seg_rows = []
    for seg in ["mobile", "desktop", "tablet"]:
        s = df[df.segment == seg]
        ca, cb = s[s.treat == 0]["conv"], s[s.treat == 1]["conv"]
        d = float(cb.mean() - ca.mean())
        se = float(np.sqrt(ca.var(ddof=1) / len(ca) + cb.var(ddof=1) / len(cb)))
        seg_rows.append(
            {
                "segment": seg,
                "n_c": int(len(ca)),
                "n_t": int(len(cb)),
                "ate": d,
                "se": se,
                "ci_low": d - 1.96 * se,
                "ci_high": d + 1.96 * se,
                "p": norm_p(d / se),
            }
        )
    seg_df = pd.DataFrame(seg_rows)
    adj, rej = multiple_comparisons.benjamini_hochberg(seg_df["p"].to_numpy())
    seg_df["p_adj"] = adj
    seg_df["sig"] = rej
    out["segment_ate"] = seg_df.to_dict(orient="records")
    return out


def truth_check() -> dict:
    """Reproduce the README headline numbers on the pipeline's 42k synthetic DGP."""
    import run_full_pipeline as P

    df = P.generate_data()
    res = P.run_pipeline(df)
    srm, cup, ratio, seg, trend = (
        res["srm"],
        res["cuped"],
        res["ratio"],
        res["segments"],
        res["trend"],
    )
    import sample_size as ss

    return {
        "n_rows": int(len(df)),
        "n_a": int((df.treat == 0).sum()),
        "n_b": int((df.treat == 1).sum()),
        "srm_p": float(srm["p_value"]),
        "srm_detected": bool(srm["srm_detected"]),
        "cuped_se_naive": float(cup["se_naive"]),
        "cuped_se_cuped": float(cup["se_cuped"]),
        "cuped_var_reduction": float(cup["var_reduction"]),
        "ctr_a": float(ratio["ratio_a"]),
        "ctr_b": float(ratio["ratio_b"]),
        "ctr_diff": float(ratio["diff"]),
        "ctr_p": float(ratio["p_value"]),
        "ctr_sig": bool(ratio["significant"]),
        "seg_p_adj": {r.segment: float(r.p_adj) for r in seg.itertuples()},
        "seg_ate": {r.segment: float(r.ate) for r in seg.itertuples()},
        "novelty_inter": float(trend["inter_coef"]),
        "novelty_p": float(trend["p_value"]),
        "sample_size": {
            "two_proportion_one_tail": ss.two_proportion(0.10, 0.02),
            "two_proportion_two_tail": ss.two_proportion(0.10, 0.02, one_tail=False),
            "two_mean_d02": ss.two_mean(effect_size=0.2, sd=1.0),
        },
    }


def main() -> None:
    df = pd.read_csv(DATA)
    metrics = {"fixture": fixture_eda(df)}
    if "--truth-check" in sys.argv:
        metrics["truth_check"] = truth_check()
    OUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    f = metrics["fixture"]
    print(f"fixture: {f['shape'][0]} rows x {f['shape'][1]} cols; days={f['n_days']}")
    print(f"arm sizes: {f['arm_sizes']} | SRM p={f['srm']['p_value']:.4f}")
    print(
        f"conv by arm: {f['conv_by_arm']} | raw lift={f['conv_by_arm'][1] - f['conv_by_arm'][0]:+.4f}"
    )
    print(
        f"CUPED var reduction={f['cuped']['var_reduction'] * 100:.1f}% (corr={f['cuped']['corr_conv_preconv']:.3f})"
    )
    print(
        f"CTR delta: {f['ctr_delta']['ratio_a']:.4f}->{f['ctr_delta']['ratio_b']:.4f} "
        f"diff={f['ctr_delta']['diff']:+.5f} p={f['ctr_delta']['p_value']:.4f} sig={f['ctr_delta']['significant']}"
    )
    print(
        f"novelty treat:day={f['novelty_inter_coef']:+.4f} se={f['novelty_se']:.4f} "
        f"CI=[{f['novelty_ci'][0]:+.4f},{f['novelty_ci'][1]:+.4f}] p={f['novelty_p']:.4f} "
        f"MDE80={f['novelty_mde_80']:.4f}"
    )
    print(
        "daily diffs: "
        + " | ".join(
            f"day{r['day']} {r['diff'] * 100:+.1f}pp "
            f"[{r['ci'][0] * 100:+.1f},{r['ci'][1] * 100:+.1f}]"
            for r in f["daily_diff"]
        )
    )
    print(
        "powered test: "
        + " | ".join(
            f"{r['lift'] * 100:.2f}pp -> {r['n_per_arm']}/arm ({r['days_at_200_per_day']}d)"
            for r in f["sample_size"]
        )
    )
    if "truth_check" in metrics:
        t = metrics["truth_check"]
        print(
            f"TRUTH: SRM {t['n_a']}/{t['n_b']} p={t['srm_p']:.4f}; "
            f"CUPED {t['cuped_var_reduction'] * 100:.1f}%; CTR {t['ctr_a']:.4f}->{t['ctr_b']:.4f} "
            f"({t['ctr_diff'] * 100:+.2f}pp) p={t['ctr_p']:.3g}"
        )


if __name__ == "__main__":
    main()
