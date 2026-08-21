# -*- coding: utf-8 -*-
"""Switchback & cluster-randomized experiments — when clustering matters.

Both designs randomize at the CLUSTER level, so independent draws are clusters,
not the rows of your dataset. Whether a clustering-ignorant SE over- or under-
rejects depends on whether treatment varies *within* a cluster:

  1. Cluster-randomized (whole cluster gets one arm, constant over time):
     positive within-cluster correlation INFLATES the treated-vs-control
     contrast (no cancellation -> the cluster effect enters the difference).
     A clustering-ignorant SE is too small -> over-rejects. Use cluster-robust SE.

  2. Switchback (treatment switches on/off over time within a cluster):
     positive within-cluster correlation CANCELS in the treated-vs-control
     contrast (both arms share the correlated part). A clustering-ignorant SE
     is too large -> conservative, wastes power. Use cluster-robust SE with
     period fixed effects (smaller, correct SE, earlier detection).

Carryover (treatment in period p leaks into p+1) biases the point estimate; SE
cannot fix it -> use a washout period or drop the first period of each cluster.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# ---- 1. cluster-randomized (treatment constant within cluster) -------------
def simulate_cluster_randomized(n_clusters=30, n_periods=14, effect=0.0,
                                cluster_sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    cluster_base = rng.normal(0, cluster_sd, n_clusters)
    period_eff = rng.normal(0, 0.3, n_periods)
    rows = []
    for c in range(n_clusters):
        treat = int(rng.random() < 0.5)            # one assignment per cluster
        for p in range(n_periods):
            y = cluster_base[c] + period_eff[p] + effect * treat + rng.normal(0, 1.0)
            rows.append({"cluster": c, "period": p, "treat": treat, "y": y})
    return pd.DataFrame(rows)


def cr_naive(df):
    res = smf.ols("y ~ treat + C(period)", data=df).fit()
    return res.params["treat"], res.bse["treat"], res.pvalues["treat"]


def cr_cluster_robust(df):
    res = smf.ols("y ~ treat + C(period)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["cluster"]})
    return res.params["treat"], res.bse["treat"], res.pvalues["treat"]


# ---- 2. switchback (treatment switches over time within a cluster) ----------
def simulate_switchback(n_clusters=30, n_periods=14, effect=0.0, ar=0.6,
                        carryover=0.0, block_size=1, seed=0):
    rng = np.random.default_rng(seed)
    period_eff = rng.normal(0, 0.3, n_periods)
    # block_size>1 makes treatment persist across consecutive periods, so the
    # previous period's treatment correlates with the current one -> carryover
    # then biases the effect estimate. block_size=1 re-randomizes every period.
    rows = []
    for c in range(n_clusters):
        eps = np.empty(n_periods)
        eps[0] = rng.normal(0, 1.0)
        for p in range(1, n_periods):
            eps[p] = ar * eps[p - 1] + rng.normal(0, np.sqrt(1 - ar**2))
        prev_treat = 0
        block_treat = 0
        for p in range(n_periods):
            if p % block_size == 0:
                block_treat = int(rng.random() < 0.5)
            treat = block_treat
            y = period_eff[p] + effect * treat + carryover * prev_treat + eps[p]
            rows.append({"cluster": c, "period": p, "treat": treat, "y": y})
            prev_treat = treat
    return pd.DataFrame(rows)


def sw_naive(df):
    res = smf.ols("y ~ treat + C(period)", data=df).fit()
    return res.params["treat"], res.bse["treat"], res.pvalues["treat"]


def sw_cluster_robust(df):
    res = smf.ols("y ~ treat + C(period)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["cluster"]})
    return res.params["treat"], res.bse["treat"], res.pvalues["treat"]


def _calibrate(sim_fn, naive_fn, cr_fn, n_sims=400, alpha=0.05, **kw):
    rn = rc = 0
    for s in range(n_sims):
        d = sim_fn(seed=s, **kw)
        _, _, pn = naive_fn(d)
        _, _, pc = cr_fn(d)
        rn += pn < alpha
        rc += pc < alpha
    return rn / n_sims, rc / n_sims


def main():
    print("=== Switchback & Cluster-Robust Inference ===\n")

    # ---- Design 1: cluster-randomized ----
    df = simulate_cluster_randomized(effect=0.30, seed=1)
    en, sen, _ = cr_naive(df)
    ec, sec, _ = cr_cluster_robust(df)
    print("[1] Cluster-randomized (30 clusters x 14 periods, effect = 0.30)")
    print(f"  {'method':<26} {'estimate':>9} {'SE':>8}")
    print(f"  {'naive (no clustering)':<26} {en:>9.3f} {sen:>8.3f}")
    print(f"  {'cluster-robust':<26} {ec:>9.3f} {sec:>8.3f}")
    print(f"  naive SE ({sen:.3f}) << cluster-robust ({sec:.3f}): effective n is")
    print("  n_clusters, not n_clusters x n_periods -> naive over-rejects.")
    rn, rc = _calibrate(simulate_cluster_randomized, cr_naive, cr_cluster_robust,
                        effect=0.0)
    print(f"  null reject rate: naive {rn*100:.1f}% (inflated) | "
          f"cluster-robust {rc*100:.1f}% (~5% expected)\n")

    # ---- Design 2: switchback ----
    df = simulate_switchback(effect=0.30, seed=1)
    en, sen, _ = sw_naive(df)
    ec, sec, _ = sw_cluster_robust(df)
    print("[2] Switchback (treatment switches within cluster over time, AR(1) shocks)")
    print(f"  {'method':<26} {'estimate':>9} {'SE':>8}")
    print(f"  {'naive (no clustering)':<26} {en:>9.3f} {sen:>8.3f}")
    print(f"  {'cluster-robust + period FE':<26} {ec:>9.3f} {sec:>8.3f}")
    print(f"  naive SE ({sen:.3f}) >= cluster-robust ({sec:.3f}): within-cluster")
    print("  correlation cancels in the contrast, so naive is conservative;")
    print("  cluster-robust recovers the smaller correct SE -> more power.")
    rn, rc = _calibrate(simulate_switchback, sw_naive, sw_cluster_robust, effect=0.0)
    print(f"  null reject rate: naive {rn*100:.1f}% (conservative) | "
          f"cluster-robust {rc*100:.1f}% (~5% expected)\n")

    # ---- Carryover bias ----
    print("[3] Carryover (prior period's treatment leaks into this period's outcome)")
    print(f"  {'carryover':<12} {'effect est':>11} {'(true=0)':>11}")
    for carry in (0.0, 0.2, 0.5):
        ests = []
        for s in range(200):
            d = simulate_switchback(effect=0.0, carryover=carry,
                                     block_size=2, seed=s)
            est, _, _ = sw_cluster_robust(d)
            ests.append(est)
        print(f"  {carry:<12.1f} {np.mean(ests):>11.3f} {0.0:>11}")
    print("  cluster-robust SE controls Type I error but does NOT fix carryover bias.")
    print("  Mitigate: washout period, or drop the first period of each cluster.")


if __name__ == "__main__":
    main()
