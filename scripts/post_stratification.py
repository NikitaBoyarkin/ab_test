# -*- coding: utf-8 -*-
"""Post-stratification for A/B tests.

The naive difference of means is biased when treatment assignment is imbalanced
across strata that predict the outcome (e.g. new vs returning users). The
post-stratified estimator weights within-stratum mean differences by the
stratum's population share, removing the imbalance bias and reducing variance
when the outcome varies across strata.

    ATE_ps = sum_s w_s * (mean(Y | T=1, s) - mean(Y | T=0, s))
    Var(ATE_ps) = sum_s w_s^2 * (Var(Y|s,T=1)/n_1s + Var(Y|s,T=0)/n_0s)

Reference: standard survey post-stratification; Miratrix, Sekhon, Yu (2013),
"Adjusting treatment effect estimates by post-stratification in randomized
experiments".
"""

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

CONF_Z = stats.norm.ppf(0.975)


def naive_ate(treatment, outcome) -> dict:
    """Unadjusted difference of means (biased under stratum imbalance)."""
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome, dtype=float)
    a = outcome[treatment == 0]
    b = outcome[treatment == 1]
    diff = b.mean() - a.mean()
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return {"diff": diff, "se": se}


def stratified_ate(treatment, outcome, strata, weights=None, alpha: float = 0.05) -> dict:
    """Post-stratified ATE with per-stratum effects.

    `weights` maps stratum label -> population share. If None, the observed
    sample proportions are used (still corrects imbalance bias). Each stratum
    must contain both treatment and control observations.
    """
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome, dtype=float)
    strata = np.asarray(strata)

    unique = np.unique(strata)
    if weights is None:
        counts = np.array([(strata == s).sum() for s in unique], dtype=float)
        w = counts / counts.sum()
    else:
        w = np.array([weights[s] for s in unique], dtype=float)
        w = w / w.sum()

    diffs, vars_ = [], []
    for s in unique:
        mask = strata == s
        a = outcome[mask & (treatment == 0)]
        b = outcome[mask & (treatment == 1)]
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"stratum {s!r} needs >=2 obs in each arm")
        diffs.append(b.mean() - a.mean())
        vars_.append(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))

    ate = float(np.sum(w * np.array(diffs)))
    se = float(np.sqrt(np.sum(w**2 * np.array(vars_))))
    z = ate / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "ate": ate,
        "se": se,
        "p_value": p,
        "ci_low": ate - CONF_Z * se,
        "ci_high": ate + CONF_Z * se,
        "significant": p < alpha,
        "strata": {
            str(s): {"diff": d, "se": np.sqrt(v)}
            for s, d, v in zip(unique, diffs, vars_, strict=True)
        },
    }


def simulate_stratified(n=2000, effect=0.0, imbalance=0.0, seed=0):
    """Two strata ("new" / "returning") with different baselines.

    `imbalance` shifts treatment probability toward the "new" stratum
    (0.5 + imbalance vs 0.5 - imbalance), which biases the naive estimator.
    """
    rng = np.random.default_rng(seed)
    is_new = rng.random(n) < 0.4
    base = np.where(is_new, 0.05, 0.30)
    p_treat = np.where(is_new, 0.5 + imbalance, 0.5 - imbalance)
    treatment = (rng.random(n) < p_treat).astype(int)
    p_out = base + effect * treatment
    outcome = (rng.random(n) < p_out).astype(float)
    strata = np.where(is_new, "new", "returning")
    return treatment, outcome, strata


def main():
    print("=== Post-Stratification ===\n")

    # A/A calibration
    n_sims = 300
    rejects = 0
    for s in range(n_sims):
        t, y, st = simulate_stratified(seed=s)
        rejects += stratified_ate(t, y, st)["significant"]
    print(f"[A/A calibration, {n_sims} sims]")
    print(f"  stratified rejection rate: {rejects / n_sims * 100:.1f}%  (expect ~5%)\n")

    # Imbalance correction: effect=0, but naive ATE is biased by imbalance.
    # Show the mean across sims (a single draw is noisy).
    n_sims = 200
    naive_errors, strat_errors = [], []
    for s in range(n_sims):
        t, y, st = simulate_stratified(n=3000, effect=0.0, imbalance=0.3, seed=s)
        naive_errors.append(naive_ate(t, y)["diff"])
        strat_errors.append(stratified_ate(t, y, st)["ate"])
    print(f"[Imbalance correction, effect=0, imbalance=0.3, {n_sims} sims]")
    print(f"  mean naive ATE:      {np.mean(naive_errors):+.5f}  (biased, should be ~0)")
    print(f"  mean stratified ATE: {np.mean(strat_errors):+.5f}  (unbiased)\n")

    # Variance reduction with predictive strata
    t, y, st = simulate_stratified(n=5000, effect=0.02, imbalance=0.0, seed=2)
    naive = naive_ate(t, y)
    strat = stratified_ate(t, y, st)
    reduction = 1 - strat["se"] ** 2 / naive["se"] ** 2
    print("[Variance reduction, effect=0.02, balanced]")
    print(f"  SE naive={naive['se']:.5f} vs SE stratified={strat['se']:.5f}")
    print(f"  variance reduction: {reduction * 100:.1f}%")
    print(f"  stratified ATE={strat['ate']:+.5f} p={strat['p_value']:.4f}")


if __name__ == "__main__":
    main()
