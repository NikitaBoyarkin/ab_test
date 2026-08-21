# -*- coding: utf-8 -*-
"""Novelty & primacy effect detection.

A measured effect that changes over the experiment window often reflects user
behaviour, not the treatment:
  - Novelty effect: treatment looks great early, then fades (users click the new
    shiny button, then settle).
  - Primacy effect: treatment looks flat early, then grows (users need time to
    learn the new flow).

Detect both with a treatment x day interaction:
    y ~ treat + day + treat:day

The sign of the treat:day coefficient tells the direction: negative => novelty
(effect decays with day), positive => primacy (effect grows with day). Report
daily uplift estimates with CIs to see the trajectory.

Only days after the experiment has fully rolled out should be used; exclude the
ramp-up period from the trend.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


def simulate_daily(n_days=21, n_per_day=400, kind="novelty", seed=0):
    """Daily A/B data with a time-varying treatment effect.

    kind:
      novelty: effect starts at +0.8 and decays linearly to ~0
      primacy: effect starts at ~0 and grows linearly to +0.8
      stable: constant +0.4
      none:   no effect (null)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    if kind == "novelty":
        effect = 0.8 * (1 - t / (n_days - 1))
    elif kind == "primacy":
        effect = 0.8 * (t / (n_days - 1))
    elif kind == "stable":
        effect = np.full(n_days, 0.4)
    else:
        effect = np.zeros(n_days)
    rows = []
    for d, eff in zip(t, effect):
        for treat in (0, 1):
            n = n_per_day // 2
            y = 5.0 + eff * treat + rng.normal(0, 2.0, n)
            rows.append(pd.DataFrame({"day": d, "treat": treat, "y": y}))
    return pd.concat(rows, ignore_index=True), effect


def fit_trend(df):
    """OLS y ~ treat * day. Returns the interaction coefficient + p-value, and
    daily uplift estimates with CIs."""
    model = smf.ols("y ~ treat * day", data=df).fit()
    inter = model.params["treat:day"]
    p = model.pvalues["treat:day"]
    daily = []
    for d in sorted(df["day"].unique()):
        sub = df[df["day"] == d]
        a = sub[sub["treat"] == 0]["y"]
        b = sub[sub["treat"] == 1]["y"]
        diff = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        daily.append({"day": d, "uplift": diff, "se": se,
                      "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se})
    return {"inter_coef": inter, "p_value": float(p)}, pd.DataFrame(daily)


def diagnose(result, alpha=0.05):
    if result["p_value"] >= alpha:
        return "no significant time trend -> stable effect (or no effect)"
    return ("novelty (effect decays over time)"
            if result["inter_coef"] < 0 else
            "primacy (effect grows over time)")


def main():
    print("=== Novelty & Primacy Detection ===\n")
    for kind in ["novelty", "primacy", "stable", "none"]:
        df, true_effect = simulate_daily(kind=kind, seed=1)
        res, daily = fit_trend(df)
        first, last = true_effect[0], true_effect[-1]
        print(f"[{kind}] true effect: day0={first:+.2f} -> day20={last:+.2f}")
        print(f"  treat:day coef = {res['inter_coef']:+.4f}  p = {res['p_value']:.4f}")
        print(f"  diagnosis: {diagnose(res)}")
        # print a compact daily-uplift trajectory (every 4th day)
        traj = daily.iloc[::4][["day", "uplift", "ci_low", "ci_high"]]
        print("  daily uplift (every 4th day):")
        for _, r in traj.iterrows():
            bar = ("+" * int(max(0, r["uplift"] * 10)) +
                   "-" * int(max(0, -r["uplift"] * 10)))
            print(f"    day {int(r['day']):>2}  {r['uplift']:+.3f} "
                  f"[{r['ci_low']:+.2f},{r['ci_high']:+.2f}] {bar}")
        print()


if __name__ == "__main__":
    main()
