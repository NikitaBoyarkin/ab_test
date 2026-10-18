# -*- coding: utf-8 -*-
"""Example 16 — Novelty effect: the lift depends on which days you report.

Scenario
--------
A 21-day checkout redesign test, ~400 users/day (200 per arm). By day 3 the
daily uplift is over +0.9 and the team wants to ship. But the lift fades --
users tried the new flow, then reverted to habit. Fit `y ~ treat * day`, read
the interaction, then re-fit on the settled window (days 7-20) to get the
number that holds in production.

Run: uv run python examples/16_novelty_primacy.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

from novelty_primacy import diagnose, fit_trend, simulate_daily  # noqa: E402

df, true_effect = simulate_daily(n_days=21, n_per_day=400, kind="novelty", seed=7)

result, daily = fit_trend(df)
print("=== 16 - Novelty: which days do you report? ===\n")
print(f"true effect    : day 0 {true_effect[0]:+.2f} -> day 20 {true_effect[-1]:+.2f}")
print(f"treat:day coef : {result['inter_coef']:+.4f}   p = {result['p_value']:.2g}")
print(f"diagnosis      : {diagnose(result)}\n")

print(f"{'day':>4}{'uplift':>9}{'95% CI':>20}")
for d in (0, 3, 7, 10, 20):
    row = daily[daily["day"] == d].iloc[0]
    print(f"{d:>4}{row['uplift']:>+9.3f}   [{row['ci_low']:>+6.2f}, {row['ci_high']:>+6.2f}]")

# The headline average is inflated by the novelty spike. Re-fit on the days
# after the spike, where behaviour has settled, and look at the final day.
settled = df[df["day"] >= 7]
settled_result, settled_daily = fit_trend(settled)
final = settled_daily.iloc[-1]
print(f"\nsettled (days 7-20) mean uplift : {settled_daily['uplift'].mean():+.3f}")
print(f"settled treat:day coef          : {settled_result['inter_coef']:+.4f}")
print(f"day 20 CI                       : [{final['ci_low']:+.2f}, {final['ci_high']:+.2f}]")

print("\nVERDICT: novelty. The +0.9 from days 0-3 is a spike, not the effect, so do")
print("         not ship on it. Report only days 7-20, where the mean is ~+0.25 --")
print("         but that window is still trending down and day 20 straddles zero.")
print("         Extend the test until the slope flattens before deciding to ship.")
