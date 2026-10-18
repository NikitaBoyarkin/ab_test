# -*- coding: utf-8 -*-
"""Example 02 — Sizing a test before you launch it.

Scenario
--------
The growth team wants two readouts from next quarter's onboarding test: an
activation lift (a proportion) and a revenue-per-user lift (a mean). Traffic is
14,000 eligible users per week, split 50/50, and the planning window is 4
weeks. Activation sits at 12% and the team wants +1.5pp; revenue per user has
sd 42 and the team wants +0.80. The question is not what the test will say, but
whether we can afford to ask it.

Run: uv run python examples/02_sample_size.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

from sample_size import power_two_proportion, two_mean, two_proportion  # noqa: E402

WEEKLY_USERS = 14_000
PER_ARM_WEEK = WEEKLY_USERS // 2
BUDGET_WEEKS = 4

# Proportion metric: activation rate, 12% today, +1.5pp is worth shipping.
base_rate, mde_prop = 0.12, 0.015
n_prop = two_proportion(base_rate=base_rate, absolute_diff=mde_prop)
weeks_prop = 2 * n_prop / WEEKLY_USERS
power_prop = power_two_proportion(base_rate, mde_prop, n_prop)

# Mean metric: revenue per user, sd 42, two candidate effects.
sd_rev = 42.0
n_rev = two_mean(effect_size=1.20, sd=sd_rev)
n_rev_tight = two_mean(effect_size=0.80, sd=sd_rev)
weeks_rev = 2 * n_rev / WEEKLY_USERS
weeks_rev_tight = 2 * n_rev_tight / WEEKLY_USERS
need_week = 2 * n_rev_tight / BUDGET_WEEKS

print("=== 02 - How many users per week, and for how long? ===\n")
print(f"traffic {WEEKLY_USERS:,}/week -> {PER_ARM_WEEK:,}/arm; window {BUDGET_WEEKS} weeks\n")
print("activation (proportion), base 12%, MDE +1.5pp")
print(f"  n/arm {n_prop:,}  total {2 * n_prop:,}  {weeks_prop:.2f} wk  power {power_prop:.2f}\n")
print("revenue per user (mean), sd 42")
print(f"  MDE +1.20 : {n_rev:>6,} per arm -> {weeks_rev:.2f} weeks")
print(f"  MDE +0.80 : {n_rev_tight:>6,} per arm -> {weeks_rev_tight:.2f} weeks")
print(f"  +0.80 in {BUDGET_WEEKS} wk needs {need_week:,.0f}/week; we have {WEEKLY_USERS:,}\n")

if weeks_rev_tight <= BUDGET_WEEKS:
    print("VERDICT: both tests fit the window; launch with the planned MDEs.")
else:
    print("VERDICT: launch the activation test -- one week of traffic answers it.")
    print(f"         Revenue at an MDE of +0.80 needs {weeks_rev_tight:.1f} weeks and does not fit")
    print(
        f"         the {BUDGET_WEEKS}-week window. Relax the MDE to +1.20 ({weeks_rev:.1f} weeks) or"
    )
    print("         extend the horizon; do not launch a revenue test sized for +0.80.")
