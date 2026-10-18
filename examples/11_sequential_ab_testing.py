# -*- coding: utf-8 -*-
"""Example 11 — Sequential testing: what the early-stopping rule buys you.

Scenario
--------
Checkout conversion is 1.0% and the team wants to detect a 20% relative lift
(1.00% -> 1.20%). A fixed-horizon test at one-sided alpha=0.05, power=80% needs
33,627 users per arm -- six weeks. Evan Miller's sequential rule instead streams
the data and stops on whichever boundary comes first: treatment leads by 56
conversions, or the arms total 808. The analyst's question: how much traffic
does the rule save before the ship/no-ship call, and how often none?

Run: uv run python examples/11_sequential_ab_testing.py
"""

import numpy as np
import pandas as pd
from _pathsetup import add_scripts_to_path

add_scripts_to_path()

from sequential_ab_testing import (  # noqa: E402
    SEQ_SIZE_TABLE,
    aggregate_test_results,
    fixed_sample_size,
    seq_ab_testing,
)

BASE_RATE, MRE, TRIALS = 0.01, 0.20, 400  # 1.0% base, +20% relative lift
rule = SEQ_SIZE_TABLE.loc[SEQ_SIZE_TABLE["min_relative_effect"] == MRE].iloc[0]
n_total, n_ahead = int(rule["n_total_success"]), int(rule["n_success_ahead"])
fixed_per_arm = fixed_sample_size(BASE_RATE, BASE_RATE * MRE)  # one-sided, same error rates

# One rng per trial: reproducible without seeding (or polluting) global RNG state.
trials = pd.DataFrame(
    seq_ab_testing(BASE_RATE, MRE, n_total, n_ahead, rng=np.random.default_rng(10_000 + i))
    for i in range(TRIALS)
)
summary = aggregate_test_results(trials, fixed_per_arm, verbose=False)
users = summary["avg_seq_method_sample_size_over_fixed_sample_method"] * fixed_per_arm
planned = fixed_per_arm * 2 * BASE_RATE
saved = summary["avg_n_total_success"] / planned - 1

print("=== 11 - Sequential rule vs a fixed horizon (1.0% base, 20% relative MDE) ===")
print(f"\nMiller's decision table:\n{SEQ_SIZE_TABLE.to_string(index=False)}")
print(f"\nfixed plan: {fixed_per_arm:,} users/arm -> {fixed_per_arm * 2:,} users, 6 weeks")
print(
    f"\nsequential rule on the {MRE:.0%} row ({n_total} conv total, or B ahead by {n_ahead}), {TRIALS} trials:"
)
print(
    f"  users/arm at stop   : {users:,.0f} vs {fixed_per_arm:,} planned "
    f"({users / fixed_per_arm - 1:+.1%})"
)
print(
    f"  conversions at stop : {summary['avg_n_total_success']:.0f} vs {planned:.0f} "
    f"planned ({saved:+.1%})"
)
print(
    f"  stopped with B ahead: {summary['treatment_group_win_ratio']:.1%}  (power; rule targets 80%)"
)
print(
    f"  needed MORE than the fixed plan: {summary['seq_method_sample_size_larger_ratio']:.1%} of trials"
)
print("\nVERDICT: adopt the sequential rule. It needs ~a quarter fewer users at the")
print("         design error rates, so a 6-week call lands at ~4.5 weeks. Size the")
print("         team for the fixed plan anyway: in ~1 trial in 4 it saves nothing.")
