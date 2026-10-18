# Examples

One runnable `.py` example per method module in `scripts/` — no notebooks.

Each example sets up a single realistic scenario on synthetic data with a known
ground truth, prints the numbers, and ends with a plain-text `VERDICT:` line
saying what an analyst should conclude. The scenario is hard-coded, so the
output is reproducible: run it twice, get the same reading.

```bash
uv run python examples/01_srm_test.py
```

`examples/_pathsetup.py` puts the repo root and `scripts/` on `sys.path`; every
example imports it first. Run the examples from anywhere — the shim resolves
paths from its own location.

Not sure where to start? `01 → 05 → 08 → 13` walks the four decisions that
matter most: *is the split clean → can I buy back variance → may I stop early →
is my interval honest.*

---

## Pre-launch & sanity checks

The checks that decide whether any number downstream is worth reading.

| # | Example | Module | Scenario |
|---|---|---|---|
| 01 | [`01_srm_test.py`](01_srm_test.py) | `srm_test.py` | A 0.5% traffic leak from a bucketing bug — invisible in the split percentages, fatal to every metric |
| 02 | [`02_sample_size.py`](02_sample_size.py) | `sample_size.py` | Sizing an A/B test before launch: n per arm for a target MDE |
| 18 | [`18_test_simulator.py`](18_test_simulator.py) | `test_simulator.py` | Calibrating a test before trusting it: empirical Type I error and power curve |

## Ratio metrics

A per-user t-test is biased when the metric is `ΣY/ΣX`. Both examples use CTR.

| # | Example | Module | Scenario |
|---|---|---|---|
| 03 | [`03_delta_method_ratio.py`](03_delta_method_ratio.py) | `delta_method_ratio.py` | CTR as a ratio: the naive per-user t-test gets the SE wrong |
| 04 | [`04_ratio_cuped.py`](04_ratio_cuped.py) | `ratio_cuped.py` | CUPED on that same ratio — linearize, adjust, delta-method scale; SE shrinks further |

## Variance reduction

Buying sensitivity with pre-period data instead of more traffic.

| # | Example | Module | Scenario |
|---|---|---|---|
| 05 | [`05_cuped.py`](05_cuped.py) | `cuped.py` | The canonical CUPED: a correlated pre-period covariate buys back sample |
| 06 | [`06_cupac.py`](06_cupac.py) | `cupac.py` | CUPAC: let a model use every pre-period column, not just one |
| 07 | [`07_post_stratification.py`](07_post_stratification.py) | `post_stratification.py` | Post-stratification when the arms drift on a segment |

## Sequential & always-valid

What it costs to look at the data before the planned horizon.

| # | Example | Module | Scenario |
|---|---|---|---|
| 08 | [`08_group_sequential.py`](08_group_sequential.py) | `group_sequential.py` | Five planned looks under one 5% alpha budget (Pocock / O'Brien-Fleming) |
| 09 | [`09_msprt_always_valid.py`](09_msprt_always_valid.py) | `msprt_always_valid.py` | mSPRT: an always-valid p-value the PM may read any morning |
| 10 | [`10_sequential_ratio.py`](10_sequential_ratio.py) | `sequential_ratio.py` | Always-valid monitoring for a **ratio** metric — delta-method + mSPRT |
| 11 | [`11_sequential_ab_testing.py`](11_sequential_ab_testing.py) | `sequential_ab_testing.py` | What the early-stopping rule actually buys you, in sample saved |

## Bayesian

| # | Example | Module | Scenario |
|---|---|---|---|
| 12 | [`12_bayesian_ab_test.py`](12_bayesian_ab_test.py) | `bayesian_ab_test.py` | Price the risk instead of reading a p-value: P(B>A), expected loss, ROPE |

## Robustness & heterogeneity

Where the average effect lies, and how to stop believing it.

| # | Example | Module | Scenario |
|---|---|---|---|
| 13 | [`13_bootstrap_ci.py`](13_bootstrap_ci.py) | `bootstrap_ci.py` | Bootstrap CIs when the normal-theory interval is useless (skewed / median metrics) |
| 14 | [`14_heterogeneous_treatment_effects.py`](14_heterogeneous_treatment_effects.py) | `heterogeneous_treatment_effects.py` | The average that hides a loss — interaction model across segments |
| 15 | [`15_multiple_comparisons.py`](15_multiple_comparisons.py) | `multiple_comparisons.py` | 12 segment cuts, 3 different answers: naive vs Bonferroni (FWER) vs BH (FDR) |
| 16 | [`16_novelty_primacy.py`](16_novelty_primacy.py) | `novelty_primacy.py` | The lift depends on which days you report — treat×day interaction |
| 17 | [`17_switchback.py`](17_switchback.py) | `switchback.py` | Cluster-robust SE for switchback designs, and why carryover still biases the estimate |

---

## Examples vs. module demos

Every `scripts/*.py` also has a `main()` that prints its own calibration. The
examples are not a replacement for those — they are narrower: **one scenario, one
decision.** If you want the calibration table, run the module; if you want "what
would I do with this on Monday", read the example.

## See also

- Root [`README.md`](../README.md) — the method table and headline results.
- `uv run python scripts/run_full_pipeline.py` — all of the above chained on one
  synthetic experiment, writing `outputs/report.md`.
- `uv run pytest` — the same calibrations as assertions.
