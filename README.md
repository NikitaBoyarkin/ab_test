# A/B Testing Methodology Toolkit

Empirical, calibration-driven A/B testing methods, implemented from the primary
literature and validated by simulation. Every module ships with an A/A null
check and a power/coverage calibration — the numbers are checked, not assumed.

The guiding idea: a method is only as good as its Type I error under the null
and its power under a real effect. Rather than trusting asymptotic promises,
each module simulates the pipeline end-to-end and reports the empirical rates.

## Quickstart

```bash
uv sync --all-groups          # install deps + dev tools
uv run python scripts/srm_test.py   # run a single module's demo
uv run pytest                 # run the calibration test suite
uv run ruff check .           # lint
uv run python scripts/run_full_pipeline.py   # end-to-end demo -> outputs/report.md
```

Requires Python >= 3.11. Managed with [uv](https://docs.astral.sh/uv/).

## Modules

| Module | Method | What the demo shows | Reference |
|---|---|---|---|
| `scripts/srm_test.py` | Sample Ratio Mismatch (χ²) | catch bucketing/traffic bugs before any downstream test | LukSyen (2019) |
| `scripts/sample_size.py` | Fixed-horizon sizing & power | n/arm for proportions and means | standard two-sample formulas |
| `scripts/delta_method_ratio.py` | Ratio metrics (CTR, RPC) | correct SE for `ΣY/ΣX`; naive per-unit t-test is biased | Deng, Knoblich, Lu (2018) |
| `scripts/ratio_cuped.py` | CUPED for ratio metrics | linearize `Z = Y − R·X`, CUPED on Z, delta-method scale; SE shrinks further | Deng et al. (2013); Deng et al. (2018) |
| `scripts/cuped.py` | Variance reduction | SE shrinks by ~corr(X,Y)² using pre-period data | Deng et al. (2013) |
| `scripts/cupac.py` | Model-based CUPED (CUPAC) | OLS on all pre-period features; variance reduction ≈ model R² | Poyarkov et al. (2016) |
| `scripts/post_stratification.py` | Post-stratified ATE | weights within-stratum diffs by population share; corrects imbalance bias, cuts variance | Miratrix, Sekhon, Yu (2013) |
| `scripts/group_sequential.py` | Alpha-spending boundaries | Pocock/OBF control Type I while naive peeking inflates it | Lan & DeMets (1983) |
| `scripts/msprt_always_valid.py` | Always-valid p-values | mSPRT lets you peek and stop any time, validly | Johari, Pekelis, Walsh (2015) |
| `scripts/sequential_ratio.py` | Sequential ratio metrics | delta-method + mSPRT for CTR, monitored continuously | combines the two above |
| `scripts/sequential_ab_testing.py` | Evan Miller's sequential rule | reproduce the size table, validate Type I/power, quantify sample savings | Evan Miller |
| `scripts/bayesian_ab_test.py` | Analytic Bayesian A/B | Beta-Binomial / Normal-Normal, P(B>A), expected loss, ROPE | conjugate posteriors |
| `scripts/bootstrap_ci.py` | Bootstrap CIs | percentile & BCa for skewed metrics and median/quantiles | Efron (1987) |
| `scripts/heterogeneous_treatment_effects.py` | HTE by segment | interaction model reveals Simpson's-paradox-like cancellation | OLS with interactions |
| `scripts/multiple_comparisons.py` | Multiple-testing correction | Bonferroni (FWER) vs Benjamini-Hochberg (FDR) | Benjamini & Hochberg (1995) |
| `scripts/novelty_primacy.py` | Time-varying effects | treat×day interaction detects novelty decay / primacy growth | — |
| `scripts/switchback.py` | Cluster & switchback designs | cluster-robust SE; naive over-/under-rejects; carryover bias | cluster-robust variance |
| `scripts/test_simulator.py` | Generic test calibration | plug any DGP + test → empirical Type I and power curve | — |

## End-to-end pipeline

`scripts/run_full_pipeline.py` ties the modules into one realistic flow on
synthetic data: SRM check → CUPED → delta-method CTR test → per-segment ATE with
BH correction → novelty check → a markdown report in `outputs/report.md`.

## Testing philosophy

The `tests/` suite re-runs every calibration with assertions:

- Type I error ≈ α (± tolerance) for each method under its null
- CI coverage ≈ 95% for the bootstrap
- naive peeking inflated, always-valid / alpha-spending controlled
- naive per-unit ratio SE inaccurate, delta-method SE accurate
- correctness on known-answer fixtures (SRM splits, segment uplifts, etc.)

Run with `uv run pytest`. CI (`.github/workflows/ci.yml`) runs ruff + pytest on
push.

## Layout

```
scripts/            method modules (one topic each) + end-to-end pipeline
tests/              calibration test suite
plots/  outputs/    generated artifacts (gitignored)
```