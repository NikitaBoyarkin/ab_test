# A/B Testing Methodology Toolkit

Empirical, calibration-driven A/B testing methods, implemented from the primary
literature and validated by simulation.

The guiding idea: a method is only as good as its Type I error under the null
and its power under a real effect. Rather than trusting asymptotic promises,
every module's `main()` prints the calibration it is judged on, and the test
suite re-runs that same calibration as assertions — Type I, power, CI coverage,
FWER/FDR — instead of assuming it.

## Headline results

`uv run python scripts/run_full_pipeline.py` runs the whole flow on a default
synthetic experiment (42,000 users, 21 days, A = 20,979 / B = 21,021) and writes
`outputs/report.md` plus three figures:

| Step | Result | Reading |
|---|---|---|
| Sample Ratio Mismatch (χ²) | A 20,979 / B 21,021, p = 0.8376 | split is clean — no bucketing bug |
| CUPED variance reduction | SE 0.00360 → 0.00327 | **17.7%** of the variance removed using pre-period data |
| Decision metric (delta-method CTR) | 0.0499 → 0.0550, +0.00512, p < 0.0001 | +0.51pp lift, significant |
| Segment heterogeneity (BH-corrected) | mobile +0.0015, desktop −0.0001, tablet −0.0128 | no segment differs once correction is applied (all p_adj > 0.32) |
| Novelty / primacy | treat×day +0.00065, p = 0.2714 | no time trend — the effect is stable |

The point of the demo is that each step can *fail loudly*: an SRM check that
fires, a CUPED adjustment that does not pay for itself, a segment that is only
significant before correction.

## Quickstart

```bash
uv sync --all-groups          # install deps + dev tools
uv run python scripts/srm_test.py   # run a single module's demo
uv run pytest                 # run the calibration test suite
uv run ruff check .           # lint
uv run python scripts/run_full_pipeline.py   # end-to-end demo -> outputs/report.md
uv run python scripts/make_figures.py        # full matplotlib figure gallery -> plots/
uv run python scripts/cli.py --help          # CLI: run any method from the terminal
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

## CLI

Run any method from the terminal without editing code. Output is stable JSON by
default.

```bash
uv run python scripts/cli.py --help
uv run python scripts/cli.py srm --counts 1000 1100
uv run python scripts/cli.py pipeline --data data/experiment_sample.csv
```

`data/experiment_sample.csv` is a small committed fixture (600 users) in the
pipeline schema `day, segment, treat, conv, pre_conv, pre_sessions, impressions,
clicks` — the schema `scripts/run_full_pipeline.py` produces. Multi-column
commands take **one numeric column per file** (a header row, if present, is
ignored), so split the columns out of the fixture first:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('data/experiment_sample.csv')
for c in ['pre_sessions', 'conv']:
    for t, g in df.groupby('treat'):
        g[c].to_csv(f'/tmp/{c}_{t}.csv', index=False, header=False)
"
uv run python scripts/cli.py cuped --y-a /tmp/conv_0.csv --y-b /tmp/conv_1.csv \
    --x-a /tmp/pre_sessions_0.csv --x-b /tmp/pre_sessions_1.csv
```

Invalid input (a bad file, a `--ratio` that does not sum to 1, a column-count
mismatch) exits 2 with one explanatory line instead of a traceback.

## End-to-end pipeline

`scripts/run_full_pipeline.py` ties the modules into one realistic flow on
synthetic data: SRM check → CUPED → delta-method CTR test → per-segment ATE with
BH correction → novelty check → a markdown report in `outputs/report.md`.

## Figures

`scripts/make_figures.py` renders the calibration story of every module as a
chart — 20 figures into `plots/` in one run (~15s). `scripts/plotting.py` holds
the shared style: headless Agg backend, a colorblind-safe Okabe-Ito palette,
constrained layout, 300 dpi, and a single `save_fig` helper. All new figures use
the object-oriented matplotlib API (`fig, ax = plt.subplots()`).

`scripts/sequential_ab_testing.py` writes two more charts on its own `main()`
(`sequential_savings_by_test_type.png`, `sequential_largest_mde.png`), which is
why a fully-populated `plots/` holds 22 files. Those two stay on plotnine and
keep their ggplot theme; matplotlib is used for the rest of the gallery.

`scripts/run_full_pipeline.py` additionally writes three figures
(`pipeline_ctr_ci.png`, `pipeline_segments.png`, `pipeline_novelty.png`) and
embeds them in `outputs/report.md`.

## Testing philosophy

The `tests/` suite re-runs every calibration with assertions:

- Type I error ≈ α (± tolerance) for each method under its null
- CI coverage ≈ 95% for the bootstrap, and the BCa interval matches
  `scipy.stats.bootstrap(method="bca")` to within a few percent of its width
- naive peeking inflated, always-valid / alpha-spending controlled
- naive per-unit ratio SE inaccurate, delta-method SE accurate
- Bonferroni holds FWER at α; BH holds FDR at α *on a mixed design* (an all-null
  design cannot show FDR, because there every rejection is a false discovery)
  while rejecting more true alternatives than Bonferroni
- the per-segment HTE confidence intervals are calibrated under a true null
  (~5% exclusion rate), which pins the SE to the fitted model
- correctness on known-answer fixtures: sample-size closed forms (3024 / 3839 /
  393), SRM splits, segment uplifts, CLI JSON invariants

Run with `uv run pytest`. CI (`.github/workflows/ci.yml`) does a frozen install,
ruff check + format check, the suite, then an end-to-end pipeline smoke run.

## Limitations & scope

- Everything runs on **synthetic** data from known DGPs. The calibrations are
  honest about the generating process — real traffic is not: no interference
  between users, no bots, no delayed conversions, no logging loss.
- These are single-experiment tools, not an experimentation platform: there is
  no assignment service, metric-definition store, or guardrail monitoring here.
- The Bayesian module is analytic conjugate (Beta-Binomial / Normal-Normal); it
  does not cover hierarchical or time-varying priors.
- `outputs/report.md` and the three `pipeline_*.png` figures are committed so the
  report renders on GitHub. Everything else under `plots/` is a regenerated
  artifact and is gitignored — regenerate it with `scripts/make_figures.py`.

## Layout

```
scripts/            method modules (one topic each) + end-to-end pipeline
scripts/plotting.py shared matplotlib style + save_fig helper
scripts/make_figures.py  full figure gallery (one function per module)
tests/              calibration test suite
data/               small committed CSV fixture used by the README examples
plots/  outputs/    generated artifacts (only report.md + pipeline_*.png committed)
```
