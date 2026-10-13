# EDA Report

**Rows:** 600 · **Columns:** 8 · **Memory:** 0.04 MB

## Profile

| Column | dtype |
|---|---|
| day | int64 |
| segment | str |
| treat | int64 |
| conv | int64 |
| pre_conv | int64 |
| pre_sessions | int64 |
| impressions | int64 |
| clicks | int64 |

## Data Quality

- Duplicated rows: **3**
- Constant columns: **0**
- Пропусков нет.

## Numeric Distributions

| Column | n | mean | median | std | p25 | p75 | p95 | outliers | skew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| day | 600 | 1.0 | 1.0 | 0.8172 | 0.0 | 2.0 | 2.0 | 0.0% | 0.0 |
| treat | 600 | 0.5133 | 1.0 | 0.5002 | 0.0 | 1.0 | 1.0 | 0.0% | -0.0535 |
| conv | 600 | 0.1717 | 0.0 | 0.3774 | 0.0 | 0.0 | 1.0 | 0.0% | 1.7458 |
| pre_conv | 600 | 0.1783 | 0.0 | 0.3831 | 0.0 | 0.0 | 1.0 | 0.0% | 1.6848 |
| pre_sessions | 600 | 3.4683 | 3.0 | 2.4892 | 2.0 | 5.0 | 8.0 | 2.2% | 1.1296 |
| impressions | 600 | 39.7467 | 34.0 | 25.4347 | 22.0 | 49.0 | 89.05 | 5.0% | 1.6771 |
| clicks | 600 | 2.0667 | 2.0 | 1.9378 | 1.0 | 3.0 | 6.0 | 3.3% | 1.2964 |

## Categorical Slices

**segment** — кардинальность 3

| Value | Count | Share |
|---|---:|---:|
| mobile | 241 | 40.2% |
| desktop | 233 | 38.8% |
| tablet | 126 | 21.0% |

## Correlations

| Pair | r | Note |
|---|---:|---|
| impressions ~ clicks | 0.6675 | moderate |

## Metric by Segment

### segment (overall mean = 0.1717, gap x1.3262)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| mobile | 241 | 0.1992 | 48.0 | 46.6% | +16.0% |
| desktop | 233 | 0.1502 | 35.0 | 34.0% | -12.5% |
| tablet | 126 | 0.1587 | 20.0 | 19.4% | -7.5% |

### treat (overall mean = 0.1717, gap x1.1752)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| 1.0 | 308 | 0.1851 | 57.0 | 55.3% | +7.8% |
| 0.0 | 292 | 0.1575 | 46.0 | 44.7% | -8.2% |


## Findings

- **[WARN]** Дубликаты строк: 3 — проверить grain
- **[WARN]** Выбросов в impressions: 5% — проверить, сигнал или ошибка

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
