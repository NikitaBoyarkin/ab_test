# План улучшения `ab_test` — аудит 2026-10-18

**Что аудировано:** `/Users/nikitaboarkin/Desktop/00 ide/00 portfolio/ab_test` — портфолио-библиотека методов A/B-тестирования (22 модуля, ~5.9k LOC + 18 примеров ~1k LOC, 86 тестов, 18 коммитов, remote `NikitaBoyarkin/ab_test`).

**Как получено:** 4 независимые оси через параллельных агентов (A — статистическая корректность, B — код/архитектура, C — тесты, D — доки↔реальность) + собственная проверка (ось E: прогон suite, ruff, git, 18 примеров, изолированный вызов упавшей фигуры). Ничего не редактировалось.

---

## 1. Что проверено лично (Evidence Ledger)

| Утверждение | Как проверено | Класс |
|---|---|---|
| Тесты зелёные | `uv run pytest -q` → 86 passed, 29.43s | **Verified** |
| ruff падает → CI Lint красный | `ruff check . --output-format=concise` → 5 ошибок в `reports/eda/scripts/plot_ab_test.py` (F401, 3×B905, B007) | **Verified** |
| `figure_switchback` падает | изолированный вызов с заглушённым `save_fig` → `KeyError: 0` | **Verified** |
| Причина падения | `ols_effect_cluster_robust` (`switchback.py:64`) возвращает `dict` через `_effect` (`switchback.py:49-56`), а `make_figures.py:528` индексирует `[0]` | **Verified** |
| `examples/` не в git | `git status --short` → `?? examples/`; `git ls-files` = 69 файлов без `examples/` | **Verified** |
| 18 примеров исполняются | последовательный прогон, все `rc=0`, у каждого строка `VERDICT:` | **Verified** |
| `simulate()` даёт Type I = 0.0 при чужом тесте | код: `test_simulator.py:34` читает `res.get("reject", res.get("significant"))`, а `cuped.py:62` и `bootstrap_ci.py:133` этих ключей не возвращают → `None` → falsy | **Verified** (по коду) |
| Математика формул верна | ось A: θ=Cov/Var, delta-SE с ÷mean(X)², Λ_t mSPRT, OBF √(K/k), BH step-up, Beta(a+k,b+n−k), BCa (z0+jackknife) — совпадают с каноном | **Verified** (ось A) |
| README headline-числа воспроизводимы | `outputs/report.md` + `reports/eda/eda_metrics.json:328,331,377-381` | **Verified** |
| «report.md воспроизводим» держится на тесте | `tests/test_pipeline.py:20-51` использует seed=1/3 и проверяет форму, не числа | **Inferred, не связано тестом** |
| «20 figures за один прогон» | `make_figures.main()` падает на 17-й фигуре | **Опровергнуто** |

---

## 2. Итог одной строкой

**Математика чистая, тесты заметно сильнее среднего по портфолио, но поставка сломана:** CI красный, галерея фигур падает, 18 примеров не в репозитории, а README обещает больше, чем отгружено. Наибольшая отдача — не в новых методах, а в том, чтобы уже заявленное стало правдой.

---

## 3. Блокеры (P0) — чинятся за один вечер, но ломают первое впечатление

| # | Где | Проблема | Фикс |
|---|---|---|---|
| B1 | `reports/eda/scripts/plot_ab_test.py:22,135,166,253` | `ruff check .` падает → **шаг Lint в CI красный** на каждом пуше. Бейдж (если добавить) будет красным | `ruff check --fix` (снимает F401), затем вручную `strict=False` в 3 `zip()` и `_e` вместо `e` |
| B2 | `make_figures.py:528` | `ols_effect_cluster_robust(...)[0]` → `KeyError: 0`: обе функции `switchback.py` возвращают `dict`, не кортеж. `figure_switchback` падает → **вся галерея и заявление README:152 «20 figures in one run» ложны** | `["coef"]` вместо `[0]`; добавить `make_figures.py` в CI (сейчас не запускается) |
| B3 | `examples/` (18 файлов, ~1025 LOC) | **untracked**: `git ls-files` их не видит. README:78 Quickstart, `README.md:107`, `examples/README.md:30-80` и `docs/diagrams/ab-test.html` на них ссылаются → на GitHub **404**, у свежего клона не работает `uv run python examples/01_srm_test.py`. При этом незакоммиченный diff `docs/prd.md:254-259` уже ставит REQ-016 в ✅ | `git add examples/` (18/18 исполняются — проверено) **или** снять ссылки, если решено не публиковать |

---

## 4. Высокий приоритет (P1) — заявления, не связанные механизмом

| # | Где | Проблема | Фикс |
|---|---|---|---|
| H1 | `test_simulator.py:34` ↔ `cuped.py:62`, `bootstrap_ci.py:133` | Докстринг обещает «any module's test fn can be plugged in», но `welch_mean_test` и `t_test` не возвращают `reject`/`significant`. `simulate()` молча получает `None` → **`rejection_rate = 0.0`**, то есть калибровочный стенд библиотеки врёт для 2 из 3 «чужих» тестов, вместо того чтобы упасть | Вернуть обязательный `reject` в оба `dict` **и** заменить `.get` на `res["reject"]` (KeyError вместо тихого нуля) |
| H2 | `README.md:55` ↔ `run_full_pipeline.py:40,50` | README: «no time trend — the effect is stable». Но DGP пайплайна генерирует `novelty=0.6`, т.е. **затухающий эффект**, а модуль его не детектирует (коэф. +0.00065 при истинном ≈−0.00025). False negative выдан за отсутствие эффекта. `outputs/report.md:33-34` осторожнее («stable effect **(or no effect)**») — хедж потерян именно в README | Либо поднять n/эффект в DGP, либо вернуть в README хедж из отчёта |
| H3 | `docs/prd.md` + `scripts/bootstrap_ci.py` + `scripts/switchback.py` + `tests/test_bootstrap_ci.py` | Незакоммичено (+83/−9). **Опубликованный PRD отличается от проверяемого**, а «pinned calibration results» в истории не соответствует рабочей копии | Закоммитить, предварительно прогнав suite |
| H4 | `tests/test_pipeline.py:51` | `se_cuped <= se_naive` проходит и при `theta = 0`, т.е. при полностью отключённом CUPED. Сквозной пайплайн CUPED фактически не проверяет | Строгое `<` с запасом (напр. `se_cuped < 0.99 * se_naive`) |
| H5 | `tests/test_srm.py:6-33` | Ни один тест не пинует `SRM_ALPHA`: фикстуры стоят на p=6.3e-5 и p=0.81, поэтому мутация α 0.001→0.05 **проходит незамеченной** | Тест с p между 0.001 и 0.05 → вердикт должен перевернуться |
| H6 | `cuped.py:61`, `bootstrap_ci.py:136`, `post_stratification.py:76`, `novelty_primacy.py:77` | p-value из t-распределения Уэлча, а CI — по нормальной квантили. Две несогласованные аппроксимации внутри одной функции; ни один тест это не сверяет | Выбрать одну (Welch t + `t.ppf(df)`), либо явно назвать это нормальной аппроксимацией |
| H7 | `ratio_cuped.py:64` | `np.cov(...)/np.var(z_pre_all, ddof=1)` без guard'а на нулевую дисперсию — **тогда как родительский CUPED guard имеет** (`cuped.py:33`). Константный pre-period → `theta = nan` → молча `nan` в se/p/CI | Перенести guard из `cuped.py:33` |
| H8 | `delta_method_ratio.py:29` ↔ `sequential_ratio.py:28` | Одна формула дисперсии отношения в двух нормировках (`÷n` внутри vs снаружи). А `sequential_ratio.py:44` берёт `n = len(x_a)` — верно только при равных плечах; при `n_a ≠ n_b` **p-value неверный** | Свести в одну функцию; для неравных плеч `1/(V_a/n_a + V_b/n_b)` |
| H9 | `novelty_primacy.py:62,66-80` | `y ~ treat*day` поглощает только **линейный** тренд; нелинейный или дифференциальный (сезонность, кампания в середине окна) целиком уходит в `treat:day` и выдаётся за novelty. Дневные CI берутся из Welch-SE внутри дня — интервалы приписаны модели, из которой не посчитаны | Pre-period/контрольный тренд либо явная оговорка об идентификации |
| H10 | `switchback.py:13-17`, `tests/test_switchback.py:31-38` | Тезис «naive-SE консервативен» доказан сравнением на **одном** seed'е; при G=30 CR1-поправка — нижняя граница, фактор Мулотона ≈1−ρ_ε | Проверять на 50+ seed'ах, оговорить G≪40 |
| H11 | `heterogeneous_treatment_effects.py:95-99` | Значимость интеракций помечается при сыром p<0.05 без поправки; в модели 2 интеракции → FWER≈10%. Совместного F-теста гетерогенности нет, хотя `multiple_comparisons.py` лежит рядом | Применить BH внутри модуля или joint F-тест |

---

## 5. Средний приоритет (P2)

**Статистическая честность (оговорки, а не баги):**
- `cuped.py:79,105`, `run_full_pipeline.py:179` — θ оценивается на пулированных данных (включая обработанную руку); канон Deng et al. (2013) — на контроле. Смещение O(effect²), нигде не оговорено.
- `cupac.py:88,133` — R² и дисперсия контрольной руки **in-sample**; в docstring оговорено, в README («variance reduction ≈ R²») — нет. Для GBM без cross-fitting утверждение сломается.
- `post_stratification.py:51-56,69` — при `weights=None` веса берутся из наблюдаемых пропорций, а дисперсия считается по формуле для **известных** весов → `Var(ATE_ps)` недооценена.
- `sample_size.py:42` — формула unpooled, слегка анти-консервативна; не сказано, что `power_two_proportion` использует ту же SE.
- `bayesian_ab_test.py:138-148` — калибровка трактует `P(B>A)` как частотный p-value; под H0 равномерность лишь асимптотическая.
- `ratio_cuped.py:70` — масштаб `1/mean(X)` по пулу, тогда как `delta_method_ratio.py:39` — по каждой группе; два модуля дают разные SE на одних данных.
- `run_full_pipeline.py:89-100` — сегменты тестируются независимыми Welch, а `fit_het` — interaction-моделью: два модуля, два разных ответа на «есть ли гетерогенность».

**Тестовое покрытие (ось C: 79% pin — это сильно, дыры точечные):**
- Нет ни одного теста на границы: `n=0/1`, пустая группа, все конверсии 0 или 1, perfect separation, сильная несбалансированность (n_a=10 vs n_b=100k).
- Нет ни одного теста инвариантности: swap A/B ⇒ тот же p; масштаб метрики ×k ⇒ тот же p; перемешивание строк входа.
- `delta_method_ratio.py:65` `naive_per_unit_ratio_test` — **ноль ассертов**, любую поломку suite пропустит.
- `run_full_pipeline` не откалиброван на A/A.
- `bootstrap_ci` — покрытие CI есть, FP-rate по `significant` нет.
- `test_bayesian_ab_test.py:6-37` — `rope`, `p_equiv_within_rope`, `expected_loss_*`, `ci95`, `lift_median` не проверяются нигде.
- `test_figures.py:30-45` — исполняются 2 из 18 фигур; `make_figures.py` (687 LOC) в CI не запускается.
- Не покрыты: guard `stratified_ate` «≥2 на арм», CLI-guard'ы `_check_counts`/`_check_same_length`, `cmd_ratio`, `sequential_ab_testing.run_grid` (а он несёт headline-claim про экономию выборки).
- `post_stratification.py:51` guard не покрыт; `delta_method_ratio.py:36` и `sequential_ratio.py:31` делят на `mean_x` без проверки.
- `sample_size.py:42` — `absolute_diff = 0` → `ZeroDivisionError`; `bayesian_ab_test.py:85` — `var_a = 0` → `ZeroDivisionError`.
- `msprt_always_valid.py:36-40` — при `rho = 0` Λ≡1, p≡1 навсегда: метод молча сообщает «эффекта нет».
- `srm_test.py:38-41` — проверяется сумма `expected_ratio`, но не `len(observed) == len(expected_ratio)`.
- `heterogeneous_treatment_effects.py:47` — `ref_segment` не проверяется на наличие в `df`.
- `bootstrap_ci.py:52-55` — `else: # BCa` ловит любое значение `method`: опечатка `"BCA"` молча выбирает BCa.
- Нет `hypothesis` ни в одном тесте; `@pytest.mark.parametrize` и `pytest.approx` не используются ни разу.
- `conftest.py` (7 строк, ноль фикстур) — ~20 скопированных циклов симуляции, 12 литералов seed'ов, 4 локальных хелпера: фикстуры `aa_stream`/`null_dgp`/`csv_arms` сняли бы ~90-120 LOC.

**Инфраструктура и поставка:**
- CI не проверяет: mypy, coverage, матрицу версий. Ставит только Python 3.12 при `requires-python = ">=3.11"` — заявленный пол не тестируется.
- Шаг smoke (`ci.yml:32-33`) утверждает лишь `exit 0`, при этом мутируя `outputs/report.md` и `plots/`; в CI не запускаются `examples/`.
- `pyproject.toml`: нет `[build-system]` → `pip install .` не работает; нет `[tool.mypy]`/`[tool.coverage]`; ruff без `C4`/`SIM`/`RUF`; в pytest только `testpaths` (нет `addopts`, `--strict-markers`).
- 6 копий `sys.path`-шима: `conftest.py:6-7`, `examples/_pathsetup.py:16-21`, `test_pipeline.py:5`, `make_figures.py:22`, `cli.py:22`, `run_full_pipeline.py:23`.
- Дублирование: Welch-SE `sqrt(va/na + vb/nb)` — **15 сайтов**; двусторонний нормальный p — **7 копий одной строки**; Welch t-тест — **3 реализации** (`cuped.py:56`, `bootstrap_ci.py:129`, `test_simulator.py:60`) с разными наборами ключей и CI (t vs z); bootstrap среднего — 2 реализации.
- Результаты — голые `dict` без типа: оценка зовётся `point`/`diff`/`coef`/`ate`/`lift_mean`, CI — то `ci_low`+`ci_high`, то кортеж `ci95`, то `ci`; `significant` есть в 4 модулях и отсутствует в `cuped`/`bootstrap_ci`. mypy нет, поэтому ничто не ловит расхождение.
- 4 точки входа для одних 18 методов: 21 × `main()` (767 LOC) + `examples/` (1025) + `cli.py` (216) + `run_full_pipeline.py` (268) ≈ **1792 LOC (~30% Python в репо)**. `make_figures.py`: 12 из 18 фигур **переписывают тот же калибровочный цикл**, что `main()` своего модуля (~250-300 LOC задвоённых симуляций) — цифры в README и на графике считаются двумя разными прогонами.
- `run_full_pipeline.py:131` вызывает `run_pipeline(df)`, потом `make_figures.figure_pipeline(df)` (`make_figures.py:596`) вызывает `run_pipeline` на том же df — двойной прогон.
- Мёртвый код: `cupac.py:120` — `t, p = stats.ttest_ind(...)`, обе переменные не используются. Недостижимо: пресет `"dark"` и параметр `preset` в `plotting.py:51-65` (вызывается только `"light"`).
- `bayesian_ab_test.py:34` документирует параметр `rope`, которого функция не принимает; `make_figures.py:585` `figure_pipeline` без `-> Path` на фоне остальных 17.

**Документация:**
- `README.md:140` обещает `exits 2`, а `cli.py:142` даёт код **1** для `pipeline`; ни один тест код выхода не проверяет.
- `docs/prd.md:322` — «plots/ outputs/ gitignored» противоречит `.gitignore` (4 файла раз-игнорированы).
- `docs/prd.md:343-349` — Phase 1 стоит `[ ]` при Phase 2/3 `[x]`: чекбоксы противоречивы.
- REQ-004 AC «works for means and ratio metrics» в `cuped.py` не выполнен (только `welch_mean_test`) — вынесено в `ratio_cuped.py`. REQ-010 AC «детектирует novelty decay» не подтверждён на собственном DGP пайплайна.
- `docs/diagrams/ab-test.html` называет **9 из 18** модулей — схема отражает ревизию до REQ-014/016, хотя добавлена последним коммитом.
- `reports/eda/eda-ab-test-2026-10-13.md:249-253` ссылается на `README.md:3,26-27,28-29,30,49-53,171` — нумерация устарела.
- `README.md:152` «~15s» — без замера. `README.md:31,54` «all p_adj > 0.32» опирается на tablet 0.3224 — запас 0.0004, от смены seed формулировка перевернётся. `README.md:162` «additionally writes three figures» — те же 3 файла, что уже входят в 20. `README.md:38-41` «every module's main()» — `plotting.py` без `main()`.

---

## 6. Что сделано хорошо — не трогать и, наоборот, заявить в README

Это сильнее, чем у типичного портфолио, и сейчас недооценено:

1. **`bootstrap_ci.py`** — BCa по учебнику (z0, jackknife-acceleration с правильными знаками для двухвыборочной разности), явные `ValueError` вместо NaN на tied-статистике, golden-тест против `scipy.stats.bootstrap(method="bca")` и coverage-тест ~95%. Лучший модуль.
2. **`msprt_always_valid.py` + `sequential_ratio.py`** — Λ_t совпадает с Johari-Pekelis-Walsh в обеих формах (по сумме наблюдений и в информационной параметризации); порог 1/α отделён от параметра смешивания ρ; clamp p≤1 закреплён regression-тестом.
3. **`multiple_comparisons.py`** — BH как step-up с монотонизацией (а не наивной сортировкой), калибровка разведена на all-null (FWER) и mixed (FDR + power) с объяснением, почему all-null не может показать FDR.
4. **`sample_size.py`** — closed forms пиннуты на известные ответы (3024/3839/393), `power_two_proportion` алгебраически согласован с `two_proportion`.
5. **`group_sequential.py`** — границы не взяты «по табличке», а калибруются симуляцией броуновского пути с бисекцией по скаляру c; Type I проверяется эмпирически на откалиброванной границе.
6. **`delta_method_ratio.py:65`** — «наивный» тест явно помечен `Biased` в docstring: это намеренный контраст, а не баг.
7. **Дисциплина тестов**: 79% ассертов — «pin»-класса, 20 тестов реально мерят Type I / coverage / FWER / FDR на 100-8000 симуляциях, в 9 файлах паттерн «наивный baseline vs корректный метод» (тест упал бы, если бы метод не отличался от наивного). Сиды везде локальные (`default_rng`), глобальный RNG не загрязняется, флейков не обнаружено.
8. **Сквозной паттерн**: каждый модуль печатает свою калибровку, а `tests/` превращает её в ассерт.

---

## 7. План по фазам

Порядок важен: сначала **сделать заявленное правдой**, потом связывать механизмом, и только потом рефакторить. Упаковывать в пакет до дедупликации нельзя — иначе миграция перепишет 60 импортов и оставит 15 копий формулы внутри нового пакета.

### Фаза 0 — гигиена (≈1 вечер, один коммит)
1. B1: `ruff check . --fix`, руками `strict=False` ×3 и `_e`.
2. B2: `make_figures.py:528` → `["coef"]`.
3. B3: `git add examples/` (или снять ссылки).
4. H3: закоммитить `docs/prd.md`, `bootstrap_ci.py`, `switchback.py`, `test_bootstrap_ci.py`.

**Приёмка:** `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` — зелёные; `uv run python scripts/make_figures.py` доходит до конца; `git ls-files examples/ | wc -l` = 19.

### Фаза 1 — связать заявления механизмом (≈1-2 вечера)
Правило: заявление без механизма — проза (`.claude/rules/claim-mechanism.md`).
5. H1: обязательный `reject` во всех тест-функциях + `res["reject"]` вместо `.get` в `simulate()`.
6. H4: `se_cuped < 0.99 * se_naive`.
7. H5: тест на `SRM_ALPHA` с p между 0.001 и 0.05.
8. H7: guard нулевой дисперсии в `ratio_cuped.py:64` + тест.
9. Добавить в CI шаг прогона `make_figures.py` и `examples/*.py`.
10. Тесты границ (n=0/1, пустая группа, все конверсии 0/1, perfect separation, несбалансированные плечи) и инвариантности (swap A/B, ×k масштаб, перемешивание строк) — параметризованно.
11. Ассерт на `naive_per_unit_ratio_test`; тесты согласованности `ratio_delta_se` ↔ `ratio_delta_var_perunit` и `bootstrap_ci.t_test` ↔ `cuped.welch_mean_test`.

**Приёмка:** каждая строка выше падает при мутации соответствующей константы/формулы.

### Фаза 2 — статистическая честность (≈1-2 вечера)
12. H2: вернуть в README:55 хедж «(or no effect)» из `outputs/report.md:33-34`, либо поднять мощность DGP.
13. H6: согласовать t-p и z-CI (одна аппроксимация на функцию).
14. H8: свести дисперсию отношения в одну функцию; починить `n` для неравных плеч.
15. H11: BH или joint F-тест внутри `heterogeneous_treatment_effects`.
16. H9: оговорка об идентификации novelty + pre-period тренд.
17. H10: switchback на 50+ seed'ах, оговорка про G≪40.
18. Оговорки в docstring/README по P2: pooled θ, in-sample R², observed-weights дисперсия, unpooled SE, асимптотика байесовской калибровки, расхождение масштаба `1/mean(X)`.
19. `README.md:140`: либо код выхода 2 в `cli.py`, либо исправить README; добавить тест кода выхода.

**Приёмка:** ни одно огрубление не подаётся как точный результат; каждое либо исправлено, либо названо.

### Фаза 3 — дедупликация (≈2 вечера)
20. `scripts/stats_core.py`: `welch_se()`, двусторонний нормальный p, дисперсия отношения; заменить 15 + 7 + дублирующие сайты.
21. `TestResult` (TypedDict или dataclass) с полями `point/se/ci_low/ci_high/p_value/significant` — единая форма результата во всех 22 модулях.
22. Свести 3 Welch-реализации и 2 bootstrap-реализации.
23. Убрать дублирование калибровочных циклов между `make_figures` и `main()` (~250-300 LOC) — фигуры должны вызывать общий код, а не переписывать симуляцию.
24. Мелочи: `cupac.py:120` мёртвый код; `plotting.py:51-65` недостижимый пресет; `bayesian_ab_test.py:34` дрейф докстринга; `run_full_pipeline.py:131` двойной прогон; guard'ы на `mean_x=0`, `absolute_diff=0`, `var_a=0`, `rho=0`, `ref_segment`, длину массивов в `srm_test`.

**Приёмка:** grep по `sqrt(np.var` и по `stats.norm.ppf(0.975)` даёт по одному сайту; `uv run pytest -q` зелёный.

### Фаза 4 — структура (≈1-2 вечера)
25. `src/ab_testing_toolkit/` + `[build-system]` (hatchling) + `[project.scripts] abtest = "...cli:main"`; снять 6 `sys.path`-шимов; `conftest.py` очищается.
26. mypy + coverage в CI, матрица 3.11/3.12/3.13, `addopts`, `--strict-markers`.
27. `[tool.coverage]` + порог; закрыть непокрытые поверхности (`sequential_ab_testing.run_grid`, `cli.cmd_ratio`, guard'ы).

**Приёмка:** `uv pip install -e .` работает; `abtest --help` работает; `import ab_testing_toolkit` без хаков.

### Фаза 5 — портфолио (≈1 вечер)
28. Бейдж CI, скриншот `plots/pipeline_ctr_ci.png` в шапке (сейчас в README нет ни одного `![`), упоминание MIT.
29. Слинковать из README `docs/prd.md` и `reports/eda/` — сейчас это невидимые, но самые сильные доказательные артефакты (в т.ч. truth-check `eda-ab-test-2026-10-13.md:245-258`).
30. Раздел «How to read the numbers» — сейчас есть только абзац `README.md:57-62`, а `p=0.0000` в отчёте (underflow) выглядит как заглушка.
31. Обновить `docs/diagrams/ab-test.html` (9/18 модулей) и `reports/eda/eda-ab-test-2026-10-13.md:249-253` (устаревшая нумерация README).
32. Починить `README.md:31,54,152,162`, `docs/prd.md:322,343-349`, чекбоксы PRD.
33. EN-summary в начале RU-доков (`prd.md`, `eda-*.md`) либо не линковать их — внешний читатель упирается в русский PRD.

---

## 8. Чего не делать

- **Не добавлять новые методы.** В библиотеке 18 методов и REQ-019 (упаковка) в PRD; дефицит не в охвате, а в том, что заявленное не подтверждено. Новый метод = ещё одна непокрытая поверхность.
- **Не упаковывать в пакет до Фазы 3** — миграция зафиксирует дублирование внутри нового пакета.
- **Не переписывать математику** — ось A ошибок формул не нашла; править нужно формулировки и оговорки, а не формулы.
- **Не удалять существующие тесты**, даже слабые: сначала заменить сильным (`tautology` → `pin`), иначе потеряется покрытие.
- **Не трогать `data/experiment_sample.csv`** — README честно оговаривает, что это отдельная синтетика (3 дня, 292/308), а не выход `generate_data()` пайплайна (21 день, 20979/21021).

---

## 9. Сводная оценка

| Фаза | Что даёт | Оценка | Риск |
|---|---|---|---|
| 0 — гигиена | CI зелёный, галерея работает, примеры в репо | ~1 вечер | нулевой |
| 1 — механизмы | заявления падают при поломке | ~1-2 вечера | низкий |
| 2 — честность | ни одно огрубление не выдаётся за точный результат | ~1-2 вечера | низкий |
| 3 — дедупликация | −300…−500 LOC, одна формула = один ответ | ~2 вечера | средний (60+ импортов) |
| 4 — структура | `pip install`-able, типы, покрытие | ~1-2 вечера | средний (переезд 22 файлов) |
| 5 — портфолио | то, что уже сделано, становится видимым | ~1 вечер | нулевой |

**Наибольший эффект на впечатление от портфолио на единицу усилий — Фаза 0 и 5.** Фазы 3-4 — это уже инженерия, а не витрина.
