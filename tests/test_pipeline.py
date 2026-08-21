# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_full_pipeline as rfp


def test_generate_data_shape_and_columns():
    df = rfp.generate_data(n_days=5, n_per_day=200, seed=1)
    assert len(df) == 1000
    assert {"day", "segment", "treat", "conv", "pre_sessions", "impressions", "clicks"} <= set(
        df.columns
    )
    assert set(df["segment"].unique()) == {"mobile", "desktop", "tablet"}
    assert set(df["treat"].unique()) == {0, 1}


def test_generate_data_seed_reproducible():
    a = rfp.generate_data(n_days=3, n_per_day=100, seed=7)
    b = rfp.generate_data(n_days=3, n_per_day=100, seed=7)
    assert a.equals(b)


def test_pipeline_writes_report(tmp_path):
    df = rfp.generate_data(n_days=5, n_per_day=200, seed=1)
    n_a, n_b = int((df["treat"] == 0).sum()), int((df["treat"] == 1).sum())
    srm = rfp.srm_test.srm_test([n_a, n_b], [0.5, 0.5])
    cup = rfp.cuped_calibrate(df)
    xa, ya, xb, yb = rfp._ratio_by_group(df)
    ratio = rfp.delta_method_ratio.delta_method_test(xa, ya, xb, yb)
    seg = rfp._segment_ate(df)
    seg = seg.assign(p_adj=seg["p_value"], sig=False)
    trend, _ = rfp.novelty_primacy.fit_trend(df.assign(y=df["conv"]))

    out = tmp_path / "report.md"
    rfp.write_report(df, srm, cup, ratio, seg, trend, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "# A/B Experiment Report" in text
    assert "Sample Ratio Mismatch" in text
    assert "Conclusion" in text
    assert "CTR" in text


def test_cuped_reduces_se():
    df = rfp.generate_data(n_days=7, n_per_day=500, seed=3)
    cup = rfp.cuped_calibrate(df)
    assert 0 <= cup["var_reduction"] <= 1
    assert cup["se_cuped"] <= cup["se_naive"]
