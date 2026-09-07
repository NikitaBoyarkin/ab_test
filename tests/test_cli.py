# -*- coding: utf-8 -*-
"""CLI smoke tests: each subcommand runs and emits stable JSON."""

import json

import cli
import cuped
import pandas as pd
import run_full_pipeline


def _run(capsys, argv):
    cli.main(argv)
    return json.loads(capsys.readouterr().out)


def test_srm_verdict(capsys):
    out = _run(capsys, ["srm", "--counts", "1000", "1100", "--alpha", "0.05"])
    assert out["srm_detected"] is True
    out = _run(capsys, ["srm", "--counts", "1000", "1000"])
    assert out["srm_detected"] is False


def test_cuped_runs(capsys, tmp_path):
    x_a, y_a, x_b, y_b = cuped.simulate_cuped(n=2000, effect=0.1, rho=0.6, seed=1)
    for name, arr in [("xa", x_a), ("ya", y_a), ("xb", x_b), ("yb", y_b)]:
        pd.Series(arr).to_csv(tmp_path / f"{name}.csv", index=False, header=False)
    out = _run(
        capsys,
        [
            "cuped",
            "--y-a",
            str(tmp_path / "ya.csv"),
            "--y-b",
            str(tmp_path / "yb.csv"),
            "--x-a",
            str(tmp_path / "xa.csv"),
            "--x-b",
            str(tmp_path / "xb.csv"),
        ],
    )
    assert out["significant"] is True
    assert out["variance_reduction"] > 0


def test_pipeline_runs(capsys, tmp_path):
    df = run_full_pipeline.generate_data(n_days=3, n_per_day=200, seed=7)
    df.to_csv(tmp_path / "exp.csv", index=False)
    out = _run(capsys, ["pipeline", "--data", str(tmp_path / "exp.csv")])
    assert {"srm", "cuped", "ratio", "segments", "trend"} <= set(out)
    assert out["ratio"]["significant"] in (True, False)
