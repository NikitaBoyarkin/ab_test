# -*- coding: utf-8 -*-
"""Command-line interface for the A/B testing toolkit.

Run any method from the terminal without editing code:

    uv run python scripts/cli.py srm --counts 1000 1100
    uv run python scripts/cli.py cuped --y-a a.csv --y-b b.csv --x-a xa.csv --x-b xb.csv
    uv run python scripts/cli.py ratio --x-a ia.csv --y-a ca.csv --x-b ib.csv --y-b cb.csv
    uv run python scripts/cli.py pipeline --data experiment.csv

Output is stable JSON by default; pass --format markdown for a report-style view.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cuped
import delta_method_ratio
import run_full_pipeline
import srm_test


def _jsonable(obj):
    """Recursively convert numpy/pandas values to plain JSON types."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _load_col(path):
    """First column of a CSV as a float array (header row, if any, is dropped)."""
    col = pd.read_csv(path, header=None).iloc[:, 0]
    return pd.to_numeric(col, errors="coerce").dropna().to_numpy(dtype=float)


def _emit(result, fmt):
    if fmt == "json":
        print(json.dumps(_jsonable(result), indent=2))
    else:
        _emit_markdown(_jsonable(result))


def _emit_markdown(result, indent=0):
    pad = "  " * indent
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"{pad}### {k}")
            _emit_markdown(v, indent + 1)
        elif isinstance(v, list):
            print(f"{pad}### {k}")
            for row in v:
                print(f"{pad}  - {row}")
        else:
            print(f"{pad}- **{k}:** {v}")


def _add_format(parser):
    parser.add_argument("--format", choices=["json", "markdown"], default="json")


def cmd_srm(args):
    _emit(srm_test.srm_test(args.counts, args.ratio, alpha=args.alpha), args.format)


def cmd_cuped(args):
    y_a, y_b = _load_col(args.y_a), _load_col(args.y_b)
    x_a, x_b = _load_col(args.x_a), _load_col(args.x_b)
    theta = cuped.cuped_theta(np.concatenate([y_a, y_b]), np.concatenate([x_a, x_b]))
    a = cuped.cuped_adjust(y_a, x_a, theta)
    b = cuped.cuped_adjust(y_b, x_b, theta)
    naive = cuped.t_test(y_a, y_b)
    adj = cuped.t_test(a, b)
    _emit(
        {
            "diff": adj["diff"],
            "se": adj["se"],
            "p_value": adj["p_value"],
            "significant": bool(adj["p_value"] < args.alpha),
            "theta": theta,
            "variance_reduction": 1
            - (adj["var_a"] + adj["var_b"]) / (naive["var_a"] + naive["var_b"]),
        },
        args.format,
    )


def cmd_ratio(args):
    x_a, y_a = _load_col(args.x_a), _load_col(args.y_a)
    x_b, y_b = _load_col(args.x_b), _load_col(args.y_b)
    _emit(delta_method_ratio.delta_method_test(x_a, y_a, x_b, y_b, alpha=args.alpha), args.format)


def cmd_pipeline(args):
    df = pd.read_csv(args.data)
    required = {"day", "segment", "treat", "conv", "pre_sessions", "impressions", "clicks"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"pipeline data must have columns {sorted(required)}; missing: {sorted(missing)}"
        )
    _emit(run_full_pipeline.run_pipeline(df), args.format)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="abtest",
        description="A/B testing methodology toolkit — run any method from the terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("srm", help="Sample Ratio Mismatch check")
    p.add_argument("--counts", nargs="+", type=int, required=True, help="observed counts per arm, e.g. 1000 1100")
    p.add_argument(
        "--ratio",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="expected split (default: 0.5 0.5)",
    )
    p.add_argument(
        "--alpha", type=float, default=srm_test.SRM_ALPHA, help="flag if p < alpha (default: 0.001)"
    )
    _add_format(p)
    p.set_defaults(func=cmd_srm)

    p = sub.add_parser("cuped", help="CUPED variance-reduced two-sample test")
    p.add_argument("--y-a", required=True, help="CSV: control outcomes")
    p.add_argument("--y-b", required=True, help="CSV: treatment outcomes")
    p.add_argument("--x-a", required=True, help="CSV: control pre-period covariate")
    p.add_argument("--x-b", required=True, help="CSV: treatment pre-period covariate")
    p.add_argument("--alpha", type=float, default=0.05)
    _add_format(p)
    p.set_defaults(func=cmd_cuped)

    p = sub.add_parser("ratio", help="Ratio metric test via the delta method (CTR, RPC)")
    p.add_argument("--x-a", required=True, help="CSV: control denominators (e.g. impressions)")
    p.add_argument("--y-a", required=True, help="CSV: control numerators (e.g. clicks)")
    p.add_argument("--x-b", required=True, help="CSV: treatment denominators")
    p.add_argument("--y-b", required=True, help="CSV: treatment numerators")
    p.add_argument("--alpha", type=float, default=0.05)
    _add_format(p)
    p.set_defaults(func=cmd_ratio)

    p = sub.add_parser("pipeline", help="End-to-end experiment flow on a CSV")
    p.add_argument(
        "--data",
        required=True,
        help="CSV with columns: day, segment, treat, conv, pre_sessions, impressions, clicks",
    )
    _add_format(p)
    p.set_defaults(func=cmd_pipeline)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
