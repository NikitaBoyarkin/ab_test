# -*- coding: utf-8 -*-
"""Sample Ratio Mismatch (SRM) test.

SRM is the first check on any experiment: did traffic actually split in the
intended ratio? A bug in bucketing, logging, or redirect logic can quietly put
more users into one arm, and *every* downstream test becomes biased. The check
is a chi-square goodness-of-fit of observed counts against the expected split.

Threshold convention: flag as SRM if p < 0.001 (stricter than the usual 0.05
test alpha) to avoid false alarms on large samples. See LukSyen (2019),
"How to detect SRM".
"""
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

SRM_ALPHA = 0.001  # conservative; false alarms are costly on large samples


def srm_test(observed, expected_ratio, alpha: float = SRM_ALPHA) -> dict:
    """Chi-square goodness-of-fit test against an expected split.

    Parameters
    ----------
    observed : array of int
        Counts per arm, e.g. [n_a, n_b] or [n_a, n_b, n_c].
    expected_ratio : array of float
        Expected share of each arm, must sum to 1, e.g. [0.5, 0.5].
    alpha : float
        Flag as mismatch if p < alpha.
    """
    observed = np.asarray(observed, dtype=float)
    expected_ratio = np.asarray(expected_ratio, dtype=float)
    if not np.isclose(expected_ratio.sum(), 1.0):
        raise ValueError(f"expected_ratio must sum to 1, got {expected_ratio.sum()}")
    expected = expected_ratio * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    pct_obs = observed / observed.sum()
    return {
        "observed": observed.tolist(),
        "expected_ratio": expected_ratio.tolist(),
        "observed_ratio": pct_obs.tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_detected": bool(p < alpha),
        "alpha": alpha,
    }


def _print(result: dict, label: str):
    flag = "❌ SRM DETECTED" if result["srm_detected"] else "✅ OK"
    print(f"[{label}] {flag}")
    print(f"  observed: {result['observed']}  "
          f"(ratios: {[f'{r:.4f}' for r in result['observed_ratio']]})")
    print(f"  expected ratio: {result['expected_ratio']}")
    print(f"  chi2={result['chi2']:.2f}  p={result['p_value']:.4g}  "
          f"(flag if p < {result['alpha']})\n")


def main():
    print("=== Sample Ratio Mismatch (SRM) Test ===\n")

    _print(srm_test([5012, 4988], [0.5, 0.5]), "clean 50/50 split")
    _print(srm_test([5200, 4800], [0.5, 0.5]), "mismatch 52/48")
    _print(srm_test([7000, 3000], [0.7, 0.3]), "clean 70/30 split")
    _print(srm_test([7300, 2700], [0.7, 0.3]), "mismatch on 70/30")
    _print(srm_test([5000, 5000, 5000], [1/3, 1/3, 1/3]), "clean 3-way A/B/C")

    # Demonstrate: large samples make even tiny deviations detectable
    print("[large sample, 1% traffic leak on 1M users]")
    _print(srm_test([500000, 500000], [0.5, 0.5]), "exact")
    _print(srm_test([505000, 495000], [0.5, 0.5]), "1% leak — flagged")


if __name__ == "__main__":
    main()
