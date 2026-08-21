# -*- coding: utf-8 -*-
"""Bayesian A/B testing (analytic, no MCMC).

Conversions (Beta-Binomial):
    prior:      theta ~ Beta(a, b)          (a=b=1 -> uniform)
    posterior:  theta | data ~ Beta(a + s, b + f)   (conjugate update)
Reports, all from posterior samples:
    - P(B > A)
    - P(A > B)
    - P(practically equivalent) within a ROPE
    - expected loss of choosing each arm (risk of the wrong call)
    - posterior 95% credible interval of the lift

Continuous metric (Normal-Normal with known variance):
    conjugate normal posterior for the mean; same decision quantities.

The decision rule: pick the arm with smaller expected loss; if expected loss
of both exceeds a threshold, keep experimenting.
"""
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


def beta_binomial(s, f, prior=(1.0, 1.0), n_samples=200_000, seed=0):
    """Posterior-decision quantities for a 2-arm binomial test.

    Parameters
    ----------
    s, f : (s_a, s_b), (f_a, f_b)  — successes and failures per arm.
    prior : (a, b) Beta prior shape, shared by both arms.
    rope : half-width of region of practical equivalence for theta_B - theta_A.
    """
    a0, b0 = prior
    rng = np.random.default_rng(seed)
    (s_a, s_b), (f_a, f_b) = s, f
    post_a = rng.beta(a0 + s_a, b0 + f_a, n_samples)
    post_b = rng.beta(a0 + s_b, b0 + f_b, n_samples)
    return post_a, post_b


def decide(post_a, post_b, rope=0.0):
    diff = post_b - post_a
    p_b_better = float(np.mean(diff > 0))
    p_a_better = float(np.mean(diff < 0))
    p_equiv = float(np.mean(np.abs(diff) <= rope))
    # expected loss: choose B -> risk that A was actually better (lost uplift)
    loss_b = float(np.mean(np.where(diff < 0, -diff, 0.0)))   # E[max(A-B, 0)]
    loss_a = float(np.mean(np.where(diff > 0, diff, 0.0)))    # E[max(B-A, 0)]
    ci = np.quantile(diff, [0.025, 0.975])
    return {
        "lift_mean": float(np.mean(diff)), "lift_median": float(np.median(diff)),
        "p_b_better": p_b_better, "p_a_better": p_a_better,
        "p_equiv_within_rope": p_equiv, "rope": rope,
        "expected_loss_choose_b": loss_b, "expected_loss_choose_a": loss_a,
        "ci95": (float(ci[0]), float(ci[1])),
        "decision": _pick(p_b_better, p_a_better, p_equiv, loss_a, loss_b, rope),
    }


def _pick(p_b, p_a, p_eq, loss_a, loss_b, rope):
    if p_eq >= 0.95:
        return "equivalent (within ROPE)"
    if loss_b < loss_a:
        return "ship B" if loss_b < 1e-4 else "lean B (keep testing)"
    if loss_a < loss_b:
        return "keep A" if loss_a < 1e-4 else "lean A (keep testing)"
    return "inconclusive"


def normal_normal(mean_a, var_a, n_a, mean_b, var_b, n_b,
                  prior_var=1e9, n_samples=200_000, seed=0):
    """Normal-Normal conjugate posterior for the mean of each arm (known variance).

    Posterior mean ~ N(mu_post, sigma_post^2) where
        1/sigma_post^2 = 1/prior_var + n / data_var.
    """
    rng = np.random.default_rng(seed)
    post_var_a = 1.0 / (1.0 / prior_var + n_a / var_a)
    post_mean_a = post_var_a * (mean_a * n_a / var_a)
    post_var_b = 1.0 / (1.0 / prior_var + n_b / var_b)
    post_mean_b = post_var_b * (mean_b * n_b / var_b)
    post_a = rng.normal(post_mean_a, np.sqrt(post_var_a), n_samples)
    post_b = rng.normal(post_mean_b, np.sqrt(post_var_b), n_samples)
    return post_a, post_b


def main():
    print("=== Bayesian A/B Testing ===\n")

    # --- Binomial: B clearly better ---
    post_a, post_b = beta_binomial(s=(120, 150), f=(1880, 1850), prior=(1, 1), seed=1)
    d = decide(post_a, post_b, rope=0.005)
    print("[Conversion test] A=120/2000 (6.0%), B=150/2000 (7.5%), ROPE=0.005")
    print(f"  P(B > A) = {d['p_b_better']:.4f}   P(A > B) = {d['p_a_better']:.4f}")
    print(f"  P(equivalent within ROPE) = {d['p_equiv_within_rope']:.4f}")
    print(f"  lift (theta_B - theta_A): mean={d['lift_mean']:.4f}, "
          f"median={d['lift_median']:.4f}, 95% CrI=[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]")
    print(f"  expected loss: choose B = {d['expected_loss_choose_b']:.5f}, "
          f"choose A = {d['expected_loss_choose_a']:.5f}")
    print(f"  decision: {d['decision']}\n")

    # --- Binomial: A/A, no difference ---
    post_a, post_b = beta_binomial(s=(100, 102), f=(1900, 1898), prior=(1, 1), seed=2)
    d = decide(post_a, post_b, rope=0.005)
    print("[A/A test] A=100/2000, B=102/2000, ROPE=0.005")
    print(f"  P(B > A) = {d['p_b_better']:.4f}   P(A > B) = {d['p_a_better']:.4f}")
    print(f"  P(equivalent within ROPE) = {d['p_equiv_within_rope']:.4f}")
    print(f"  decision: {d['decision']}\n")

    # --- Continuous: ARPU, normal approx ---
    post_a, post_b = normal_normal(mean_a=2.10, var_a=9.0, n_a=5000,
                                    mean_b=2.25, var_b=9.0, n_b=5000, seed=3)
    d = decide(post_a, post_b, rope=0.05)
    print("[Continuous test] A mean=2.10, B mean=2.25, sd=3, n=5000, ROPE=0.05")
    print(f"  P(B > A) = {d['p_b_better']:.4f}")
    print(f"  lift: mean={d['lift_mean']:.4f}, 95% CrI=[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]")
    print(f"  expected loss: choose B = {d['expected_loss_choose_b']:.5f}, "
          f"choose A = {d['expected_loss_choose_a']:.5f}")
    print(f"  decision: {d['decision']}")

    # --- Calibration: A/A over many sims, P(B>A) should be ~uniform ---
    rng = np.random.default_rng(0)
    ps = []
    for s in range(2000):
        sa = rng.binomial(2000, 0.10); sb = rng.binomial(2000, 0.10)
        pa, pb = beta_binomial(s=(sa, sb), f=(2000 - sa, 2000 - sb), n_samples=400, seed=s)
        ps.append(np.mean(pb > pa))
    ps = np.array(ps)
    print(f"\n[Calibration, 2000 A/A sims] P(B>A) mean={ps.mean():.3f} "
          f"(expect ~0.5), frac <=0.05={np.mean(ps<=0.05)*100:.1f}% "
          f"(expect ~5%)")


if __name__ == "__main__":
    main()
