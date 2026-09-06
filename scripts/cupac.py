# -*- coding: utf-8 -*-
"""CUPAC — CUPED with Any Covariate (model-based CUPED).

Classic CUPED uses a single pre-period covariate and a scalar theta. CUPAC
generalises it: fit a model that predicts the post-period outcome from *all*
pre-period features, then adjust the outcome by the model's predictions:

    Y_adj = Y - (pred(X) - mean(pred(X)))

The adjustment removes the X-explained component, so variance drops by
~R^2 of the model. Here the model is a linear OLS (statsmodels) — the
prediction function is swappable for any ML model (e.g. gradient boosting)
without changing the adjustment or the test.

Reference: Poyarkov, Drutsa, Khalman, Gusev, Serdyukov (2016), "Accelerated
Online Controlled Experiments with CUPED" (CUPAC).
"""

import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")


def fit_model(y, X):
    """OLS: outcome ~ pre-period features. Returns the fitted model."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X)).fit()


def predict(model, X) -> np.ndarray:
    """Model predictions for pre-period features X."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return model.predict(sm.add_constant(X))


def cupac_adjust(y, X, model, center=None) -> np.ndarray:
    """Y_adj = Y - (pred(X) - center).

    `center` is the pooled mean of predictions (an estimate of E[pred(X)]).
    Centering by the *per-group* mean would make the adjustment a no-op for
    the group mean (mean(Y_adj) = mean(Y) exactly) and kill the variance
    reduction of the estimator.
    """
    pred = predict(model, X)
    if center is None:
        center = pred.mean()
    return np.asarray(y, dtype=float) - (pred - center)


def cupac_test(y_a, X_a, y_b, X_b, alpha: float = 0.05) -> dict:
    """Two-sample Welch test on model-adjusted outcomes.

    The model is fit on the control group only and applied to both groups
    (standard CUPAC). Predictions are centered by their pooled mean so the
    variance reduction of the difference estimator is real.
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    model = fit_model(y_a, X_a)
    pred_a = predict(model, X_a)
    pred_b = predict(model, X_b)
    center = float(np.concatenate([pred_a, pred_b]).mean())
    a = cupac_adjust(y_a, X_a, model, center)
    b = cupac_adjust(y_b, X_b, model, center)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = b.mean() - a.mean()
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return {
        "diff": diff,
        "se": se,
        "p_value": float(p),
        "significant": p < alpha,
        "r_squared": float(model.rsquared),
    }


def simulate_cupac(n=2000, effect=0.0, n_features=3, seed=0):
    """Outcome driven by a weighted sum of pre-period features plus noise.

    Treatment adds `effect` to the post-period outcome only; pre-period
    features are unaffected by treatment.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (2 * n, n_features))
    beta = np.linspace(0.3, 0.9, n_features)
    y = X @ beta + rng.normal(0, 1, 2 * n)
    y[n:] += effect
    return X[:n], y[:n], X[n:], y[n:]  # (X_a, y_a, X_b, y_b)


def main():
    print("=== CUPAC — Model-Based CUPED ===\n")

    # A/A calibration
    n_sims = 200
    rejects = 0
    for s in range(n_sims):
        Xa, ya, Xb, yb = simulate_cupac(seed=s)
        rejects += cupac_test(ya, Xa, yb, Xb)["significant"]
    print(f"[A/A calibration, {n_sims} sims]")
    print(f"  CUPAC rejection rate: {rejects / n_sims * 100:.1f}%  (expect ~5%)\n")

    # Variance reduction: naive vs single-covariate CUPED vs CUPAC
    Xa, ya, Xb, yb = simulate_cupac(n=3000, effect=0.05, n_features=3, seed=1)
    t, p = stats.ttest_ind(ya, yb, equal_var=False)
    se_naive = np.sqrt(np.var(ya, ddof=1) / len(ya) + np.var(yb, ddof=1) / len(yb))

    # single-covariate CUPED on the strongest feature (X[:, 2])
    import cuped

    theta = cuped.cuped_theta(np.concatenate([ya, yb]), np.concatenate([Xa[:, 2], Xb[:, 2]]))
    a_cuped = cuped.cuped_adjust(ya, Xa[:, 2], theta)
    b_cuped = cuped.cuped_adjust(yb, Xb[:, 2], theta)
    se_cuped = np.sqrt(
        np.var(a_cuped, ddof=1) / len(a_cuped) + np.var(b_cuped, ddof=1) / len(b_cuped)
    )

    res = cupac_test(ya, Xa, yb, Xb)
    print("[Variance reduction, effect=0.05, 3 features]")
    print(f"  {'method':<22} {'SE':>10} {'var reduction':>14}")
    print(f"  {'naive':<22} {se_naive:>10.5f} {'—':>14}")
    print(f"  {'CUPED (1 covariate)':<22} {se_cuped:>10.5f} {1 - se_cuped**2 / se_naive**2:>13.1%}")
    print(
        f"  {'CUPAC (3 features)':<22} {res['se']:>10.5f} {1 - res['se'] ** 2 / se_naive**2:>13.1%}"
    )
    print(f"  model R^2 = {res['r_squared']:.3f}  (theory: variance reduction ~ R^2)")
    print(f"  CUPAC diff={res['diff']:+.5f} p={res['p_value']:.4f}")


if __name__ == "__main__":
    main()
