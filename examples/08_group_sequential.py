# -*- coding: utf-8 -*-
"""Example 08 — Five planned looks under one 5% alpha budget.

Scenario
--------
A pricing test fills 240,000 users per arm over five weekly looks and the PM
reads the dashboard every morning. Checking |Z| > 1.96 at each look is not a 5%
test: under a true null it fires far more often. We calibrate Pocock and
O'Brien-Fleming boundaries over the five looks, read the observed z-path
against them, and decide at which look the test may actually stop.

Run: uv run python examples/08_group_sequential.py
"""

from _pathsetup import add_scripts_to_path

add_scripts_to_path()

import numpy as np  # noqa: E402
from group_sequential import (  # noqa: E402
    calibrate,
    crossing_rate,
    obf_shape,
    pocock_shape,
    simulate_paths,
)

K, ALPHA, N_PATHS = 5, 0.05, 20_000

c_pocock, shape_pocock = calibrate(pocock_shape, K, ALPHA, N_PATHS, seed=1)
c_obf, shape_obf = calibrate(obf_shape, K, ALPHA, N_PATHS, seed=2)
bound_pocock = c_pocock * shape_pocock
bound_obf = c_obf * shape_obf
bound_naive = np.full(K, 1.96)

print("=== 08 - five looks, one 5% alpha budget ===\n")
print("z-boundaries per look (cross when |Z| exceeds):")
for name, bounds in [("Pocock", bound_pocock), ("OBF", bound_obf), ("naive", bound_naive)]:
    print(f"  {name:<7}: {[round(float(v), 2) for v in bounds]}")

null_paths = simulate_paths(N_PATHS, K, drift=0.0, seed=99)
print("\nFalse-positive rate under the null (empirical):")
print(
    f"  naive peeking {crossing_rate(null_paths, bound_naive):.3f} (inflated)"
    f" | Pocock {crossing_rate(null_paths, bound_pocock):.3f}"
    f" | OBF {crossing_rate(null_paths, bound_obf):.3f}"
)

z = simulate_paths(1, K, drift=2.6, seed=31)[0]  # what the dashboard actually showed
print("\nThe z-path the PM reads each morning:")
for k in range(K):
    naive_hit = "CROSS" if abs(z[k]) > bound_naive[k] else "     "
    print(
        f"  look {k + 1}: z={z[k]:+.2f}  naive {bound_naive[k]:.2f} {naive_hit}"
        f"   OBF {bound_obf[k]:.2f} {'CROSS' if abs(z[k]) > bound_obf[k] else ''}"
    )

first_naive = next((k + 1 for k in range(K) if abs(z[k]) > bound_naive[k]), None)
first_pocock = next((k + 1 for k in range(K) if abs(z[k]) > bound_pocock[k]), None)
first_obf = next((k + 1 for k in range(K) if abs(z[k]) > bound_obf[k]), None)
print(
    f"\nfirst crossing -- naive: look {first_naive} | Pocock: look {first_pocock} | OBF: look {first_obf}"
)

print(
    f"\nVERDICT: the naive morning read fires at look {first_naive}, but under the OBF budget the"
)
print(
    f"         boundary is not crossed until look {first_obf}. Stop at look {first_obf}, not at the"
)
print("         earlier peek -- calling it at 1.96 is the inflation the alpha budget prevents.")
