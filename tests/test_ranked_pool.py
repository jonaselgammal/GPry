"""
Testing speed and effectiveness of different RankedPool creation approaches.

- Creates a likelihood, samples some starting points from it and fits a GP.
- Samples many more points and tries to add them.

IMPORTANT: speed and performance stability of ranking benefits greatly from fixing the
number of threads (e.g. OMP_NUM_THREADS=X). Otherwise the overhead is very large (many
individual calls to thread-parallelised funcs that will check for free cores).
"""

dim = 8
n_start = 20 * dim
n_add = 1000 * dim
bounds = -1, 1
test_methods = ["single", "single sort acq", "single sort y", "bulk"]
log_pool = False
n_tests = 50

from time import time
from functools import partial

import numpy as np

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.acquisition_functions import LogExp
from gpry.gp_acquisition import RankedPool

from model_generator import Random_gaussian

print(f"Testing for dim={dim}. Training with {n_start} points. Adding {n_add} points.")
print(f"Will test methods {test_methods}")
print("Creating GPR...")

generator = Random_gaussian(ndim=dim)
generator.redraw()
points_start = generator.rv.rvs(n_start)
y_start_truth = generator.rv.logpdf(points_start)
gpr = GPR(bounds=np.array([list(bounds)] * dim), account_for_inf=None,
          n_restarts_optimizer=2*dim)
gpr.append_to_data(points_start, y_start_truth, fit=True, simplified_fit=False)
print("")
print("GPR:", gpr)
print("Fitted kernel:", gpr.kernel_)
print("")

print("Generating points to be added...")
points_add = [generator.rv.rvs(n_add) for _ in range(n_tests)]
y_add, sigma_add = np.concatenate(
    [np.array(gpr.predict(p, return_std=True)).T for p in points_add]).T
y_add = [y_add[i:i + n_add] for i in range(n_tests)]
sigma_add = [sigma_add[i:i + n_add] for i in range(n_tests)]
acq_func = LogExp(dimension=dim, zeta_scaling=-0.89)
acq_func_y_sigma = partial(
    acq_func.f, baseline=gpr.y_max,
    noise_level=gpr.noise_level, zeta=acq_func.zeta
)
acq_add = [acq_func_y_sigma(y, s) for y, s in zip(y_add, sigma_add)]

# Tests!
for method in test_methods:
    print("\n-------------------------------------------------------------------------\n")
    print(f"TESTING METHOD: {method}")
    n_caches = []
    start = time()
    for p, y, s, a in zip(points_add, y_add, sigma_add, acq_add):
        pool = RankedPool(size=dim, gpr=gpr, acq_func=acq_func_y_sigma, verbose=0)
        pool.add(p, y, s, a, method=method)
        n_caches.append(pool.cache_counter)
        if log_pool:
            pool.log_pool(level=0)
    print(f"TOOK avg {(time() - start) / n_tests} sec.")
    print(f"Avg #caches: {np.mean(n_caches)}")
