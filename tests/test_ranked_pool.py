"""
Testing speed and effectiveness of different RankedPool creation approaches.

- Creates a likelihood, samples some starting points from it and fits a GP.
- Samples many more points and tries to add them.

"""

dim = 8
n_start = 20 * dim
n_add = 1000
bounds = -1, 1

from time import time
from functools import partial

import numpy as np

import gpry.mpi as mpi
from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.acquisition_functions import LogExp
from gpry.gp_acquisition import RankedPool

from model_generator import Random_gaussian

gpr = None
points_add = None
y_add = None
if mpi.is_main_process:
    generator = Random_gaussian(ndim=dim)
    generator.redraw()
    points_start = generator.rv.rvs(n_start)
    y_start_truth = generator.rv.logpdf(points_start)
    gpr = GPR(bounds=np.array([list(bounds)] * dim), account_for_inf=None,
              n_restarts_optimizer=2*dim)
    gpr.append_to_data(points_start, y_start_truth, fit=True, simplified_fit=False)
    points_add = generator.rv.rvs(n_add)
    y_add, sigma_add = gpr.predict(points_add, return_std=True)
    print("")
    print("GPR:", gpr)
    print("Fitted kernel:", gpr.kernel_)
    print("")
gpr, points_add, y_add, sigma_add = mpi.comm.bcast([gpr, points_add, y_add, sigma_add])
acq_func = LogExp(dimension=dim, zeta_scaling=-0.89)
acq_func_y_sigma = partial(
    acq_func.f, baseline=gpr.y_max,
    noise_level=gpr.noise_level, zeta=acq_func.zeta
)

acq_add = acq_func_y_sigma(y_add, sigma_add)

print("METHOD: add_one, random order")
pool = RankedPool(size=dim, gpr=gpr, acq_func=acq_func_y_sigma, verbose=0)
start = time()
pool.add(points_add, y_add, sigma_add, acq_add, method="single")
print(f"TOOK {time() - start} sec.")
print(f"#caches: {pool._cache_counter}")
pool.log_pool(level=0)

print("\n-----------------------------------------------------------------------------\n")

print("METHOD: add_one, descending acq order")
pool = RankedPool(size=dim, gpr=gpr, acq_func=acq_func_y_sigma, verbose=0)
start = time()
pool.add(points_add, y_add, sigma_add, acq_add, method="single sort acq")
print(f"TOOK {time() - start} sec.")
print(f"#caches: {pool._cache_counter}")
pool.log_pool(level=0)

print("\n-----------------------------------------------------------------------------\n")

print("METHOD: add_one, descending logpost order")
pool = RankedPool(size=dim, gpr=gpr, acq_func=acq_func_y_sigma, verbose=0)
start = time()
pool.add(points_add, y_add, sigma_add, acq_add, method="single sort acq y")
print(f"TOOK {time() - start} sec.")
print(f"#caches: {pool._cache_counter}")
pool.log_pool(level=0)

print("\n-----------------------------------------------------------------------------\n")

print("METHOD: add_bulk, random order")
pool = RankedPool(size=dim, gpr=gpr, acq_func=acq_func_y_sigma, verbose=0)
start = time()
pool.add(points_add, y_add, sigma_add, acq_add, method="bulk")
print(f"TOOK {time() - start} sec.")
print(f"#caches: {pool._cache_counter}")
pool.log_pool(level=0)
