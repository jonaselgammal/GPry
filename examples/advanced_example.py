"""
Advanced example showing off the deeper functionality and modules of the
algorithm.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create likelihood
def log_lkl(x_1, x_2):
    return  -(10*(0.45-x_1))**2./4. - (20*(x_2/4.-x_1**4.))**2.

# Construct model instance
from cobaya.model import get_model
info = {"likelihood": {"curved_degeneracy": log_lkl}}
info["params"] = {
    "x_1": {"prior": {"min": -0.5, "max": 1.5}},
    "x_2": {"prior": {"min": -0.5, "max": 2.}}
    }
model = get_model(info)

# Dimensionality and prior bounds
n_d = model.prior.d()
prior_bounds = model.prior.bounds()

# Construct kernel
from gpry.kernels import Matern, ConstantKernel as C
kernel = C(1.0, (1e-3, 1e3)) * Matern([0.1]*n_d, [[1e-5, 1e5]]*n_d, nu=2.5)

#Construct GP
from gpry.gpr import GaussianProcessRegressor
from gpry.preprocessing import Normalize_bounds
gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,
    preprocessing_X=Normalize_bounds(prior_bounds)
    )

# Construct Acquisition function
from gpry.acquisition_functions import LogExp
af = LogExp(zeta=0.1)

# Construct Acquisition procedure
from gpry.gp_acquisition import GP_Acquisition
acq = GP_Acquisition(
    prior_bounds,
    acq_func=af,
    n_restarts_optimizer=10,
    preprocessing_X=Normalize_bounds(prior_bounds)
    )

# Construct convergence criterion
from gpry.convergence import GaussianKL
conv = GaussianKL(model.prior, {"limit": 1e-2})

# Construct options dictionary
options = {"max_init": 100, "max_points": 200,
           "n_initial": 8, "n_points_per_acq": 2}

# Run the GP
from gpry.run import Runner
checkpoint = "output/advanced"
runner = Runner(
    model, gpr=gpr, gp_acquisition=acq, convergence_criterion=conv, options=options,
    checkpoint=checkpoint, load_checkpoint="resume")
runner.run()

# Run the MCMC
options = {"Rminus1_stop": 0.01, "max_tries": 1000}
updated_info_gp, sampler_gp = runner.generate_mc_sample(add_options=options)

# Validate with cobaya
from cobaya.run import run as cobaya_run
info["sampler"] = {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 1000}}
updated_info_mcmc, sampler_mcmc = cobaya_run(info)

# Extracting samples
from getdist.mcsamples import MCSamplesFromCobaya
gdsamples_gp = MCSamplesFromCobaya(updated_info_gp,
                                   sampler_gp.products()["sample"])
gdsamples_mcmc = MCSamplesFromCobaya(updated_info_mcmc,
                                     sampler_mcmc.products()["sample"])
# Plot triangle plot with getdist
import getdist.plots as gdplt
from gpry.plots import getdist_add_training
gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot([gdsamples_mcmc, gdsamples_gp],
                     ["x_1", "x_2"], filled=[False, True],
                     legend_labels=['MCMC', 'GPry'])
getdist_add_training(gdplot, model, runner.gpr)
plt.savefig(os.path.join(checkpoint, "images/Comparison_triangle.png"), dpi=300)

# Plot convergence
from gpry.plots import plot_convergence
plot_convergence(runner.convergence, evaluations="accepted")
plt.savefig(os.path.join(checkpoint, "images/convergence.png"), dpi=300)
