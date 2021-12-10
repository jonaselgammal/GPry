"""
Example code for LCDM with the Planck lite likelihood (1 nuisance parameter)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from gpry.mpi import is_main_process, mpi_comm
from gpry.plots import getdist_add_training

# Building the likelihood
# (does programmatically the same as calling cobaya-cosmo-generator)
from cobaya.cosmo_input import create_input
from cobaya.model import get_model

# Choose classy (CLASS) or camb (CAMB)
theory_code = "classy"
theory_code = "camb"

preset = "planck_2018_" + theory_code.lower()
info = create_input(preset=preset)
# Substitute high-ell likelihood by Planck-lite
info["likelihood"].pop("planck_2018_highl_plik.TTTEEE")
info["likelihood"]["planck_2018_highl_plik.TTTEEE_lite_native"] = None

# Temporary solution: reduce priors: (CAMB case only)
# See https://wiki.cosmos.esa.int/planck-legacy-archive/images/2/21/Baseline_params_table_2018_95pc_v2.pdf
info["params"]["logA"]["prior"] = {"min": 2.9, "max": 3.2}
info["params"]["ns"]["prior"] = {"min": 0.95, "max": 0.98}
info["params"]["theta_MC_100"]["prior"] = {"min": 1.035, "max": 1.05}
info["params"]["ombh2"]["prior"] = {"min": 0.022, "max": 0.023}
info["params"]["omch2"]["prior"] = {"min": 0.11, "max": 0.13}
info["params"]["tau"]["prior"] = {"min": 0.03, "max": 0.08}
info["params"]["A_planck"] = 1.00044  # fixed for now

model = get_model(info)

# Run the GP
from gpry.run import run

kwargs = {
    "convergence_criterion": "ConvergenceCriterionGaussianMCMC",
    "convergence_options": {"limit": 2e-2}}

model, gpr, acquisition, convergence, options = run(model, **kwargs)


# Run the MCMC and extract samples
from gpry.run import mcmc
updated_info, sampler = mcmc(model, gpr, convergence, output="chains/gp_model")

# Plotting
if is_main_process:
    sampled_params_names = list(model.parameterization.sampled_params())
    from getdist.mcsamples import MCSamplesFromCobaya
    import getdist.plots as gdplt
    gdsamples_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples_gp, sampled_params_names, filled=True)
    getdist_add_training(gdplot, model, gpr)
    plt.savefig("images/Planck_surrogate_triangle.png", dpi=300)

#############################################################
# Validation part

# If Planck chains downloaded from somewhere, plot them on top

# TODO
