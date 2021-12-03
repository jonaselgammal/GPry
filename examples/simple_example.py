"""
Example code for a simple GP Characterization of a likelihood.
"""

import os
from gpry.mpi import is_main_process

# Building the likelihood
from scipy.stats import multivariate_normal
import numpy as np

rv = multivariate_normal([3, 2], [[0.5, 0.4], [0.4, 1.5]])


def lkl(x, y):
    return np.log(rv.pdf(np.array([x, y]).T))

#############################################################
# Plotting the likelihood

if is_main_process:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    a = np.linspace(-10., 10., 200)
    b = np.linspace(-10., 10., 200)
    A, B = np.meshgrid(a, b)

    x = np.stack((A, B), axis=-1)
    xdim = x.shape
    x = x.reshape(-1, 2)
    Y = -1 * lkl(x[:, 0], x[:, 1])
    Y = Y.reshape(xdim[:-1])

    # Plot ground truth
    fig = plt.figure()
    im = plt.pcolor(A, B, Y, norm=LogNorm())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    if not "images" in os.listdir("."):
        os.makedirs("images")
    plt.savefig("images/Ground_truth.png", dpi=300)
    plt.close()

#############################################################

# Define the model (containing the prior and the likelihood)
from cobaya.model import get_model

info = {"likelihood": {"normal": lkl}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}},
    "y": {"prior": {"min": -10, "max": 10}}
    }

model = get_model(info)

# Run the GP
from gpry.run import run
model, gpr, acquisition, convergence, options = run(model)

# Run the MCMC and extract samples
from gpry.run import mcmc
updated_info, sampler = mcmc(model, gpr, convergence)

# Plotting
if is_main_process:
    from getdist.mcsamples import MCSamplesFromCobaya
    import getdist.plots as gdplt
    gdsamples_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples_gp, ["x", "y"], filled=True)
    plt.savefig("images/Surrogate_triangle.png", dpi=300)

#############################################################
# Validation part

# MCMC run on the actual function

from cobaya.run import run as cobaya_run
info = {"likelihood": {"true_lkl": lkl}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2},
    "y": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 1000}}

updated_info, sampler = cobaya_run(info)

if is_main_process:
    gdsamples_mcmc = MCSamplesFromCobaya(updated_info,
                                         sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples_mcmc, ["x", "y"], filled=True)
    plt.savefig("images/Ground_truth_triangle.png", dpi=300)

    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot([gdsamples_mcmc, gdsamples_gp], ["x", "y"], filled=True,
                         legend_labels=['MCMC', 'GP'])
    plt.savefig("images/Comparison_triangle.png", dpi=300)
