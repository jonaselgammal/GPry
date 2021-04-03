"""
Example code for a simple GP Characterization of a likelihood.
"""

# ####### Imports ########
import os  # To make directories
import sys
from copy import deepcopy  # Needed to copy our GP instance

# numpy and scipy
from scipy.stats import multivariate_normal
import numpy as np

# Several things to plot stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# GPry things needed for building the model
from gpry.acquisition_functions import Log_exp
from gpry.gpr import GaussianProcessRegressor
from gpry.kernels import RBF, ConstantKernel as C
from gpry.preprocessing import Normalize_y, Normalize_bounds
from gpry.convergence import KL_from_draw, KL_from_MC_training
from gpry.gp_acquisition import GP_Acquisition

# Cobaya things needed for building the model
from cobaya.run import run
from cobaya.model import get_model
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

rv = multivariate_normal([3, 2], [[0.5, 0.4], [0.4, 1.5]])


def f(x, y):
    return np.log(rv.pdf(np.array([x, y]).T))


# Define the likelihood and the prior of the model
info = {"likelihood": {"f": f}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2},
    "y": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2}}

model = get_model(info)

#############################################################
# Modelling part

a = np.linspace(-10., 10., 200)
b = np.linspace(-10., 10., 200)
A, B = np.meshgrid(a, b)

x = np.stack((A, B), axis=-1)
xdim = x.shape
x = x.reshape(-1, 2)
Y = -1 * f(x[:, 0], x[:, 1])
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
# Training part
bnds = model.prior.bounds(confidence_for_unbounded=0.99995)

kernel = C(1.0, (1e-3, 1e5)) * RBF([1.0]*2, np.array([[1e-3, 1e5]]*2))
gp = GaussianProcessRegressor(kernel=kernel,
                              preprocessing_X=Normalize_bounds(bnds),
                              preprocessing_y=Normalize_y(),
                              n_restarts_optimizer=20,
#                              account_for_inf=None,  # disable SVM for tests
                              noise_level=0.01)
af = Log_exp()

acquire = GP_Acquisition(bnds,
                         acq_func=af,
                         n_restarts_optimizer=20)

init_X = model.prior.sample(3)
init_y = f(init_X[:, 0], init_X[:, 1])

gp.append_to_data(init_X, init_y, fit=True)

convergence_criterion = KL_from_draw(model.prior,
                                     {"limit": 1e-2,
                                      "n_draws": 5000})
#convergence_criterion = KL_from_MC_training(model.prior, {})


n_points = 2 # Number of acquired points per step
y_s = init_y

for _ in range(10):
    old_gp = deepcopy(gp)
    new_X, y_lies, acq_vals = acquire.multi_optimization(gp, n_points=n_points)
    new_y = f(new_X[:, 0], new_X[:, 1])
    y_s = np.append(y_s, new_y)
    gp.append_to_data(new_X, new_y, fit=True)
    print(convergence_criterion.criterion_value(gp, old_gp))
    # This should stop the algorithm but since the KL divergence doesn't work
    # I turned it off.
    """
    if convergence_criterion.is_converged(gp, old_gp):
        break
    """


# Getting the prediction
x_gp = gp.X_train[:, 0]
y_gp = gp.X_train[:, 1]
y_fit, std_fit = gp.predict(x, return_std=True)
y_fit = -1 * y_fit.reshape(xdim[:-1])
std_fit = std_fit.reshape(xdim[:-1])

# Plot surrogate
fig = plt.figure()
im = plt.pcolor(A, B, y_fit, norm=LogNorm())
plt.scatter(x_gp[:3], y_gp[:3], color="purple")
plt.scatter(x_gp[3:], y_gp[3:], color="black")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim((-10, 10))
plt.ylim((-10, 10))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
plt.savefig("images/Surrogate.png", dpi=300)
plt.close()

#############################################################
# Cobaya Part

# First the MCMC Run on the actual function

info = {"likelihood": {"true_func": f}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2},
    "y": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}}

updated_info, sampler = run(info)

gdsamples_mcmc = MCSamplesFromCobaya(updated_info,
                                     sampler.products()["sample"])
gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(gdsamples_mcmc, ["x", "y"], filled=True)
plt.savefig("images/Ground_truth_triangle.png", dpi=300)

# Second the MCMC Run on the Surrogate model


def callonmodel(x, y):
    return gp.predict(np.array([[x, y]]))


info = {"likelihood": {"gpsurrogate": callonmodel}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2},
    "y": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}}

updated_info, sampler = run(info)

gdsamples_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(gdsamples_gp, ["x", "y"], filled=True)
plt.savefig("images/Surrogate_triangle.png", dpi=300)

gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot([gdsamples_mcmc, gdsamples_gp], ["x", "y"], filled=True,
                     legend_labels=['MCMC', 'GP'])
plt.savefig("images/Comparison_triangle.png", dpi=300)
