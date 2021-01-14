"""
Example code for a simple GP Characterization of a likelihood.
"""

######## Imports ########
import os # To make directories

# numpy and scipy
from scipy.stats import multivariate_normal
import numpy as np 

# Several things to plot stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Sklearn and Sklearn gp minimize
from gpry.acquisition_functions import Expected_improvement
from gpry.gpr import GaussianProcessRegressor
from gpry.kernels import RBF, ConstantKernel as C
from gpry.gp_acquisition import GP_Acquisition

rv = multivariate_normal([3,2],[[0.5, 0.4],[0.4, 1.5]])

def f(X):
    return -1 * np.log(rv.pdf(X)) 

#############################################################
# Modelling part

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-5, 1e5))
gp = GaussianProcessRegressor(kernel=kernel,
                             n_restarts_optimizer=30)
af = -1 * Expected_improvement(xi=1e-5)

a = np.linspace(-10., 10., 200)
b = np.linspace(-10., 10., 200)
A, B = np.meshgrid(a, b)

x = np.stack((A,B),axis=-1)
xdim = x.shape
x = x.reshape(-1,2)
Y = f(x)
Y = Y.reshape(xdim[:-1])

# Plot ground truth
fig = plt.figure()
im = plt.pcolor(A, B, Y, norm=LogNorm())
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
plt.savefig("images/Ground_truth.png", dpi=300)
plt.close()

#############################################################
# Training part
bnds = np.array([[-10.,10.], [-10.,10.]])
acquire = GP_Acquisition(bnds,
                 surrogate_model=gp,
                 acq_func=af,
                 acq_optimizer="sampling",
                 optimize_direction="minimize",
                 random_state=None,
                 model_queue_size=None,
                 n_restarts_optimizer=20)

init_1 = np.random.uniform(bnds[0,0], bnds[0,1], 5)
init_2 = np.random.uniform(bnds[1,0], bnds[1,1], 5)

init_X = np.stack((init_1, init_2), axis=1)
init_y = f(init_X)

acquire.surrogate_model.append_to_data(init_X, init_y, fit=True)

n_points = 2
for _ in range(20):
    new_X, new_func = acquire.multi_optimization(n_points=n_points)
    new_y = f(new_X)
    acquire.surrogate_model.append_to_data(new_X, new_y)


# Getting the prediction
gp = acquire.surrogate_model
x_gp = gp.X_train_[:,0]
y_gp = gp.X_train_[:,1]
y_fit, std_fit = gp.predict(x, return_std=True)
y_fit = y_fit.reshape(xdim[:-1])

# Plot surrogate
fig = plt.figure()
im = plt.pcolor(A, B, y_fit, norm=LogNorm())
plt.scatter(x_gp[:5], y_gp[:5], color="purple")
plt.scatter(x_gp[5:], y_gp[5:], color="black")
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

from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

# First the MCMC Run on the actual function

def true_func(x,y):
    return -1 * f(np.array([[x,y]]))

info = {"likelihood": {"true_func": true_func}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2},
    "y": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}}

updated_info, sampler = run(info)

gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(gdsamples, ["x", "y"], filled=True)
plt.savefig("images/Ground_truth_triangle.png", dpi=300)

# Second the MCMC Run on the Surrogate model

gp = acquire.surrogate_model
def callonmodel(x,y):
    return -1 * gp.predict(np.array([[x,y]]))

info = {"likelihood": {"gpsurrogate": callonmodel}}
info["params"] = {
    "x": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2},
    "y": {"prior": {"min": -10, "max": 10}, "ref": 0.5, "proposal": 0.2}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}}

updated_info, sampler = run(info)

gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(gdsamples, ["x", "y"], filled=True)
plt.savefig("images/Surrogate_triangle.png", dpi=300)
