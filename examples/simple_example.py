"""
Example code for a simple GP Characterization of a likelihood.
"""

import os
from gpry.mpi import is_main_process
from gpry.plots import getdist_add_training, plot_distance_distribution, \
    plot_2d_model_acquisition

# Building the likelihood
from scipy.stats import multivariate_normal
import numpy as np

mean = [3, 2]
cov = [[0.5, 0.4], [0.4, 1.5]]
rv = multivariate_normal(mean, cov)

# Sampler to use for tests: "mcmc" or "polychord"
mc_sampler = "mcmc"

def lkl(x, y):
    return np.log(rv.pdf(np.array([x, y]).T))


def callback(model, current_gpr, gp_acquisition, convergence_criterion, options, progress,
             previous_gpr, new_X, new_y, y_pred):
    print("_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_")
    print(current_gpr)
    print("Current kernel:", current_gpr.kernel_)  # rescaled?????
    print(previous_gpr)
    print("New points")
    for x, y in zip(new_X, new_y):
        print(x, y)
    print(convergence_criterion)
    # Plot distribution of points, and contours of model and acquisition
    plot_distance_distribution(current_gpr.X_train, mean, cov)
    plt.savefig("images/Distance_distribution.png", dpi=300)
    plot_distance_distribution(current_gpr.X_train, mean, cov, density=True)
    plt.savefig("images/Distance_density_distribution.png", dpi=300)
    plot_2d_model_acquisition(current_gpr, gp_acquisition, last_points=new_X)
    plt.savefig("images/Contours_model_acquisition.png", dpi=300)
    old_zeta = gp_acquisition.acq_func.zeta
    gp_acquisition.acq_func.zeta = 0
    plot_2d_model_acquisition(current_gpr, gp_acquisition, last_points=new_X)
    plt.savefig("images/Contours_model_acquisition_std.png", dpi=300)
    gp_acquisition.acq_func.zeta = old_zeta
    print("_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_")
    input("Press Enter to continue...")


# Uncomment this line to disable the example callback function above
callback = callback

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
    if "images" not in os.listdir("."):
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

options={}#'zeta_scaling': 5}

# Run the GP
from gpry.run import run
checkpoint = "output/simple"
model, gpr, acquisition, convergence, options = run(
    model, callback=callback, checkpoint=checkpoint, load_checkpoint="overwrite",
    options=options)

# Run the MCMC and extract samples
from gpry.run import mc_sample_from_gp

updated_info, sampler = mc_sample_from_gp(
    gpr, model.prior.bounds(confidence_for_unbounded=0.99995),
    paramnames=model.parameterization.sampled_params(),
    convergence=convergence, sampler=mc_sampler, output="chains/gp_model")

# Plotting
if is_main_process:
    from getdist.mcsamples import MCSamplesFromCobaya
    import getdist.plots as gdplt
    gdsamples_gp = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(
        gdsamples_gp, model.parameterization.sampled_params(), filled=True)
    getdist_add_training(gdplot, model, gpr)
    plt.savefig("images/Surrogate_triangle.png", dpi=300)

#############################################################
# Validation part

# TODO: remove: we can use getdist's Gaussian mixtures for this

# # MCMC run on the actual function
# # NB: we don't re-initialise the model, but use the one defined above

# # Optional: define an output driver
# from cobaya.output import get_output
# out = get_output(prefix="chains/truth", resume=False, force=True)

# from cobaya.sampler import get_sampler
# if mc_sampler.lower() == "mcmc":
#     info_sampler = {"mcmc": {"Rminus1_stop": 0.005, "max_tries": 1e6}}
# elif mc_sampler.lower() == "polychord":
#     info_sampler = {"polychord": {"nlive": "50d", "num_repeats": "5d"}}
# mcmc_sampler = get_sampler(info_sampler, model=model, output=out)
# success = False
# try:
#     mcmc_sampler.run()
#     success = True
# except Exception as excpt:
#     print(f"Chain failed: {str(excpt)}")
#     pass
# success = all(mpi_comm.allgather(success))
# if not success and is_main_process:
#     print("Sampling failed!")
#     exit()
# all_chains_truth = mpi_comm.gather(mcmc_sampler.products()["sample"], root=0)

if is_main_process:
    # from getdist.mcsamples import MCSamplesFromCobaya
    # upd_info = model.info()
    # upd_info["sampler"] = {"mcmc": sampler.info()}
    # gdsamples_truth = MCSamplesFromCobaya(upd_info, all_chains_truth)
    from getdist.gaussian_mixtures import GaussianND
    gdsamples_truth = GaussianND(mean, cov, names=list(info["params"]))
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples_truth, list(info["params"]), filled=True)
    plt.savefig("images/Ground_truth_triangle.png", dpi=300)
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot([gdsamples_truth, gdsamples_gp], list(info["params"]),
                         filled=[False, True],
                         legend_labels=['Truth', 'MC from GP'])
    getdist_add_training(gdplot, model, gpr)
    plt.savefig("images/Comparison_triangle.png", dpi=300)

print(gpr.X_train)
