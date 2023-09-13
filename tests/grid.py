"""
Script to generate grids of gaussian models for different input parameter values.

Run with arguments [dim N path] to generate N runs per parameter value with dimension dim,
and save the results in `path`.

Run with "plot" as a single argument to plot the results of a run.

The grid is defined by a `generate_input` function, that provides a dict whose values are
input arguments dicts for the Runner (except a model, which is generated separately), and
the keys are their labels.
"""
import os
import sys
from copy import deepcopy
from random import random, shuffle
import numpy as np

from gpry.gp_acquisition import GPAcquisition
from gpry.proposal import PartialProposer, CentroidsProposer
from gpry.preprocessing import Normalize_bounds
from gpry.run import Runner
from gpry.tools import kl_norm

from model_generator import Random_gaussian


_kl_truth_col = "kl_truth"

# Diagnosis mode: progress printing and plotting at every iteration.
diag = False

mean_fname = "mean.txt"
cov_fname = "cov.txt"


def generate_inputs(bounds):
    """
    Creates dict of Runner input dicts, defining the grid. Keys are the labels of the
    different input sets.
    """
    verbose = 0  # quiet, except for errors
    d = len(bounds)
    default_input = {"gpr": "RBF", "convergence_criterion": "DontConverge",
                     "verbose": verbose, "plots": False, "options": {
                         "n_initial": 3 * d,
                         "max_init": 10 * d,  # fail if sth funny: not useful for test
                         "n_points_per_acq": 1,  # 1 when not testing it in particular
                         "max_points": int(np.ceil(1.5 * n_approx_conv(d)))}}
    default_input["gp_acquisition"] = GPAcquisition(
        bounds, acq_func="LogExp",
        proposer=PartialProposer(bounds, CentroidsProposer(bounds)),
        acq_optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5 * d,
        n_repeats_propose=10, preprocessing_X=Normalize_bounds(bounds),
        zeta_scaling=1.1, verbose=verbose)
    # Creating grid -- User modifications usually from here down
    # 1. zeta values
    inputs = {}
    if d < 10:
        zetas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    else:
        zetas = [0.05, 0.1, 0.2, 0.5, 1]
        shuffle(zetas)
    for value in zetas:
        key = f"zeta_{value}"
        inputs[key] = deepcopy(default_input)
        acq = GPAcquisition(
            bounds, acq_func="LogExp",
            proposer=PartialProposer(bounds, CentroidsProposer(bounds)),
            acq_optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5 * d,
            n_repeats_propose=10, preprocessing_X=Normalize_bounds(bounds),
            zeta=value, verbose=verbose)
        inputs[key]["gp_acquisition"] = acq
    return inputs


def n_approx_conv(dim):
    """Order-of-magnitude-approx number of true posterior evaluations for convergence."""
    return 3.5 * dim**2
    # was for 2, 4, 8: 5 * dim**(5 / 3)


def random_hex_name():
    """Generates a random 8-character hexadecimal name."""
    return str(hex(int(random() * 1e9))[2:])


def callback(runner):
    """
    Runs an MCMC on the current model at specific intervals, and computes KL_truth with
    a gaussian approximation.
    """
    # Do tests only 5 times before expected convergence, and a few times after
    n_train = runner.gpr.n_total
    n_step = int(np.round(n_approx_conv(dim) / 5))
    # TODO: this will not work for n_acq != 1
    if max(n_train, 1) % n_step:
        return
    true_mean = likelihood_generator.mean
    true_cov = likelihood_generator.cov
    paramnames = list(runner.model.parameterization.sampled_params())
    surr_info, sampler = runner.generate_mc_sample(
        sampler="mcmc", add_options={"covmat": true_cov, "covmat_params": paramnames},
        output=False)
    mc_mean = sampler.products()["sample"].mean()
    mc_cov = sampler.products()["sample"].cov()
    # Compute KL_truth and save it to progress table (hacky!)
    kl = max(kl_norm(true_mean, true_cov, mc_mean, mc_cov),
             kl_norm(mc_mean, mc_cov, true_mean, true_cov))
    if _kl_truth_col not in runner.progress.data.columns:
        runner.progress.data = runner.progress.data.assign(**{_kl_truth_col: np.nan})
    runner.progress.data.iloc[-1,
                              runner.progress.data.columns.get_loc(_kl_truth_col)] = kl
    if diag:
        print(runner.progress)
        from getdist.gaussian_mixtures import MixtureND
        true_dist = MixtureND([true_mean], [true_cov], names=paramnames)
        runner.plot_mc(surr_info, sampler, add_samples={"Truth": true_dist})
        import matplotlib.pyplot as plt
        plt.show()


def generate(nruns, likelihood_generator, path=None):
    if path is None:
        path = "."
    for i_run in range(nruns):
        # TODO: MPI and randomness!
        likelihood_generator.redraw()
        this_model = likelihood_generator.get_model()
        # Needs to generate inputs *after* model, bc they need bounds
        bounds = this_model.prior.bounds(confidence_for_unbounded=0.99995)
        inputs = generate_inputs(bounds)
        for this_label, this_input in inputs.items():
            this_path = os.path.join(path, this_label, random_hex_name())
            # Write mean and covmat
            np.savetxt(os.path.join(this_path, mean_fname), likelihood_generator.mean)
            np.savetxt(os.path.join(this_path, cov_fname), likelihood_generator.cov)
            # Do the run
            this_input["checkpoint"] = this_path
            this_input["load_checkpoint"] = "overwrite"
            this_input["callback"] = callback
            runner = Runner(this_model, **this_input)
            runner.run()
        print(f"--- Generated {len(inputs)} models.")


if __name__ == "__main__":
    # contains a list of values for grid_parameter
    arg_err_msg = "Pass [dim N path] as arguments, or ['plot' path] as single argument."
    if len(sys.argv[1:]) != 3:
        raise ValueError(arg_err_msg)
    try:
        # Assuming expected input
        global dim
        dim = int(sys.argv[1])
        nruns = int(sys.argv[2])
        path = str(sys.argv[3])
    except ValueError as excpt:
        raise ValueError(arg_err_msg + " Error: " + str(excpt)) from excpt
    path = os.path.join(path, f"d{dim}")
    global likelihood_generator  # so that the callback function can read mean and covmat
    likelihood_generator = Random_gaussian(ndim=dim, prior_size_in_std=5)
    print(f"Generating {nruns} runs per value in the grid...")
    generate(nruns, likelihood_generator, path=path)
    print("Finished generation.")
