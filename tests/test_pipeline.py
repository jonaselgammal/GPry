import pytest
from gpry.run import Runner
import gpry.mpi as mpi
from model_generator import *

import numpy as np


def _test_pipeline(model, gpr="RBF", gp_acquisition="LogExp",
                   convergence_criterion="CorrectCounter", callback=None,
                   callback_is_MPI_aware=False, options={},
                   checkpoint="files", load_checkpoint="overwrite", verbose=3,
                   mc_sampler="mcmc", desired_kl=0.05, mean=None, cov=None):
    # Sets up a runner, calls run and mc and checks whether it converged within reason.
    bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
    # gp_acquisition = {"NORA": {}}
    options["n_points_per_acq"] = model.prior.d()
    runner = Runner(model, gpr=gpr, gp_acquisition=gp_acquisition,
                    convergence_criterion=convergence_criterion, callback=callback,
                    callback_is_MPI_aware=callback_is_MPI_aware, options=options,
                    checkpoint=checkpoint, load_checkpoint=load_checkpoint,
                    verbose=verbose)
    # Call runner.run to test the BO loop
    runner.run()
    assert runner.has_run, "The run hasn't completed."
    if not runner.has_converged:
        print("*WARNING*: The run hasn't converged within the given number of samples.")

    # Call the MC sampler on the GP
    mc_sample = runner.generate_mc_sample(sampler=mc_sampler)
    if mpi.is_main_process:
        print("Plotting results...")
        runner.plot_progress()  # plots timing and convergence
        runner.plot_mc()  # plots last obtained mc samples
        runner.plot_distance_distribution()  # plots last obtained mc samples
        if mean is not None:
            import os
            import getdist.plots as gdplt
            import matplotlib.pyplot as plt
            from getdist.gaussian_mixtures import GaussianND
            gdsamples_gp = mc_sample.to_getdist()
            gdplot = gdplt.get_subplot_plotter(width_inch=5)
            to_plot = [gdsamples_gp, GaussianND(
                mean, cov, names=mc_sample.sampled_params, label="Ground truth")]
            gdplot.triangle_plot(to_plot, mc_sample.sampled_params, filled=True)
            plt.savefig(os.path.join(runner.plots_path, "Surrogate_triangle_truth.png"),
                        dpi=300)

    # Compare with the true function to get the KL divergence
    if desired_kl is None:
        return

    if mpi.is_main_process:
        print("Computing comparison metrics...")
        s = mc_sample
        x_values = s.data[s.sampled_params]
        logp = s['minuslogpost']
        logp = -logp
        weights = s['weight']
        y_values = []
        # VERY hacky, is there a better way?
        for i in range(len(x_values)):
            sample = dict(zip(
                list(model.parameterization.sampled_params()), np.array(x_values.iloc[i])))
            # Only works for uniform priors for now. Need to fix this
            y_values = np.append(y_values, model.logpost(sample) + model.logprior(sample))
        logq = np.array(y_values)
        mask = np.isfinite(logq)
        logp2 = logp[mask]
        logq2 = logq[mask]
        weights2 = weights[mask]
        kl = np.abs(np.sum(weights2 * (logp2 - logq2)) / np.sum(weights2))
        assert kl <= desired_kl, f"The desired KL value wasn't reached: {kl} < {desired_kl}"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_gaussian(dim):
    # Supported models:
    # Himmelblau, Rosenbrock, Spike, Loggaussian, Ring, Random_gaussian, Curved_degeneracy
    generator = None
    if mpi.is_main_process:
        generator = Random_gaussian(ndim=dim)
        generator.redraw()
    generator = mpi.comm.bcast(generator)
    model = generator.get_model()
    _test_pipeline(model, desired_kl=0.05, mean=generator.mean, cov=generator.cov)


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("Pass the dimensionality as first argument.")
        exit(1)
    test_gaussian(dim=int(sys.argv[1]))
