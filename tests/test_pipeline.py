import pytest
from gpry.run import Runner
from model_generator import *

import numpy as np


def _test_pipeline(model, gpr="RBF", gp_acquisition="LogExp",
                   convergence_criterion="CorrectCounter", callback=None,
                   callback_is_MPI_aware=False, options={},
                   checkpoint="files", load_checkpoint="overwrite", verbose=3,
                   mc_sampler="mcmc", desired_kl=0.05):
    # Sets up a runner, calls run and mc and checks whether it converged within reason.
    runner = Runner(model, gpr=gpr, gp_acquisition=gp_acquisition,
                    convergence_criterion=convergence_criterion, callback=callback,
                    callback_is_MPI_aware=callback_is_MPI_aware, options=options,
                    checkpoint=checkpoint, load_checkpoint=load_checkpoint,
                    verbose=verbose)
    # Call runner.run to test the BO loop
    runner.run()
    assert runner.has_run, "The run hasn't completed."
    assert runner.has_converged, (
        "The run hasn't converged within the given number of samples.")

    # Call the MC sampler on the GP
    surr_info, sampler = runner.generate_mc_sample(sampler=mc_sampler)
    runner.plot_mc(surr_info, sampler)

    # Compare with the true function to get the KL divergence
    s = sampler.products()["sample"]
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
    generator = Random_gaussian(ndim=dim)
    model = generator.get_model()
    _test_pipeline(model)
