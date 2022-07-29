import pytest
from gpry.run import Runner
from gpry.io import check_checkpoint, read_checkpoint, save_checkpoint
from model_generator import *

import numpy as np


def _test_io(load_checkpoint, gpr="RBF", gp_acquisition="LogExp",
                   convergence_criterion="CorrectCounter", callback=None,
                   callback_is_MPI_aware=False, convergence_options=None, options={},
                   checkpoint="files", verbose=3):
    # Use 2d gaussian as test model
    generator = Random_gaussian(ndim=2)
    model = generator.get_model()
    # Sets up a runner, calls run and mc and checks whether it converged within reason.
    runner = Runner(model, gpr=gpr, gp_acquisition=gp_acquisition,
                    convergence_criterion=convergence_criterion, callback=callback,
                    callback_is_MPI_aware=callback_is_MPI_aware,
                    convergence_options=convergence_options, options=options,
                    checkpoint=checkpoint, load_checkpoint=load_checkpoint,
                    verbose=verbose)
    # Call runner.run and let it converge
    runner.run()
    # Delete the old runner instance
    del runner

    # Check if the checkpoint files have been saved and can be recovered
    assert np.all(check_checkpoint(checkpoint))
    model, gpr, acquisition, convergence, options, progress = read_checkpoint(checkpoint)

    if load_checkpoint=="resume":
        runner = Runner(model,
                        checkpoint=checkpoint, load_checkpoint=load_checkpoint)
        assert runner.gpr.n_total > 0, "The loaded GP regressor doesn't seem to "\
            "contrain any training points."
        # Try a test predict
        y = runner.gpr.predict(np.atleast_2d(runner.gpr.X_train[0]))
        runner.run()
        assert runner.has_run, "The runner couldn't be run."


@pytest.mark.parametrize("load_checkpoint", ["resume"])
@pytest.mark.parametrize("convergence_criterion", ["CorrectCounter", "DontConverge"])
def test_io(load_checkpoint, convergence_criterion):
    if convergence_criterion == "CorrectCounter":
        options = {}
    else:
        options = {"max_total": 20, "max_finite":20}
    _test_io(load_checkpoint, convergence_criterion, options=options)
