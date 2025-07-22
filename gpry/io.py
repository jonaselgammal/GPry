"""
This module provides some handy methods for creating paths and pickling and
saving/checking/loading checkpoint files.

Under normal circumstances you shouldn't have to use any of the methods in here if you use
the :class:`run.Runner` class to run GPry.
"""

import os

import dill as pickle  # type: ignore

from gpry.surrogate import SurrogateModel
from gpry.truth import get_truth, Truth

_checkpoint_filenames = {
    "truth": "tru.pkl",
    "surrogate": "sur.pkl",
    "acquisition": "acq.pkl",
    "convergence": "con.pkl",
    "options": "opt.pkl",
    "progress": "pro.pkl",
}

# For backwards compatibility (TODO: deprecate)
_checkpoint_filename_model = "mod.pkl"


def create_path(path, verbose=True):
    """
    Creates a path if it doesn't exist already and prints a message if creating a new
    directory. If the directory already exits it does nothing.

    Parameters
    ----------
    path : string or path
        The path which shall be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            print("Successfully created the directory %s" % path)


def check_checkpoint(path=None):
    """
    Checks if there are checkpoint files in a specific location and if so if they
    are complete. Returns a list of bools.

    Parameters
    ----------
    path : string, optional
        The path where the files are located. If ``None``, reports files as non-found.

    Returns
    -------
    A boolean array containing whether the files exist in the specified
    location in the following order:
    [truth, gp, acquisition, convergence, options]
    """
    if path is None:
        return [False] * len(_checkpoint_filenames)
    return [
        os.path.exists(os.path.join(path, f)) for f in _checkpoint_filenames.values()
    ]


def read_checkpoint(path, truth=None):
    """
    Loads checkpoint files to be able to resume a run or save the results for
    further processing.

    Parameters
    ----------
    path : string
        The path where the files are located.

    truth : gpry.truth.Truth, optional
        If passed, it will be used instead of the loaded one.

    Returns
    -------
    (truth, surrogate, acquisition, convergence, options, progress)
    If any of the files does not exist or cannot be read the function will
    return None instead.
    """
    # Check if a file exists in the checkpoint and if so resume from there.
    checkpoint_files = check_checkpoint(path)
    # Read in checkpoint
    if truth is not None and not isinstance(truth, Truth):
        raise ValueError(
            "If 'truth' is not None, it must be a gpry.truth.Truth instance."
        )
    if truth is None:
        with open(os.path.join(path, _checkpoint_filenames["truth"]), "rb") as i:
            truth = pickle.load(i) if checkpoint_files[0] else None
        # Backwards compatibility: load Cobaya model (TODO: deprecate soon)
        filename_model = os.path.exists(os.path.join(path, _checkpoint_filename_model))
        if truth is None and os.path.exists(filename_model):
            with open(filename_model, "rb") as i:
                truth = {"loglike": pickle.load(i)}
        truth = get_truth(**truth)
    with open(os.path.join(path, _checkpoint_filenames["surrogate"]), "rb") as i:
        surrogate = pickle.load(i) if checkpoint_files[1] else None
    with open(os.path.join(path, _checkpoint_filenames["acquisition"]), "rb") as i:
        acquisition = pickle.load(i) if checkpoint_files[2] else None
    with open(os.path.join(path, _checkpoint_filenames["convergence"]), "rb") as i:
        convergence = pickle.load(i) if checkpoint_files[3] else None
    with open(os.path.join(path, _checkpoint_filenames["options"]), "rb") as i:
        options = pickle.load(i) if checkpoint_files[5] else None
    with open(os.path.join(path, _checkpoint_filenames["progress"]), "rb") as i:
        progress = pickle.load(i) if checkpoint_files[4] else None
    return truth, surrogate, acquisition, convergence, options, progress


def save_checkpoint(
    path, truth, surrogate, acquisition, convergence, options, progress
):
    """
    This function is used to save all relevant parts of the GP loop for reuse
    as checkpoint in case the procedure crashes.
    This function creates ``.pkl`` files which contain the instances
    of the different modules.
    The files can be loaded with the read_checkpoint function.

    Parameters
    ----------
    path : The path where the files shall be saved
        The files will be saved as *path* +(mod, sur, acq, con, opt).pkl

    truth : Truth

    surrogate : SurrogateModel

    acquisition : GPAcquisition

    convergence : Convergence_criterion

    options : dict

    progress : Progress instance
    """
    if path is None:
        return
    create_path(path, verbose=False)
    try:
        if truth is not None:
            with open(os.path.join(path, _checkpoint_filenames["truth"]), "wb") as f:
                pickle.dump(truth.as_dict(), f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["surrogate"]), "wb") as f:
            pickle.dump(surrogate, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["acquisition"]), "wb") as f:
            pickle.dump(acquisition, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["convergence"]), "wb") as f:
            pickle.dump(convergence, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["options"]), "wb") as f:
            pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["progress"]), "wb") as f:
            pickle.dump(progress, f, pickle.HIGHEST_PROTOCOL)
    except Exception as excpt:
        raise RuntimeError(
            "Could not save the checkpoint. Check if the path "
            "is correct and exists. Error message: " + str(excpt)
        ) from excpt


def ensure_surrogate(
    surrogate,
    truth=None,
    acquisition=None,
    convergence=None,
    options=None,
    progress=None,
):
    """
    Returns (if instance passed) or loads (if string) the given surrogate model and
    associated objects.

    If loading, any object passed as a keyword will be preferred to the loaded one.

    Parameters
    ----------
    surrogate : SurrogateModel

    truth : Truth

    acquisition : GPAcquisition, optional

    convergence : Convergence_criterion, optional

    options : dict, optional

    progress : Progress instance, optional

    Returns
    -------
    (truth, surrogate, acquisition, convergence, options, progress)
    If any of the files does not exist or cannot be read the function will
    return None instead.
    """
    if not isinstance(surrogate, (str, SurrogateModel)):
        raise TypeError(
            "`surrogate` needs to be a gpry SurrogateModel or a string "
            "with a path to a checkpoint file."
        )
    if isinstance(surrogate, str):
        truth_, surrogate, acq_, conv_, opt_, prog_ = read_checkpoint(
            surrogate, truth=truth
        )
    else:
        truth_, acq_, conv_, opt_, prog_ = None, None, None, None, None
    truth = truth or truth_
    acquisition = acquisition or acq_
    convergence = convergence or conv_
    options = options or opt_
    progress = progress or prog_
    return (truth, surrogate, acquisition, convergence, options, progress)
