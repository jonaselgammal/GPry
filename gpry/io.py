import os
from cobaya.model import get_model, Model


def _get_dill():
    try:
        import dill
    except ImportError as excpt:
        raise ImportError("Could not find the 'dill' package. This is not a strict "
                          "requirement for gpry, but without it the checkpoint "
                          "functionality does not work.") from excpt
    return dill


_checkpoint_filenames = {
    "model": "mod.pkl", "gpr": "gpr.pkl", "acquisition": "acq.pkl",
    "convergence": "con.pkl", "options": "opt.pkl", "progress": "pro.pkl"}

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
    [model, gp, acquisition, convergence, options]
    """
    if path is None:
        return [False] * len(_checkpoint_filenames)
    return [os.path.exists(os.path.join(path, f)) for f in _checkpoint_filenames.values()]


def read_checkpoint(path, model=None):
    """
    Loads checkpoint files to be able to resume a run or save the results for
    further processing.

    Parameters
    ----------
    path : string
        The path where the files are located.

    model : cobaya.model.Model, optional
        If passed, it will be used instead of the loaded one.

    Returns
    -------
    (model, gpr, acquisition, convergence, options, progress)
    If any of the files does not exist or cannot be read the function will
    return None instead.
    """
    pickle = _get_dill()
    # Check if a file exists in the checkpoint and if so resume from there.
    checkpoint_files = check_checkpoint(path)
    # Read in checkpoint
    if model is not None and not isinstance(model, Model):
        raise ValueError(
            "If 'model' is not None, it must be a cobaya.model.Model instance."
        )
    if model is None:
        with open(os.path.join(path, _checkpoint_filenames["model"]), 'rb') as i:
            model = pickle.load(i) if checkpoint_files[0] else None
            # Convert model from dict to model object
            model = get_model(model)
    with open(os.path.join(path, _checkpoint_filenames["gpr"]), 'rb') as i:
        gpr = pickle.load(i) if checkpoint_files[1] else None
    with open(os.path.join(path, _checkpoint_filenames["acquisition"]), 'rb') as i:
        acquisition = pickle.load(i) if checkpoint_files[2] else None
    with open(os.path.join(path, _checkpoint_filenames["convergence"]), 'rb') as i:
        if checkpoint_files[3]:
            convergence = pickle.load(i)
            convergence.prior = model.prior
        else:
            convergence = None
    with open(os.path.join(path, _checkpoint_filenames["options"]), 'rb') as i:
        options = pickle.load(i) if checkpoint_files[5] else None
    with open(os.path.join(path, _checkpoint_filenames["progress"]), 'rb') as i:
        progress = pickle.load(i) if checkpoint_files[4] else None
    return model, gpr, acquisition, convergence, options, progress


def save_checkpoint(path, model, gpr, acquisition, convergence, options, progress):
    """
    This function is used to save all relevant parts of the GP loop for reuse
    as checkpoint in case the procedure crashes.
    This function creates ``.pkl`` files which contain the instances
    of the different modules.
    The files can be loaded with the _read_checkpoint function.

    Parameters
    ----------
    path : The path where the files shall be saved
        The files will be saved as *path* +(mod, gpr, acq, con, opt).pkl

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_

    gpr : GaussianProcessRegressor

    acquisition : GPAcquisition

    convergence : Convergence_criterion

    options : dict

    progress : Progress instance
    """
    if path is None:
        return
    pickle = _get_dill()
    create_path(path, verbose=False)
    try:
        with open(os.path.join(path, _checkpoint_filenames["model"]), 'wb') as f:
            # Save model as dict
            model_dict = model.info()
            pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["gpr"]), 'wb') as f:
            pickle.dump(gpr, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["acquisition"]), 'wb') as f:
            pickle.dump(acquisition, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["convergence"]), 'wb') as f:
            # TODO: maybe convergence should just not keep the prior!
            # Need to delete the prior object in convergence so it doesn't
            # do weird stuff while pickling
            from copy import deepcopy
            convergence = deepcopy(convergence)
            convergence.prior = None
            pickle.dump(convergence, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["options"]), 'wb') as f:
            pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, _checkpoint_filenames["progress"]), 'wb') as f:
            pickle.dump(progress, f, pickle.HIGHEST_PROTOCOL)
    except Exception as excpt:
        raise RuntimeError("Could not save the checkpoint. Check if the path "
                           "is correct and exists. Error message: " + str(excpt))
