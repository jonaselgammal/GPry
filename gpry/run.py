"""
Top level run file which constructs the loop for mapping a posterior
distribution.
"""

from gpry.gpr import GaussianProcessRegressor
from gpry.gp_acquisition import GP_Acquisition
from gpry.preprocessing import Normalize_bounds, Normalize_y
from gpry.kernels import ConstantKernel as C, RBF, Matern
from gpry.convergence import Convergence_criterion, KL_from_MC_training, \
    KL_from_draw_approx
from cobaya.model import get_model
from copy import deepcopy
import numpy as np
import warnings
import pickle
import os


def run(model, gp="RBF", gp_acquisition="Log_exp",
        convergence_criterion="KL", options={},
        callback=None, verbose=1):
    """
    This function takes care of constructing the Bayesian quadrature/likelihood
    characterization loop. This is the easiest way to make use of the
    gpry algorithm. The minimum requirements for running this are a Cobaya
    prior object and a likelihood. Furthermore the details of the GP and
    and acquisition can be specified by the user.

    Parameters
    ----------

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. Cobaya is only used here
        as a wrapper to get the logposterior etc.

    gp : GaussianProcessRegressor, "RBF" or "Matern", optional (default="RBF")
        The GP used for interpolating the posterior. If None or "RBF" is given
        a GP with a constant kernel multiplied with an anisotropic RBF kernel
        and dynamic bounds is generated. The same kernel with a Matern 3/2
        kernel instead of a RBF is generated if "Matern" is passed. This might
        be useful if the posterior is not very smooth.
        Otherwise a custom GP regressor can be created and passed.

    gp_acquisition : GP_Acquisition, optional (default="Log_exp")
        The acquisition object. If None is given the Log_exp acquisition
        function is used and the GP's X-values are preprocessed to be in the
        uniform hypercube before optimizing the acquistion function.

    convergence_criterion : Convergence_criterion, optional (default="KL")
        The convergence criterion. If None is given the KL-divergence between
        consecutive runs is calculated with an MCMC run and the run converges
        if KL<0.02 for two consecutive steps.

    options : dict, optional (default=None)
        A dict containing all options regarding the bayesian optimization loop.
        The available options are:

            * n_initial : Number of initial samples before starting the BO loop
              (default: 3*number of dimensions)
            * n_points_per_acq : Number of points which are aquired with
              Kriging believer for every acquisition step (default: 1)
            * max_points : Maximum number of points before the run fails
              (default: 1000)
            * max_init : Maximum number of points drawn at initialization
              before the run fails (default: 10*number of dimensions). If the
              run fails repeatadly at initialization try decreasing the volume
              of your prior.

    callback : string, optional (default=None)
        Path for storing callback information from which to resume in case the
        algorithm crashes. If None is given no callback is saved.

    Returns
    -------

    gp : The GP regressor This can be used to call an MCMC sampler for getting
        marginalized properties.
    """

    # Construct GP if it's not already constructed
    if isinstance(gp, str):
        if gp == "RBF":
            # Construct RBF kernel
            n_d = model.prior.d()
            prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
            kernel = C(1.0, [0.001, 10000]) \
                * RBF([0.01]*n_d, "dynamic", prior_bounds=prior_bounds)
            # Construct GP
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                preprocessing_X=Normalize_bounds(prior_bounds),
                preprocessing_y=Normalize_y(),
                verbose=verbose
                )
        elif gp == "Matern":
            # Construct RBF kernel
            n_d = model.prior.d()
            prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
            kernel = C(1.0, [0.001, 10000]) \
                * Matern([0.01]*n_d, "dynamic", prior_bounds=prior_bounds)
            # Construct GP
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                preprocessing_X=Normalize_bounds(),
                preprocessing_y=Normalize_y(),
                verbose=verbose
                )
        else:
            raise ValueError("Currently only 'RBF' and 'Matern' are supported "
                             "as standard GPs. Got %s" % gp)

    elif isinstance(gp, GaussianProcessRegressor):
        gpr = gp
    else:
        raise TypeError("gp should be a GP regressor, 'RBF' or 'Matern', got "
                        "%s" % gp)

    # Construct the acquisition object if it's not already constructed
    if isinstance(gp_acquisition, str):
        prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
        if gp_acquisition == "Log_exp":
            acquisition = GP_Acquisition(prior_bounds,
                                         acq_func="Log_exp",
                                         acq_optimizer="fmin_l_bfgs_b",
                                         n_restarts_optimizer=5,
                                         preprocessing_X=Normalize_bounds(
                                            prior_bounds),
                                         verbose=verbose)
    elif isinstance(gp_acquisition, GP_Acquisition):
        acquisition = acquisition
    else:
        raise TypeError("gp_acquisition should be an Acquisition object or "
                        "'Log_exp', got %s" % gp_acquisition)

    # Construct the convergence criterion
    if isinstance(convergence_criterion, str):
        if convergence_criterion == "KL":
            params = {"limit": 0.02}
            convergence = KL_from_MC_training(model.prior,
                                              params)
    elif isinstance(convergence_criterion, Convergence_criterion):
        convergence = convergence_criterion
    else:
        raise TypeError("convergence_criterion should be a "
                        "Convergence_criterion object or KL, got %s"
                        % convergence_criterion)

    # Check if a callback exists already and if so resume from there
    callback_files = check_callback(callback)
    if np.any(callback_files):
        if np.all(callback_files):
            print("#########################################")
            print("Callback found. Resuming from there...")
            print("If this behaviour is unintentional either")
            print("turn the callback option off or rename it")
            print("to a file which doesn't exist.")
            print("#########################################")

            model, gpr, acquisition, convergence, options = read_callback(
                callback)
        else:
            warnings.warn("Callback files were found but are incomplete. "
                          "Ignoring those files...")

    print("Model has been initialized")

    # Read in options for the run
    if options is None:
        options = {}
    n_d = model.prior.d()
    n_initial = options.get("n_initial", 3*n_d)
    n_points_per_acq = options.get("n_points_per_acq", 1)
    max_points = options.get("max_points", 1000)
    max_init = options.get("max_init", 10*n_d)

    # Sanity checks
    if n_initial >= max_points:
        raise ValueError("The number of initial samples needs to be "
                         "smaller than the maximum number of points")
    if n_initial <= 0:
        raise ValueError("The number of initial samples needs to be bigger "
                         "than 0")

    # Check if the GP already contains points. If so they are reused.
    pretrained = 0
    if hasattr(gpr, "y_train"):
        if len(gpr.y_train) > 0:
            pretrained = len(gpr.y_train)

    # Draw initial samples if the model hasn't been trained before
    if pretrained < n_initial:
        # Initial samples loop. The initial samples are drawn from the prior
        # and according to the distribution of the prior.
        X_init = np.empty((0, n_d))
        y_init = np.empty(0)
        n_finite = pretrained
        for iter in range(max_init-pretrained):
            # Draw point from prior and evaluate logposterior at that point
            X = model.prior.sample(n=1)
            y = model.logpost(X[0])
            # Only if the value is finite it contributes to the number of
            # initial samples
            if np.isfinite(y):
                n_finite += 1
            X_init = np.append(X_init, X, axis=0)
            y_init = np.append(y_init, y)
            # Break loop if the desired number of initial samples is reached
            if n_finite >= n_initial:
                print("Initial samples drawn, starting with Bayesian "
                      "optimization loop.")
                break
        # Raise error if the number of initial samples hasn't been reached
        if n_finite < n_initial:
            if n_finite >= n_d:
                raise RuntimeError("The desired number of finite initial "
                                   "samples hasn't been reached. Try "
                                   "increasing max_init or decreasing the "
                                   "volume of the prior")

        # Append the initial samples to the gpr
        gpr.append_to_data(X_init, y_init)
        # Save callback
        save_callback(callback, model, gpr, acquisition, convergence, options)
    else:
        n_finite = len(gpr.y_train)

    # Run bayesian optimization loop
    n_iterations = int((max_points-n_finite) / n_points_per_acq)
    for iter in range(n_iterations):
        print(f"+++ Iteration {iter} (of {n_iterations}) +++++++++")
        # Save old gp for convergence criterion
        old_gpr = deepcopy(gpr)
        # Get new point(s) from Bayesian optimization
        new_X, y_lies, acq_vals = acquisition.multi_optimization(
            gpr, n_points=n_points_per_acq)
        # Get logposterior value(s) for the acquired points and append to the
        # current model
        new_y = np.empty(0)
        for x in new_X:
            new_y = np.append(new_y, model.logpost(x))
        gpr.append_to_data(new_X, new_y, fit=True)
        # Calculate convergence and break if the run has converged
        if convergence.is_converged(gpr, old_gpr):
            print("The run has converged, stopping the program...")
            break

        print(f"Value of convergence criterion: {convergence.values[-1]}")

        # Save
        save_callback(callback, model, gpr, acquisition, convergence, options)

    # Save
    save_callback(callback, model, gpr, acquisition, convergence, options)

    if iter == n_iterations-1:
        warnings.warn("The maximum number of points was reached before "
                      "convergence. Either increase max_points or try to "
                      "choose a smaller prior.")

    # Now that the run has converged we can return the gp and all other
    # relevant quantities which can then be processed with an MCMC etc.

    return model, gpr, acquisition, convergence, options


def save_callback(path, model, gpr, acquisition, convergence, options):
    """
    This function is used to save the callback files. TODO: Proper documentation
    """
    if path is not None:
        try:
            with open(path+"mod.pkl", 'wb') as f:
                # Save model as dict
                model_dict = model.info()
                pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)
            with open(path+"gpr.pkl", 'wb') as f:
                pickle.dump(gpr, f, pickle.HIGHEST_PROTOCOL)
            with open(path+"acq.pkl", 'wb') as f:
                pickle.dump(acquisition, f, pickle.HIGHEST_PROTOCOL)
            with open(path+"con.pkl", 'wb') as f:
                # Need to delete the prior object in convergence so it doesn't
                # do weird stuff while pickling
                convergence = deepcopy(convergence)
                convergence.prior = None
                pickle.dump(convergence, f, pickle.HIGHEST_PROTOCOL)
            with open(path+"opt.pkl", 'wb') as f:
                pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)
        except:
            raise RuntimeError("Couldn't save the callback. Check if the path "
                               "is correct and exists.")


def check_callback(path):
    """
    Checks if there are callback files in a specific location and if so if they
    are complete. Returns a list of bools.
    """
    if path is not None:
        callback_files = [os.path.exists(path + "mod.pkl"),
                          os.path.exists(path + "gpr.pkl"),
                          os.path.exists(path + "acq.pkl"),
                          os.path.exists(path + "con.pkl"),
                          os.path.exists(path + "opt.pkl")]
    else:
        callback_files = [False] * 5
    return callback_files


def read_callback(path):
    """
    Loads callback files. Currently this doesn't work for the convergene class
    (I suspect that there are some issues with Cobaya/Monte Python)
    TODO: proper documentation
    """
    # Check if a file exists in the callback and if so resume from there.
    callback_files = check_callback(path)
    # Read in callback
    with open(path+"mod.pkl", 'rb') as i:
        model = pickle.load(i) if callback_files[0] else None
        # Convert model from dict to model object
        model = get_model(model)
    with open(path+"gpr.pkl", 'rb') as i:
        gpr = pickle.load(i) if callback_files[1] else None
    with open(path+"acq.pkl", 'rb') as i:
        acquisition = pickle.load(i) if callback_files[2] else None
    with open(path+"con.pkl", 'rb') as i:
        if callback_files[3]:
            convergence = pickle.load(i)
            convergence.prior = model.prior
        else:
            convergence = None

    with open(path+"opt.pkl", 'rb') as i:
        options = pickle.load(i) if callback_files[4] else None

    return model, gpr, acquisition, convergence, options
