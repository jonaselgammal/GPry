"""
Module containing the class holding the true log-posterior and associated parameters.
"""

from warnings import warn
from typing import Sequence, Mapping

import numpy as np

from gpry import check_cobaya_installed
from gpry.tools import check_and_return_bounds, generic_params_names, is_in_bounds, wrap_likelihood


def get_truth(loglike, bounds=None, ref_bounds=None, params=None):
    """
    Instantiates and returns a Truth|TruthCobaya object.
    """
    if callable(loglike):
        return Truth(loglike, bounds=bounds, ref_bounds=ref_bounds, params=params)
    elif check_cobaya_installed():
        # pylint: disable=import-outside-toplevel
        from cobaya.log import LoggedError
        from cobaya.model import Model, get_model

        if isinstance(loglike, Mapping):
            try:
                loglike = get_model(loglike)
            except LoggedError as excpt:
                raise TypeError(
                    "'loglike' was passed as a dict, but could not be used to "
                    "initialise a Cobaya model."
                ) from excpt
        if not isinstance(loglike, Model):
            raise TypeError("'loglike' needs to be either a callable or a Cobaya model.")
        if bounds is not None or ref_bounds is not None or params is not None:
            warn("A Cobaya model was passed. Ignoring bounds and parameter names.")
        return TruthCobaya(loglike)
    else:
        raise TypeError(
            "`loglike` seems not to be a callable function. If attempting to pass"
            " a Cobaya model, install Cobaya first: python -m pip install cobaya"
        )

class Truth:
    """
    Class holding the true log-posterior and some information about it.

    `reference_bounds` must have the same length as `bounds`, with None as an entry for
    which reference bounds different from the prior bounds are not given.
    """

    def __init__(self, loglike, bounds=None, ref_bounds=None, params=None):
        if bounds is None:
            raise ValueError(
                "'bounds' need to be defined if a likelihood function is passed."
            )
        self._prior_bounds = check_and_return_bounds(bounds)
        self.log_prior_volume = np.sum(
            np.log(self.prior_bounds[:, 1] - self.prior_bounds[:, 0])
        )
        self._loglike = wrap_likelihood(loglike, self.d)
        self._ref_bounds = self.d * [None]
        self._ref_bounds_default_prior = np.copy(self._prior_bounds)
        if ref_bounds is not None:
            try:
                if len(ref_bounds) != self.d:
                    raise TypeError
                for i, v in enumerate(ref_bounds):
                    if v is None:
                        continue
                    v = np.copy(np.atleast_1d(v))
                    if v.shape != (2,):
                        raise TypeError
                    self._ref_bounds[i] = v
                    self._ref_bounds_default_prior[i] = v
            except (TypeError, ValueError, IndexError) as excpt:
                raise TypeError(
                    "`ref_bounds` must be a sequence with as many elements as "
                    "parameters, valued either None (use prior bounds) or [min, max]."
                ) from excpt
        err_msg_type_params = (
            f"`params must be either a list of {self.d} parameter names, or a "
            f"dictionary {{name: label}} with that many entries. Got {params}"
        )
        if params is None:
            self._params = generic_params_names(self.d, "x_")
            self._labels = [p + "}" for p in generic_params_names(self.d, "x_{")]
        elif isinstance(params, Sequence):
            if len(params) != self.d or any(not isinstance(p, str) for p in params):
                raise TypeError(err_msg_type_params)
            self._params = params
            self._labels = deepcopy(params)
        elif isinstance(params, Mapping):
            if (
                len(params) != self.d
                or any(not isinstance(p, str) for p in params)
                or any(not isinstance(p, str) for p in params.values())
            ):
                raise TypeError(err_msg_type_params)
            self._params = list(params)
            self._labels = list(params.values())
        else:
            raise TypeError(err_msg_type_params)

    @property
    def d(self):
        """Dimensionality of the problem."""
        return len(self._prior_bounds)

    @property
    def prior_bounds(self):
        """Prior bounds, as an array of shape = (dim, 2)."""
        return self._prior_bounds

    @property
    def params(self):
        """Returns the list of parameter names."""
        return self._params

    @property
    def labels(self):
        """
        Returns the list of labels.
        """
        return self._labels

    def logprior(self, X):
        """
        Evaluates and returns the log-prior.
        """
        if not is_in_bounds(X, self.prior_bounds, check_shape=False):
            return -np.inf
        return -1. * self.log_prior_volume

    def loglike(self, X):
        """
        Evaluates and returns the log-likelihood.
        """
        return self._loglike(X)

    def logp(self, X):
        """
        Evaluates and returns the log-posterior.
        """
        logpost = self.logprior(X)
        if logpost != -np.inf:
            logpost += self.loglike(X)
        return logpost

    def prior_sample(self, rng):
        """Draws one point from the prior."""
        return rng.uniform(*(self.prior_bounds.T))

    def ref_sample(self, rng):
        """Draws one point from the reference distribution."""
        return rng.uniform(*(self._ref_bounds_default_prior.T))

    def as_dict(self):
        """
        Returns this instance as a dictionary so that it can be re-initialised with the
        returned values as args.
        """
        return {
            "loglike": self._loglike,
            "bounds": self.prior_bounds,
            "ref_bounds": self._ref_bounds,
            "params": (
                self.params
                if self.labels is None
                else dict(zip(self.params, self.labels))
            ),
        }

class TruthCobaya(Truth):
    """
    Truth class wrapping a Cobaya model.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, model):
        self._cobaya_model = model
        self._prior_bounds = self._cobaya_model.prior.bounds(
            confidence_for_unbounded=0.99995
        )
        self.log_prior_volume = np.sum(
            np.log(self.prior_bounds[:, 1] - self.prior_bounds[:, 0])
        )
        self._params = list(self._cobaya_model.parameterization.sampled_params())
        labels = self._cobaya_model.parameterization.labels()
        self._labels = [labels[p] for p in self._params]

    def logprior(self, X):
        """
        Evaluates and returns the log-prior.
        """
        return self._cobaya_model.prior.logp(X)

    def loglike(self, X):
        """
        Evaluates and returns the log-likelihood.
        """
        return self._cobaya_model.loglike(X)

    def logp(self, X):
        """
        Evaluates and returns the log-posterior.
        """
        return self._cobaya_model.logpost(X)

    def prior_sample(self, rng):
        """Draws one point from the prior."""
        return self._cobaya_model.prior.sample(random_state=rng)[0]

    def ref_sample(self, rng):
        """Draws one point from the reference distribution."""
        return self._cobaya_model.prior.reference(
            max_tries=np.inf,
            warn_if_tries="10d",
            ignore_fixed=True,
            warn_if_no_ref=False,
            random_state=rng,
        )

    def as_dict(self):
        """
        Returns this instance as a dictionary so that it can be re-initialised with the
        returned values as args.
        """
        return {"loglike": self._cobaya_model.info()}
