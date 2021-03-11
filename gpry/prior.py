"""
Provides different methods for drawing initial points from the prior.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from cobaya.log import LoggedError
from cobaya.tools import get_scipy_1d_pdf


class Prior:
    """
    Class which is used to construct the prior for the algorithm. This works by
    giving independent priors in each dimension through a single "prior" dict
    which defines which distribution which parameter shall follow. The methods
    in this class then allow to get bounds for the optimization algorithm and
    draw initial points from the prior. Parts of this module are copied from
    the Cobaya-package.

    Parameters
    ----------

    params : dict
        Dictionary containing the priors for every variable. Should follow the
        format::

            params={"param1" : {"dist": "uniform"
                                "min": ..,
                                "max": ..},
                    "param2" : {"dist": "norm"
                                "min": ..,
                                "max": ..},
                    ...
                   }

        The available distributions are:

        # ``uniform`` with ``min`` and ``max`` defining the interval
        # ``norm`` (normal distribution) with ``loc`` being the mean and
          ``scale`` the standard deviation
        # ``halfnorm`` (half-normal distribution). ``loc`` and ``scale`` have
          the same meaning as for the normal distribution.

        Additionally to this the dictionary can contain other values (like
        LaTex symbols etc.) These will be ignored by this class.


    **Currently I'm just using the prior module of Cobaya so just ignore this
    for now**

    """

    def __init__(self, prior):
        self.prior_dict = prior

        self.param_pdfs = {}
        for param in list(prior):
            self.param_pdfs[param] = get_scipy_1d_pdf({param: prior[param]})

    def draw_samples(self, n_samples=1, order=None):
        """
        Draw n samples from the prior.
        """
        samples = np.empty((len(list(self.param_pdfs)), n_samples))
        if order is None:
            order = list(self.param_pdfs)
        elif not np.iterable(order):
            raise ValueError("order needs to be a list or None, got %s"
                             % order)
        print(order)
        for i, param in enumerate(order):
            samples[i, :] = self.param_pdfs[param].rvs(size=n_samples)
        return samples

    def get_bounds(self, order=None):
        """
        Get the bounds for the prior. For unbounded priors in returns the
        interval which contains the n-sigma inverval.
        """

def main():
    info = {"x_1" : {"dist": "uniform", "min": 3, "max":3.1},
            "x_2" : {"dist": "norm", "loc": 2, "scale":1}}
    prior = Prior(info)
    print(prior.draw_samples(5))

if __name__ == "__main__":
    main()
