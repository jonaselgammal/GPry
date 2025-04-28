Gaussian Process Acquisition
============================

This module implements tools for active sampling with Gaussian Process surrogate models.


Acquisition
===========

The acquisition module contains both the acquisition function as well as the
optimization procedure for it. It operates similarly to the GP regressor module.

Acquisition function
--------------------

The acquisition function is the centerpiece of the Bayesion optimization
procedure and decides which point the algorithm samples next. The
:mod:`acquisition_functions` module has multiple inbuilt acquisition functions
as well as building blocks for custom acquistion functions which can be
constructed using the + and * operators. Since it tends to perform best we will
use the standard :class:`acquisition_functions.Log_exp` acquisition function
with a :math:`\zeta` value of 0.05 to encourage exploration (as we know that
the shape of the posterior distribution is not very gaussian)::

    from gpry.acquisition_functions import Log_exp
    af = Log_exp(zeta=0.1)

Then it is time for the actual GP Acquisition. For this we need to
build our instance of the :class:`gp_acquisition.GPAcquisition` class which
also takes the acquisition function. Furthermore it needs the prior bounds
so it knows which volume to sample in. Furthermore like with the GP regressor
it is usually a good idea to scale the prior bounds to a unit hypercube
(assuming that the mode occupies roughly the same portion of the prior in each
dimension) as the optimizer tends to struggle with very different scales across
different dimensions::

    from gpry.gp_acquisition import GPAcquisition
    acq = GPAcquisition(
        prior_bounds,
        acq_func=af,
        n_restarts_optimizer=10,
        preprocessing_X=Normalize_bounds(prior_bounds)
        )


Acquisition Functions
=====================

This module contains the implementation of an "Aquisition Function" class with a structure similar to the one
provided by the Kernels module of sklearn. Additionally to some internally provided base AF's
this module also overwrites arithmetic operators for AF's in order to enable the construction of composite
AF's.

.. note::

    Bear in mind, that you first need to initialize an instance of an acquisition function before calling it.
    Composite acquisition functions are possible too, e.g.::

        from acquisition_functions import ConstantAcqFunc, Mu, Sigma
        af = ConstantAcqFunc(2) * Mu + (-3) * Sigma ** 2.5

.. warning::
    Currently only the ``+``, ``*`` and ``**`` operator are supported but I am sure you can figure out how to
    work around using ``-`` and ``/`` on your own ;)


.. _acq_nora:

The NORA acquisition engine
===========================

[TODO]



.. automodule:: acquisition_functions
    :members:
    :special-members: __call__
    :noindex:





.. automodule:: gp_acquisition
   :members:
   :show-inheritance:
