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

.. automodule:: acquisition_functions
    :members:
    :special-members: __call__
    :noindex:
