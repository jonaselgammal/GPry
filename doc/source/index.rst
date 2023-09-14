======================
The GPry documentation
======================

`GPry` was developed as a result of my master thesis and and can be seen as an alternative
to established samplers like MCMC and Nested Sampling. It is targeted at a specific class
of posterior distributions (with the initial goal of speeding up inference in cosmology)
with the aim of creating an algorithm which can efficiently obtain marginal quantities
from (computationally) expensive likelihoods.

Additionally this package is designed to communicate with the
`Cobaya-Package <https://pypi.org/project/cobaya/>`_ making for easy use in cosmological
applications.

`Deployed on PyPI <https://pypi.org/project/gpry/>`_

`Source code on GitHub <https://github.com/jonaselgammal/GPry>`_

Contents:
=========
.. toctree::
  :maxdepth: 2

  installation
  intro_example
  adv_example
  running
  running_cobaya
  callback
  modules

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
