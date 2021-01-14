.. GPry documentation

======================
The GPry documentation
======================

Welcome to the documentation of the `GPry`-Package. Currently this project is still
in progress so please beware, that the contents might change frequently.

The Goal of this project is to provide a framework for performing fast Bayesian inference on expensive posterior
functions using Gaussian Processes.
Additionally this package is designed to communicate with the `Cobaya-Package <https://pypi.org/project/cobaya/>`_
to be able to use it easily in cosmological applications.
This package is a result of my master thesis and implements several novel ideas. Most importantly this algorithm
uses nested sampling to take advantage of speed hierarchies between different dimensions in the parameter space.

Contents:
=========
.. toctree::
  :maxdepth: 2
  
  installation
  intro_example
  adv_example
  modules	

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

