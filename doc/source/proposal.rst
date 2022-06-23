Proposal
========

This module provides different random number generators for getting proposals from which
to start the optimizer for the acquisition function. The way these points are proposed
becomes increasingly important with higher dimensions as the volume of the parameter
space grows exponentially.

As standard GPry uses the PartialProposer which consists of a centroid proposer
with 25% of samples drawn from a uniform distribution (to make the acquisition more
robust). Every proposer has a ``get`` method which returns a random sample from the
proposer.

.. automodule:: proposal
   :members:
   :undoc-members:
   :show-inheritance:
