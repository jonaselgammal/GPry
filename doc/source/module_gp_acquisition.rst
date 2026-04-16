``gp_acquisition``
------------------

This module contains the acquisition engines. NORA is the recommended engine
for robust batch acquisition. The JAX development path keeps the NORA internals
in transformed parameter space and supports BlackJAX as a portable nested
sampling backend, while retaining the non-JAX sampler interfaces.

Some acquisition options, including explicit high-uncertainty exploration
points, clustered candidate selection, and optimistic surrogate sampling, are
experimental and disabled by default.

.. automodule:: gpry.gp_acquisition
   :members:
   :undoc-members:
   :show-inheritance:
