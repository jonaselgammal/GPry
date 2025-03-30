Convergence
===========

Convergence policy
^^^^^^^^^^^^^^^^^^

You can define a ``policy`` for each convergence criterion:

- ``'n'``: necessary (default if not specified)
- ``'s'``: sufficient
- ``'ns'``: necessary and sufficient
- ``'m'``: monitoring (tracked, but will not affect convergence)

If there are no criteria specified as *necessary* or *sufficient*, i.e. all criteria are set to *monitor*, the run will never converge (but it will stop at evaluation budget exhaustion).

.. automodule:: convergence
   :members:
   :exclude-members: builtin_names, DummyMPIConvergeCriterion
   :private-members:
   :show-inheritance:
