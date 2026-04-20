``surrogate``
-------------

This module contains the surrogate model that combines preprocessing,
classification of invalid/low-probability regions, and the GP regressor.
Only points that pass the finite/informative-region logic are used to train the
GPR. The current implementation tracks the actual indices added to the GPR on
each append, so classifier ordering cannot leave the GP stale.

.. automodule:: gpry.surrogate
   :members:
   :undoc-members:
   :show-inheritance:
