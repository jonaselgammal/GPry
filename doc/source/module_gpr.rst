``gpr``
-------

This module contains GPry's standard Gaussian-process regressor. In the JAX
development baseline it remains the canonical GP backend, with optional JAX
acceleration enabled when JAX is importable. The conservative defaults are a
fixed noise level of ``0.01`` and transformed-space length-scale bounds
``[0.01, 100]``.

.. automodule:: gpry.gpr
   :members:
   :undoc-members:
   :exclude-members: set_score_request, set_predict_request, set_fit_request
   :show-inheritance:
