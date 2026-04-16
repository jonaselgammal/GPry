``ns_interfaces``
-----------------

This module contains nested-sampler adapters used by NORA and by final surrogate
sampling. Supported interfaces include PolyChord, UltraNest, nessai, and
BlackJAX when their optional dependencies are installed.

BlackJAX support is optional. It is the preferred portable backend for the
JAX-accelerated NORA path, but GPry should still import and run without it.

.. automodule:: gpry.ns_interfaces
   :members:
   :undoc-members:
   :show-inheritance:
