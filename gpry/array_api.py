"""Lean numpy↔JAX boundary helpers.

This module exists to keep one source of truth for the few places GPry needs
to cross the numpy/JAX boundary: backend classes declare their preferred I/O
via :class:`ArrayContract`, callers materialise jax arrays via :func:`to_jax`,
and the surrogate / classifier plan a boundary conversion via
:func:`make_array_converter`.

Public surface
==============

.. autosummary::

    ArrayContract
    to_jax
    make_array_converter

The module intentionally does **not** grow into a generic array shim — backends
own their own boundary conversions. See ``AGENTS/jax_split_refactor_plan.md``
Phase 6 for the decision history.
"""

from dataclasses import dataclass
from typing import Callable, FrozenSet, Literal
import warnings

import numpy as np

ArrayKind = Literal["numpy", "jax"]

try:  # pragma: no cover - exercised in JAX envs
    import jax  # noqa: F401
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in non-JAX envs
    jnp = None
    _JAX_AVAILABLE = False


def to_jax(x):
    """Materialise ``x`` as a JAX float64 array. Requires JAX to be installed."""
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX arrays requested, but JAX is not available.")
    return jnp.asarray(x, dtype=jnp.float64)


@dataclass(frozen=True)
class ArrayContract:
    """Declared I/O preferences of a backend class.

    Attributes
    ----------
    accepted_inputs
        The array kinds the backend's public methods accept without
        conversion (typically only the preferred kind, but some backends
        accept both numpy and jax).
    preferred_input
        The array kind the backend would like to receive — callers crossing a
        boundary should convert to this kind.
    output_kind
        The array kind the backend returns from its public methods.
    """

    accepted_inputs: FrozenSet[ArrayKind]
    preferred_input: ArrayKind
    output_kind: ArrayKind


def make_array_converter(
    source_kind: ArrayKind,
    target_kind: ArrayKind,
    *,
    warn: bool = False,
    context: str | None = None,
) -> Callable:
    """Return a callable that converts a single array from ``source_kind`` to
    ``target_kind``.

    Identity when the kinds match. Optionally emits a one-time ``UserWarning``
    (with ``context``) so a hot path that ends up inserting conversions can be
    diagnosed without changing the algorithm.
    """
    if source_kind == target_kind:
        return lambda x: x

    if warn:
        warnings.warn(
            f"Array conversion {source_kind} -> {target_kind} inserted"
            + (f" for {context}." if context else ".")
        )

    if target_kind == "numpy":
        return lambda x: np.asarray(x)
    if target_kind == "jax":
        return lambda x: to_jax(x)
    raise ValueError(f"Unknown target array kind: {target_kind}")
