"""Lean array-kind helpers used to preplan representation conversions."""

from dataclasses import dataclass
from typing import Callable, FrozenSet, Literal
import warnings

import numpy as np

ArrayKind = Literal["numpy", "jax"]

try:
    import jax
    import jax.numpy as jnp

    _JAX_ARRAY_TYPES = tuple(
        t for t in (
            getattr(jax, "Array", None),
            getattr(jnp, "ndarray", None),
        ) if t is not None
    )
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in non-JAX envs
    jax = None
    jnp = None
    _JAX_ARRAY_TYPES = ()
    _JAX_AVAILABLE = False


def detect_array_kind(x) -> ArrayKind:
    if _JAX_AVAILABLE and isinstance(x, _JAX_ARRAY_TYPES):
        return "jax"
    return "numpy"


def to_numpy(x):
    return np.asarray(x)


def to_jax(x):
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX arrays requested, but JAX is not available.")
    return jnp.asarray(x, dtype=jnp.float64)


def convert_array(x, target_kind: ArrayKind):
    if target_kind == "numpy":
        return to_numpy(x)
    if target_kind == "jax":
        return to_jax(x)
    raise ValueError(f"Unknown target array kind: {target_kind}")


@dataclass(frozen=True)
class ArrayContract:
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
    if source_kind == target_kind:
        return lambda x: x

    if warn:
        message = (
            f"Array conversion {source_kind} -> {target_kind} inserted"
            + (f" for {context}." if context else ".")
        )
        warnings.warn(message)

    return lambda x: convert_array(x, target_kind)
