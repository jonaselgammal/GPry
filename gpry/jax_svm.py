r"""JAX-native binary RBF SVM closely matching sklearn's dense SVC mathematics.

This module intentionally implements only the subset needed by GPry's infinities
classifier path:

* dense binary C-SVC
* RBF kernel
* ``gamma={"scale", "auto"}`` or a positive float
* optional sample weights and per-class ``class_weight``

The optimization problem matches sklearn/libsvm's soft-margin C-SVC dual:

.. math::

   \max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{ij}\alpha_i\alpha_j y_i y_j
   K(x_i, x_j)

subject to :math:`0 \le \alpha_i \le C_i` and :math:`\sum_i \alpha_i y_i = 0`,
where :math:`C_i` includes the global ``C``, optional class weights and optional
sample weights.

The solver is a simple SMO implementation. It is not meant as a drop-in libsvm
replacement, but the kernel, constraints, decision function and public attribute
conventions are aligned with sklearn's binary ``SVC``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np

from gpry.array_api import ArrayContract, to_jax


@jax.jit
def _rbf_kernel_matrix_jax(X: jnp.ndarray, Y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    x_sq = jnp.sum(jnp.square(X), axis=1, keepdims=True)
    y_sq = jnp.sum(jnp.square(Y), axis=1, keepdims=True).T
    sqdist = jnp.maximum(x_sq + y_sq - 2.0 * X @ Y.T, 0.0)
    return jnp.exp(-gamma * sqdist)


@jax.jit
def _decision_function_jax(
    X: jnp.ndarray,
    support_vectors: jnp.ndarray,
    signed_dual_coef: jnp.ndarray,
    intercept: float,
    gamma: float,
) -> jnp.ndarray:
    kernel = _rbf_kernel_matrix_jax(X, support_vectors, gamma)
    return kernel @ signed_dual_coef + intercept


def _validate_dense_numeric_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {X.shape}.")
    return X


def _validate_binary_y(y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y)
    if y.ndim != 1:
        y = np.ravel(y)
    classes, y_encoded = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError(
            "JaxBinaryRBFSVC currently supports binary classification only; "
            f"got {len(classes)} classes."
        )
    y_signed = np.where(y_encoded == 0, -1.0, 1.0).astype(np.float64, copy=False)
    return y, classes, y_signed


def _compute_gamma(gamma, X: np.ndarray) -> float:
    if isinstance(gamma, str):
        if gamma == "scale":
            X_var = X.var()
            return 1.0 / (X.shape[1] * X_var) if X_var != 0.0 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError(f"Unsupported gamma specifier {gamma!r}.")
    if not isinstance(gamma, Real) or gamma <= 0:
        raise ValueError(
            f"gamma must be 'scale', 'auto' or a positive float; got {gamma!r}."
        )
    return float(gamma)


def _compute_class_weight_vector(class_weight, y_raw: np.ndarray, classes: np.ndarray):
    if class_weight is None:
        weights = np.ones(len(classes), dtype=np.float64)
    elif class_weight == "balanced":
        counts = np.array([(y_raw == cls).sum() for cls in classes], dtype=np.float64)
        weights = len(y_raw) / (len(classes) * counts)
    elif isinstance(class_weight, dict):
        weights = np.array(
            [float(class_weight.get(cls, 1.0)) for cls in classes], dtype=np.float64
        )
    else:
        raise ValueError(
            "class_weight must be None, 'balanced', or a dict mapping class labels "
            f"to weights; got {class_weight!r}."
        )
    return weights


@dataclass
class _SmoState:
    alpha: np.ndarray
    decision: np.ndarray
    bias: float
    changed: int


class JaxBinaryRBFSVC:
    """Binary dense RBF-kernel C-SVC with sklearn-like public attributes."""

    def __init__(
        self,
        C: float = 1.0,
        *,
        gamma: str | float = "scale",
        tol: float = 1e-3,
        max_iter: int = 1000,
        max_passes: int = 10,
        class_weight=None,
        support_tol: float = 1e-8,
    ):
        if C <= 0:
            raise ValueError(f"C must be strictly positive, got {C}.")
        if tol <= 0:
            raise ValueError(f"tol must be strictly positive, got {tol}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if max_passes <= 0:
            raise ValueError(f"max_passes must be positive, got {max_passes}.")
        self.C = float(C)
        self.gamma = gamma
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.max_passes = int(max_passes)
        self.class_weight = class_weight
        self.support_tol = float(support_tol)

    @property
    def array_contract(self):
        return ArrayContract(
            accepted_inputs=frozenset({"numpy", "jax"}),
            preferred_input="jax",
            output_kind="jax",
        )

    @property
    def supports_native_prediction(self):
        return True

    def fit(self, X, y, sample_weight=None):
        X = _validate_dense_numeric_X(X)
        y_raw, classes, y_signed = _validate_binary_y(y)
        if len(X) != len(y_signed):
            raise ValueError(
                f"X and y have incompatible shapes: {X.shape} and {y_raw.shape}."
            )
        sample_weight = (
            np.ones(len(X), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        if sample_weight.shape != (len(X),):
            raise ValueError(
                "sample_weight must have shape (n_samples,), got "
                f"{sample_weight.shape}."
            )

        self.classes_ = classes
        self.class_weight_ = _compute_class_weight_vector(
            self.class_weight, y_raw, classes
        )
        class_weight_per_sample = np.where(y_signed < 0, self.class_weight_[0], self.class_weight_[1])
        C_i = self.C * sample_weight * class_weight_per_sample
        self._gamma = _compute_gamma(self.gamma, X)
        K = np.asarray(
            _rbf_kernel_matrix_jax(jnp.asarray(X), jnp.asarray(X), self._gamma)
        )

        state = self._solve_binary_dual(K, y_signed, C_i)
        alpha = state.alpha
        support = alpha > self.support_tol
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[self.support_]
        self._support_labels = y_signed[self.support_]
        self._support_alpha = alpha[self.support_]
        self._signed_dual_coef = self._support_alpha * self._support_labels
        self.dual_coef_ = self._signed_dual_coef[None, :]
        self.intercept_ = np.array([state.bias], dtype=np.float64)
        self.fit_status_ = 0 if state.changed >= 0 else 1
        self.n_iter_ = state.changed
        self.n_features_in_ = X.shape[1]
        self.shape_fit_ = X.shape
        self._X = X
        self._y_signed = y_signed
        self._C_i = C_i
        self.n_support_ = np.array(
            [
                int(np.sum(self._support_labels < 0)),
                int(np.sum(self._support_labels > 0)),
            ],
            dtype=np.int32,
        )
        return self

    def decision_function_jax(self, X) -> jnp.ndarray:
        self._check_is_fitted()
        X = to_jax(X)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {X.shape}.")
        return _decision_function_jax(
            X,
            jnp.asarray(self.support_vectors_),
            jnp.asarray(self._signed_dual_coef),
            float(self.intercept_[0]),
            float(self._gamma),
        )

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(self.decision_function_jax(X))

    def decision_function_native(self, X) -> jnp.ndarray:
        return self.decision_function_jax(X)

    def predict(self, X):
        decision = self.decision_function(X)
        return self.classes_.take((decision >= 0.0).astype(np.intp))

    def predict_native(self, X) -> jnp.ndarray:
        decision = self.decision_function_jax(X)
        return decision >= 0.0

    def _check_is_fitted(self):
        if not hasattr(self, "support_vectors_"):
            raise ValueError("This JaxBinaryRBFSVC instance is not fitted yet.")

    def _solve_binary_dual(
        self, K: np.ndarray, y: np.ndarray, C_i: np.ndarray
    ) -> _SmoState:
        n_samples = len(y)
        alpha = np.zeros(n_samples, dtype=np.float64)
        decision = np.zeros(n_samples, dtype=np.float64)
        bias = 0.0
        examine_all = True
        outer_iters = 0
        stale_passes = 0

        while outer_iters < self.max_iter and (examine_all or stale_passes < self.max_passes):
            num_changed = 0
            if examine_all:
                indices = np.arange(n_samples)
            else:
                indices = np.flatnonzero(
                    (alpha > self.support_tol) & (alpha < (C_i - self.support_tol))
                )

            for i in indices:
                changed, alpha, decision, bias = self._examine_example(
                    i, K, y, C_i, alpha, decision, bias
                )
                num_changed += int(changed)

            outer_iters += 1
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
                stale_passes += 1
            else:
                stale_passes = 0

        bias = self._recompute_bias(K, y, alpha, C_i, bias)
        return _SmoState(alpha=alpha, decision=decision, bias=bias, changed=outer_iters)

    def _examine_example(
        self,
        i2: int,
        K: np.ndarray,
        y: np.ndarray,
        C_i: np.ndarray,
        alpha: np.ndarray,
        decision: np.ndarray,
        bias: float,
    ) -> tuple[bool, np.ndarray, np.ndarray, float]:
        E2 = decision[i2] - y[i2]
        r2 = E2 * y[i2]
        at_lower = alpha[i2] <= self.support_tol
        at_upper = alpha[i2] >= (C_i[i2] - self.support_tol)
        violates = (r2 < -self.tol and not at_upper) or (r2 > self.tol and not at_lower)
        if not violates:
            return False, alpha, decision, bias

        non_bound = np.flatnonzero(
            (alpha > self.support_tol) & (alpha < (C_i - self.support_tol))
        )
        if len(non_bound) > 1:
            errors = decision - y
            if E2 > 0:
                i1 = int(non_bound[np.argmin(errors[non_bound])])
            else:
                i1 = int(non_bound[np.argmax(errors[non_bound])])
            if i1 != i2:
                updated = self._take_step(i1, i2, K, y, C_i, alpha, decision, bias)
                if updated is not None:
                    return True, *updated

        for i1 in non_bound:
            if i1 == i2:
                continue
            updated = self._take_step(int(i1), i2, K, y, C_i, alpha, decision, bias)
            if updated is not None:
                return True, *updated

        for i1 in range(len(alpha)):
            if i1 == i2:
                continue
            updated = self._take_step(i1, i2, K, y, C_i, alpha, decision, bias)
            if updated is not None:
                return True, *updated

        return False, alpha, decision, bias

    def _take_step(
        self,
        i1: int,
        i2: int,
        K: np.ndarray,
        y: np.ndarray,
        C_i: np.ndarray,
        alpha: np.ndarray,
        decision: np.ndarray,
        bias: float,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        if i1 == i2:
            return None

        alpha1_old = alpha[i1]
        alpha2_old = alpha[i2]
        y1 = y[i1]
        y2 = y[i2]
        E1 = decision[i1] - y1
        E2 = decision[i2] - y2

        if y1 != y2:
            L = max(0.0, alpha2_old - alpha1_old)
            H = min(C_i[i2], C_i[i1] + alpha2_old - alpha1_old)
        else:
            L = max(0.0, alpha1_old + alpha2_old - C_i[i1])
            H = min(C_i[i2], alpha1_old + alpha2_old)
        if L >= H:
            return None

        eta = 2.0 * K[i1, i2] - K[i1, i1] - K[i2, i2]
        if eta >= 0.0:
            return None

        alpha2_new = alpha2_old - y2 * (E1 - E2) / eta
        alpha2_new = float(np.clip(alpha2_new, L, H))
        if abs(alpha2_new - alpha2_old) < self.support_tol:
            return None

        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        b1 = (
            bias
            - E1
            - y1 * (alpha1_new - alpha1_old) * K[i1, i1]
            - y2 * (alpha2_new - alpha2_old) * K[i1, i2]
        )
        b2 = (
            bias
            - E2
            - y1 * (alpha1_new - alpha1_old) * K[i1, i2]
            - y2 * (alpha2_new - alpha2_old) * K[i2, i2]
        )

        if 0.0 < alpha1_new < C_i[i1]:
            bias_new = b1
        elif 0.0 < alpha2_new < C_i[i2]:
            bias_new = b2
        else:
            bias_new = 0.5 * (b1 + b2)

        alpha_new = alpha.copy()
        alpha_new[i1] = alpha1_new
        alpha_new[i2] = alpha2_new
        delta1 = alpha1_new - alpha1_old
        delta2 = alpha2_new - alpha2_old
        decision_new = (
            decision
            + y1 * delta1 * K[:, i1]
            + y2 * delta2 * K[:, i2]
            + (bias_new - bias)
        )
        return alpha_new, decision_new, float(bias_new)

    def _recompute_bias(
        self,
        K: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        C_i: np.ndarray,
        fallback_bias: float,
    ) -> float:
        signed_alpha = alpha * y
        raw_decision = K @ signed_alpha
        margin = (alpha > self.support_tol) & (alpha < (C_i - self.support_tol))
        if np.any(margin):
            return float(np.mean(y[margin] - raw_decision[margin]))
        return float(fallback_bias)
