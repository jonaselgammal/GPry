"""
Contract tests: the JAX re-implementations of the GP mean must agree with the
numpy surrogate.

`gpry.mc_interfaces` does not call the GP. It walks the kernel tree, extracts
``(amplitude, length_scale, family, nu)``, and re-derives the kernel and the
predictive mean in JAX, reproducing the preprocessors from probe points. That
shadow implementation can silently drift from `gpry.kernels` / `SurrogateModel`
with no test noticing -- which is how a float32 bug (the large output scale and
strongly cancelling ``alpha`` destroy the mean in single precision) survived
until it was caught by hand.

These tests pin the agreement so any drift fails loudly.
"""
import numpy as np
import pytest

from gpry.preprocessing import NormalizeBounds, NormalizeY
from gpry.surrogate import SurrogateModel

# Agreement tolerance, relative to the scale of the GP mean.
#
# Measured on these fixtures: the JAX form matches numpy to MACHINE PRECISION
# (<=1e-15 relative) whenever K is well conditioned; it degrades to ~2e-7 only at
# d=2/3, where a 60-point GP in a small box gives cond(K) ~ 1e20 -- numerically
# singular, beyond float64's reach. That is problem conditioning, not backend drift.
#
# The bug this test exists to catch is a float32 regression, which is catastrophic
# exactly in GPry's characteristic regime (output scale railed at 1e6 with a large,
# strongly cancelling alpha): measured relative error 61, i.e. 6100%. So at d=2 the
# failure mode (61) and the conditioning noise (2e-7) are eight orders apart, and
# 1e-6 separates them cleanly.
RTOL = 1e-6


def _fitted_surrogate(d=4, n=60, seed=0, kernel="RBF", noise_fixed=True):
    """A small fitted surrogate on an anisotropic Gaussian log-posterior."""
    rng = np.random.default_rng(seed)
    bounds = np.array([[-4.0, 4.0]] * d)
    stds = np.geomspace(1.0, 0.4, d)

    def logp(X):
        X = np.atleast_2d(X)
        return -0.5 * np.sum((X / stds) ** 2, axis=1)

    # Mirror the defaults Runner would supply; SurrogateModel does not fill them in.
    sur = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor={
            "kernel": kernel,
            "output_scale_prior": [1e-2, 1e3],
            "length_scale_prior": [1e-3, 1e2],
            "noise_level": 1e-2,
            "noise_fixed": noise_fixed,
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 2,
        },
        # Configured as Runner does. Also required: with `infinities_classifier=None`
        # SurrogateModel crashes on an unfitted NormalizeY (the DummyPreprocessor
        # fallback is nested inside the classifier branch) -- a separate main bug.
        infinities_classifier={"svm": {"threshold": "20s"}},
        random_state=rng,
        verbose=0,
    )
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, d))
    sur.append(X, logp(X))
    return sur, bounds


def _raw_gp_mean(sur, X):
    """
    The GP mean the JAX backends target: raw, in original units, WITHOUT the
    infinities classifier or the upper clipping (both are non-differentiable and
    are deliberately excluded from the JAX path).
    """
    Xn = sur.preprocessing_X.transform(np.atleast_2d(np.asarray(X, float)))
    mu_n = np.ravel(sur.gpr.predict(Xn))
    return np.ravel(sur.preprocessing_y.inverse_transform(mu_n))


@pytest.mark.parametrize("d", [2, 4, 8])
def test_jax_loglike_matches_raw_gp_mean(d):
    """`build_jax_gp_loglike` works in the ORIGINAL space and keeps the y-offset,
    so it must reproduce the raw GP mean pointwise."""
    jax = pytest.importorskip("jax")
    from gpry.mc_interfaces import build_jax_gp_loglike

    sur, bounds = _fitted_surrogate(d=d, seed=d)
    loglike = build_jax_gp_loglike(sur)

    rng = np.random.default_rng(123)
    Xt = rng.uniform(bounds[:, 0], bounds[:, 1], size=(40, d))
    got = np.array([float(loglike(jax.numpy.asarray(x))) for x in Xt])
    ref = _raw_gp_mean(sur, Xt)
    scale = max(np.max(np.abs(ref)), 1.0)
    assert np.allclose(got, ref, rtol=0, atol=RTOL * scale), (
        f"JAX loglike drifted from the numpy GP mean at d={d}: "
        f"max |diff| = {np.max(np.abs(got - ref)):.3e} (scale {scale:.3e})"
    )


@pytest.mark.parametrize("d", [2, 5])
def test_jax_logdensity_matches_raw_gp_mean_up_to_constant(d):
    """The NUTS log-density works on the unconstrained space and drops the
    y-offset, so it must match the raw mean up to a single additive constant
    once the bijector's log-Jacobian is removed."""
    jax = pytest.importorskip("jax")
    from gpry.mc_interfaces import _build_jax_logdensity

    sur, bounds = _fitted_surrogate(d=d, seed=10 + d)
    lo, hi = sur.preprocessing_X.transform_bounds(bounds).T
    beta = 1.0
    logdens, x_of_u, u_of_x = _build_jax_logdensity(sur, lo, hi, beta)

    rng = np.random.default_rng(7)
    Xt = rng.uniform(bounds[:, 0], bounds[:, 1], size=(30, d))
    Xn = sur.preprocessing_X.transform(Xt)
    U = np.array([u_of_x(xn) for xn in Xn])

    vals = np.array([float(logdens(jax.numpy.asarray(u))) for u in U])
    # strip the bijector log-Jacobian to recover beta * mean
    width = hi - lo
    log_jac = np.array([
        np.sum(np.log(width) - np.logaddexp(0, -u) - np.logaddexp(0, u)) for u in U
    ])
    got = vals - log_jac
    ref = beta * _raw_gp_mean(sur, Xt)
    diff = got - ref
    assert np.ptp(diff) < RTOL * max(np.max(np.abs(ref)), 1.0), (
        f"NUTS log-density is not a constant offset from the GP mean at d={d}: "
        f"spread of the difference = {np.ptp(diff):.3e}"
    )


def test_x_preprocessor_probe_reproduction():
    """The JAX backend reproduces the x-preprocessor from two probe points,
    which is only valid while it is diagonal-affine. Assert that it is."""
    sur, bounds = _fitted_surrogate(d=6, seed=3)
    d = 6
    t0 = np.ravel(sur.preprocessing_X.transform(np.zeros((1, d))))
    t1 = np.ravel(sur.preprocessing_X.transform(np.ones((1, d))))
    scale, offset = t1 - t0, t0
    rng = np.random.default_rng(11)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, d))
    got = offset + scale * X
    ref = sur.preprocessing_X.transform(X)
    assert np.allclose(got, ref, atol=1e-12), (
        "x-preprocessor is no longer diagonal-affine; the two-point probe in "
        "build_jax_gp_loglike is invalid."
    )


def test_y_preprocessor_is_affine():
    """`std_y` is recovered by probing `inverse_transform_scale(ones)`, which
    assumes an affine y-preprocessor. A nonlinear one (e.g. a soft clip) would
    silently produce a wrong target rather than failing."""
    sur, _ = _fitted_surrogate(d=3, seed=4)
    pre = sur.preprocessing_y
    scale = float(np.ravel(pre.inverse_transform_scale(np.ones(1)))[0])
    offset = float(np.ravel(pre.inverse_transform(np.zeros(1)))[0])
    probe = np.linspace(-3.0, 3.0, 25)
    got = offset + scale * probe
    ref = np.ravel(pre.inverse_transform(probe))
    assert np.allclose(got, ref, atol=1e-10), (
        "y-preprocessor is not affine; the scale/offset probe used by the JAX "
        "backends is invalid and would silently mis-scale the target."
    )


def test_extract_kernel_rejects_sums_instead_of_silently_mangling_them():
    """
    A genuine sum of two stationary kernels is not a single scaled stationary
    kernel, so the JAX backend cannot represent it. It must REJECT it.

    Regression guard: `Sum` and `Product` both expose `k1`/`k2`, so a walker that
    keys on those attributes treats a sum exactly like a product -- silently
    returning amp = c1*c2 and the last length scale, i.e. a wrong number with no
    error. This was the behaviour until 2026-08-19.
    """
    from gpry.kernels import RBF, ConstantKernel as C
    from gpry.mc_interfaces import _extract_stationary_kernel

    k = C(2.0) * RBF(np.array([1.0, 1.0])) + C(3.0) * RBF(np.array([5.0, 5.0]))
    with pytest.raises(ValueError, match="cannot represent the sum"):
        _extract_stationary_kernel(k)


def test_extract_kernel_accepts_the_default_and_drops_white_noise():
    """GPry's default `C * RBF + WhiteKernel` must still work: the white term
    contributes nothing to k(x*, X_train) off the training set."""
    from gpry.kernels import RBF, ConstantKernel as C, WhiteKernel
    from gpry.mc_interfaces import _extract_stationary_kernel

    base = C(4.0) * RBF(np.array([2.0, 3.0]))
    amp0, ell0, fam0, nu0 = _extract_stationary_kernel(base)
    amp1, ell1, fam1, nu1 = _extract_stationary_kernel(base + WhiteKernel(0.1))
    assert (amp0, fam0, nu0) == (4.0, "rbf", None)
    assert np.allclose(ell0, [2.0, 3.0])
    assert (amp1, fam1, nu1) == (amp0, fam0, nu0) and np.allclose(ell1, ell0), (
        "adding white noise changed the extracted mean kernel"
    )


def test_extract_kernel_rejects_two_stationary_factors():
    """A product of two stationary kernels is also not one (amp, length_scale)."""
    from gpry.kernels import RBF, ConstantKernel as C
    from gpry.mc_interfaces import _extract_stationary_kernel

    k = C(2.0) * RBF(np.array([1.0, 1.0])) * RBF(np.array([4.0, 4.0]))
    with pytest.raises(ValueError, match="single stationary factor"):
        _extract_stationary_kernel(k)


def test_fixture_still_exercises_the_float32_failure_regime():
    """
    Guard against the suite silently losing its teeth.

    A float32 regression is only catastrophic when the output scale is large and
    ``alpha`` strongly cancels (GPry's characteristic regime: amp ~ 1e6). On a
    benign fixture (amp ~ 1) float32 agrees to ~1e-16 and the consistency tests
    above would pass even if precision were dropped. At least one fixture must
    therefore reach the dangerous regime, or those tests prove nothing.
    """
    from gpry.mc_interfaces import _extract_stationary_kernel

    sur, _ = _fitted_surrogate(d=2, seed=2)
    amp, _ell, _fam, _nu = _extract_stationary_kernel(sur.gpr.kernel_)
    alpha_max = float(np.max(np.abs(np.ravel(sur.gpr.alpha_))))
    assert amp > 1e3 and alpha_max > 10.0, (
        f"the d=2 fixture no longer reaches the large-amplitude/cancelling-alpha "
        f"regime (amp={amp:.3e}, max|alpha|={alpha_max:.3e}), so the JAX "
        f"consistency tests can no longer detect a float32 regression."
    )


def test_nonlinear_y_preprocessor_is_rejected_not_silently_mis_scaled():
    """
    The JAX backends undo the y-preprocessing with a single scale recovered by
    probing. That is valid only for an affine preprocessor -- a nonlinear one
    (a soft clip, say) would not raise on its own, it would silently mis-scale
    the sampling target. `_y_scale_offset` must therefore verify affinity.
    """
    pytest.importorskip("jax")
    from gpry.mc_interfaces import _y_scale_offset, build_jax_gp_loglike

    sur, _ = _fitted_surrogate(d=3, seed=4)
    scale, offset = _y_scale_offset(sur)          # affine: must succeed
    assert np.isfinite(scale) and np.isfinite(offset)

    class _SoftClipY:
        """Stand-in for a nonlinear y-preprocessor (cf. the SoftClipY idea)."""

        def __init__(self, inner):
            self._inner = inner

        def inverse_transform_scale(self, x):
            return self._inner.inverse_transform_scale(x)

        def inverse_transform(self, y):
            y = np.asarray(y, float)
            return self._inner.inverse_transform(np.tanh(y / 3.0) * 3.0)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    sur.preprocessing_y = _SoftClipY(sur.preprocessing_y)
    with pytest.raises(TypeError, match="affine y-preprocessor"):
        _y_scale_offset(sur)
    # and the guard must hold through the public entry point, not just the helper
    with pytest.raises(TypeError, match="affine y-preprocessor"):
        build_jax_gp_loglike(sur)


# --------------------------------------------------------------------------- #
# Kernel-level agreement.
#
# The tests above compare end-to-end (jax_loglike vs surrogate.predict), which
# catches drift but does not localise it. These compare the JAX kernel directly
# against gpry.kernels, so a failure points at the kernel math rather than
# anywhere in the stack. They also pin the invariant that the planned
# array-namespace refactor (one kernel source evaluated under numpy or JAX) has
# to preserve.
# --------------------------------------------------------------------------- #
KERNEL_CASES = [
    ("rbf", None),
    ("matern", 0.5),
    ("matern", 1.5),
    ("matern", 2.5),
]


@pytest.mark.parametrize("family,nu", KERNEL_CASES)
@pytest.mark.parametrize("d", [1, 3, 7])
def test_jax_k_vec_matches_gpry_kernel(family, nu, d):
    """`_jax_k_vec` must reproduce `gpry.kernels`' own k(x*, X_train)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from gpry.kernels import RBF, Matern, ConstantKernel as C
    from gpry.mc_interfaces import _jax_k_vec, _ensure_x64

    _ensure_x64()
    rng = np.random.default_rng(abs(hash((family, nu, d))) % (2 ** 31))
    ell = rng.uniform(0.3, 3.0, size=d)
    amp = float(rng.uniform(0.5, 4.0))
    Xtr = rng.normal(size=(23, d))
    base = RBF(ell) if family == "rbf" else Matern(ell, nu=nu)
    kernel = C(amp) * base

    for x in rng.normal(size=(7, d)):
        got = np.asarray(_jax_k_vec(jnp.asarray(x), jnp.asarray(Xtr),
                                    jnp.asarray(ell), amp, family, nu))
        ref = np.ravel(kernel(np.atleast_2d(x), Xtr))
        assert np.allclose(got, ref, rtol=0, atol=1e-12), (
            f"JAX kernel disagrees with gpry.kernels for {family} nu={nu} d={d}: "
            f"max |diff| = {np.max(np.abs(got - ref)):.3e}"
        )


def test_jax_k_vec_rejects_unsupported_matern_nu():
    """Only nu in {0.5, 1.5, 2.5} have closed forms in the backend; a general nu
    needs a Bessel function and must be refused, not approximated."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from gpry.mc_interfaces import _jax_k_vec

    with pytest.raises(ValueError, match="nu="):
        _jax_k_vec(jnp.zeros(2), jnp.ones((3, 2)), jnp.ones(2), 1.0, "matern", 1.75)


def test_jax_k_vec_padding_contract():
    """The multi-chain runner pads X_train to a fixed capacity with zero rows and
    zero alpha entries. Padding must not change the resulting GP mean."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from gpry.mc_interfaces import _jax_k_vec, _ensure_x64

    _ensure_x64()
    rng = np.random.default_rng(5)
    d, n, cap = 4, 12, 32
    ell, amp = rng.uniform(0.5, 2.0, size=d), 1.7
    Xtr, alpha, x = rng.normal(size=(n, d)), rng.normal(size=n), rng.normal(size=d)

    Xpad = np.zeros((cap, d)); Xpad[:n] = Xtr
    apad = np.zeros(cap); apad[:n] = alpha

    plain = float(np.dot(np.asarray(_jax_k_vec(
        jnp.asarray(x), jnp.asarray(Xtr), jnp.asarray(ell), amp, "rbf", None)), alpha))
    padded = float(np.dot(np.asarray(_jax_k_vec(
        jnp.asarray(x), jnp.asarray(Xpad), jnp.asarray(ell), amp, "rbf", None)), apad))
    assert abs(plain - padded) < 1e-12 * max(1.0, abs(plain)), (
        f"padding changed the GP mean: {plain!r} vs {padded!r}"
    )
