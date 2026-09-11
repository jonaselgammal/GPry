"""
Regression tests for the noise floor that used to sit inside ``LogExp``.

THE BUG. ``LogExp`` (and ``NonlinearLogExp``, and the mask in ``BaseLogExp.__call__``)
computed :math:`\\log\\sqrt{\\sigma^2-\\sigma_n^2}`, which treats :math:`\\sigma` as the
predictive std of a new OBSERVATION and therefore as having :math:`\\sigma_n` for a
floor. It is not: ``predict(return_std=True)`` returns the LATENT std, which already
excludes the observation noise. At a training point the latent std tends to
:math:`\\sigma_n` from above -- with a fixed nugget the posterior variance there is
:math:`k\\alpha/(k+\\alpha)\\to\\alpha=\\sigma_n^2` -- so the difference is ~0 wherever the
training set is dense and negative under rounding. The mask then scored those
candidates :math:`-\\infty`.

WHY IT WAS INVISIBLE. It is gated by :math:`\\sigma_n`, and the default is 1e-2, far
below any realistic latent std away from the training points. It only bites when the
noise level is set to something a genuinely noisy objective requires. Measured on a
Planck surrogate at ``noise_level = 1``: the latent std across 1920 candidates had a
maximum of 1.0212, so exactly ONE cleared the floor and 1919 scored -inf. The ranked
pool came back empty and the run raised ``GPAcquisitionError`` -- reported as a
convergence failure, when in fact the acquisition simply could not express a
preference. Independently reproduced on hyperparameter-tuning surfaces: with the noise
declared, 4 of 15 runs at :math:`\\sigma/T=1` died this way; 0 of 15 with the term
dropped.

WHAT IS PINNED HERE
  1. the acquisition is finite at the training points themselves, where the old
     expression was exactly at (or below) its floor;
  2. it stays finite for a large declared noise level over a dense candidate set --
     the empty-pool condition;
  3. it remains strictly increasing in ``std``, i.e. dropping the term did not break
     the exploration ordering the acquisition exists to provide;
  4. the analytic gradient matches finite differences, since the value and the
     gradient used two DIFFERENT and inconsistent noise corrections before the fix
     (``sqrt(std**2 - sigma_n**2)`` against ``std - sigma_n``);
  5. the value no longer depends on ``noise_level`` at all.
"""
import numpy as np
import pytest

from gpry.acquisition_functions import LogExp, NonlinearLogExp
from gpry.preprocessing import NormalizeBounds, NormalizeY
from gpry.surrogate import SurrogateModel

BOX = 3.0


def _fitted_gp(noise_level, d=2, n=40, seed=0):
    """A surrogate fitted on a smooth target, with a noise level a caller would declare.

    Built as a SurrogateModel rather than a bare regressor because that is what the
    acquisition is handed at run time (it reads ``y_max`` and ``noise_level`` off it,
    and calls ``predict(..., validate=)``).
    """
    rng = np.random.default_rng(seed)
    bounds = np.array([[-BOX, BOX]] * d)
    sur = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor={"kernel": "RBF", "output_scale_prior": [1e-2, 1e3],
                   "length_scale_prior": [1e-3, 1e2], "noise_level": noise_level,
                   "optimizer": "fmin_l_bfgs_b", "n_restarts_optimizer": 2},
        infinities_classifier={"svm": {"threshold": "20s"}},
        random_state=rng, verbose=0,
    )
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, d))
    sur.append(X, -0.5 * np.sum(X**2, axis=1))
    return sur, X


@pytest.mark.parametrize("noise_level", [1e-2, 0.5, 1.0, 3.0])
def test_finite_at_training_points(noise_level):
    """The old expression was at its floor exactly where the GP has been evaluated."""
    gp, X = _fitted_gp(noise_level)
    values = LogExp(zeta=1.0, dimension=X.shape[1])(X, gp)
    assert np.all(np.isfinite(values)), (
        f"{np.sum(~np.isfinite(values))}/{len(values)} training points scored -inf "
        f"at noise_level={noise_level}"
    )


@pytest.mark.parametrize("noise_level", [1.0, 3.0])
def test_candidate_pool_does_not_empty(noise_level):
    """The failure mode itself: a pool in which nothing can be ranked."""
    gp, X = _fitted_gp(noise_level)
    rng = np.random.default_rng(1)
    pool = rng.uniform(-BOX, BOX, size=(500, X.shape[1]))
    values = LogExp(zeta=1.0, dimension=X.shape[1])(pool, gp)
    finite = np.isfinite(values)
    assert finite.mean() > 0.99, (
        f"only {finite.sum()}/{len(pool)} candidates are rankable at "
        f"noise_level={noise_level}; the ranked pool would come back empty"
    )
    assert len(np.unique(values[finite])) > 1, "acquisition cannot express a preference"


def test_monotone_in_std():
    """Dropping the noise term must not disturb the exploration ordering."""
    mu = np.zeros(5)
    std = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
    for fn in (LogExp.f, NonlinearLogExp.f):
        v = fn(mu, std, 0.0, 1.0, 1.0)
        assert np.all(np.diff(v) > 0), f"{fn.__qualname__} is not increasing in std"


def test_value_is_independent_of_noise_level():
    """`noise_level` is retained in the signature but must no longer enter the value."""
    mu, std = np.zeros(4), np.array([0.05, 0.5, 1.0, 2.0])
    for fn in (LogExp.f, NonlinearLogExp.f):
        ref = fn(mu, std, 0.0, 0.0, 1.0)
        for nl in (1e-2, 1.0, 5.0):
            np.testing.assert_allclose(fn(mu, std, 0.0, nl, 1.0), ref, rtol=0, atol=0)


class _AnalyticGP:
    """A stand-in exposing just the surrogate API the acquisition uses.

    A real surrogate returns gradients in its own PREPROCESSED coordinates, which
    makes a finite-difference check in raw coordinates a test of the preprocessor's
    Jacobian rather than of the acquisition. This stub removes that: mu, std and both
    gradients are analytic in one coordinate system, so any mismatch is the
    acquisition's own. The std is deliberately taken BELOW the noise level over part
    of the domain -- the regime where the old expression was clipped to zero.
    """

    y_max = 0.0

    def __init__(self, noise_level):
        self.noise_level = noise_level

    @staticmethod
    def _mu(X):
        return np.sum(np.sin(X), axis=1)

    @staticmethod
    def _std(X):
        return 0.3 + 0.2 * np.cos(np.sum(X, axis=1))

    def predict(self, X, return_std=False, return_mean_grad=False,
                return_std_grad=False, validate=True):
        X = np.atleast_2d(X)
        out = [self._mu(X)]
        if return_std:
            out.append(self._std(X))
        if return_mean_grad:
            out.append(np.cos(X))
        if return_std_grad:
            out.append(-0.2 * np.sin(np.sum(X, axis=1))[:, None] * np.ones_like(X))
        return tuple(out)


@pytest.mark.parametrize("noise_level", [1e-2, 0.5, 1.0])
def test_gradient_matches_finite_differences(noise_level):
    """Value and gradient used two different noise corrections before the fix.

    The value computed ``log sqrt(std**2 - sigma_n**2)`` while the gradient computed
    ``std_grad / (std - sigma_n)`` -- not the derivative of the value under any
    convention. They must now agree.
    """
    gp = _AnalyticGP(noise_level)
    acq = LogExp(zeta=1.0, dimension=2)
    x = np.array([[0.7, -1.1]])
    _, grad = acq(x, gp, eval_gradient=True)
    grad = np.atleast_2d(np.asarray(grad, float))[0]
    eps = 1e-6
    num = np.empty_like(grad)
    for j in range(x.shape[1]):
        xp, xm = x.copy(), x.copy()
        xp[0, j] += eps
        xm[0, j] -= eps
        num[j] = (acq(xp, gp)[0] - acq(xm, gp)[0]) / (2 * eps)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-6)


def test_std_below_noise_level_is_still_rankable():
    """The exact regime that emptied the pool: latent std everywhere under sigma_n."""
    gp = _AnalyticGP(noise_level=1.0)          # std in [0.1, 0.5], always < 1.0
    rng = np.random.default_rng(3)
    X = rng.uniform(-np.pi, np.pi, size=(200, 2))
    values = LogExp(zeta=1.0, dimension=2)(X, gp)
    assert np.all(np.isfinite(values))
    assert len(np.unique(values)) > 1
