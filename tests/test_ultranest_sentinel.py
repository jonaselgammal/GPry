"""
Regression tests for the finite stand-in used for ``-inf`` on the UltraNest paths,
and for the batch-composition independence of :meth:`SurrogateModel.predict`.

The bug these pin: ``_do_mc_sample_ultranest`` (and `gpry.mc.mc_sample_from_gp_ns`)
set ``surrogate.minus_inf_value = -1e-300`` so that UltraNest, which cannot handle
-inf, would see a finite value. But -1e-300 is zero to ~300 digits -- the *largest*
log-posterior representable, not the smallest. Nested sampling therefore converged
onto the classifier-masked region rather than the posterior, terminating after ~2
e-folds with `Explored until L=-1e-300`.

It is invisible at low dimension and total at high dimension, because it is gated by
how much of the prior the classifier masks: ~0 at d=8, ~2e-4 at d=16, and 0.18-0.50
at d=30 on the paper's targets. That is why it survived a full benchmark campaign --
and why it corrupted exactly the two high-d cells.

The second, coupled hazard pinned here: `SurrogateModel.predict` took an early return
when a batch was entirely masked, which skipped the upper clipper. The same point
could then come back with two different values depending on which other points shared
its batch. The clipper is now applied on both return paths, so masked points score
identically however they are batched -- for *any* value of ``minus_inf_value``.

The substitute is now the constant ``MINUS_INF_SUBSTITUTE = -1e30``: far below any
realistic log-posterior, but with enough headroom left to survive squaring, summing
and a float32 cast -- which ``-1e300`` (the likely intent of the ``-1e-300`` typo)
does not.
"""
import warnings

import numpy as np
import pytest

from gpry.preprocessing import NormalizeBounds, NormalizeY
from gpry.surrogate import MINUS_INF_SUBSTITUTE, SurrogateModel


def _masked_surrogate(d=8, n=90, seed=0, sigma=0.35):
    """
    A fitted surrogate whose infinities classifier masks a real fraction of the
    prior box -- the situation in which the sentinel matters at all.

    The classifier only masks where the log-posterior falls more than its
    threshold below the max (default "20s", a drop of ~218 at d=8), so the
    target must be peaked enough relative to the box for that to happen at all.
    A unit Gaussian in a +-6 box spans only ~144 and masks NOTHING, which is why
    a naive fixture makes every test in this file skip and prove nothing. A
    sigma=0.35 target spans ~1180 and masks a substantial fraction -- the same
    regime the d=30 campaign surrogates are in (0.18-0.50 masked).
    """
    rng = np.random.default_rng(seed)
    bounds = np.array([[-6.0, 6.0]] * d)

    def logp(X):
        X = np.atleast_2d(X)
        return -0.5 * np.sum((X / sigma) ** 2, axis=1)

    sur = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor={"kernel": "RBF", "output_scale_prior": [1e-2, 1e3],
                   "length_scale_prior": [1e-3, 1e2], "noise_level": 1e-2,
                   "optimizer": "fmin_l_bfgs_b", "n_restarts_optimizer": 2},
        infinities_classifier={"svm": {"threshold": "20s"}},
        random_state=rng, verbose=0,
    )
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, d))
    sur.append(X, logp(X))
    return sur, bounds


def _split_by_mask(sur, bounds, rng, n=4000):
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))
    finite = np.asarray(
        sur.infinities_classifier.is_finite_X(
            sur.preprocessing_X.transform(X), validate=False),
        dtype=bool,
    )
    return X[finite], X[~finite]


def test_substitute_is_worse_than_the_finite_region():
    """
    The stand-in must sit BELOW every value attainable in the finite region.
    With -1e-300 it sat above all of them, so nested sampling maximised straight
    into the masked region.
    """
    rng = np.random.default_rng(1)
    sur, bounds = _masked_surrogate(seed=1)
    ok, masked = _split_by_mask(sur, bounds, rng)
    if len(masked) < 20:
        pytest.skip("classifier masks too little of this box to exercise the path")

    sentinel = sur.minus_inf_value_substitute
    assert np.isfinite(sentinel), "the stand-in for -inf must be finite"

    finite_vals = np.ravel(sur.predict(ok[:200], validate=False))
    assert sentinel < np.min(finite_vals), (
        f"the -inf stand-in ({sentinel:.4g}) is not below the worst attainable "
        f"finite value ({np.min(finite_vals):.4g}); a maximiser will run into the "
        "masked region"
    )
    # and the specific historical value must not come back
    assert sentinel < -1.0, f"sentinel {sentinel!r} is not a 'very bad' value"


def test_masked_prediction_is_at_most_the_minimum_finite_prediction():
    """
    What the sampler actually sees: the value `predict` returns for a masked point,
    after clipping, must not exceed the lowest value it returns for a finite one.
    """
    rng = np.random.default_rng(4)
    sur, bounds = _masked_surrogate(seed=4)
    ok, masked = _split_by_mask(sur, bounds, rng)
    if len(masked) < 20 or len(ok) < 20:
        pytest.skip("need both masked and finite points to compare")

    prev = sur.minus_inf_value
    sur.minus_inf_value = sur.minus_inf_value_substitute
    try:
        masked_vals = np.ravel(sur.predict(masked[:64], validate=False))
        finite_vals = np.ravel(sur.predict(ok[:200], validate=False))
    finally:
        sur.minus_inf_value = prev

    assert np.max(masked_vals) <= np.min(finite_vals), (
        f"masked points score up to {np.max(masked_vals):.4g}, which is above the "
        f"worst finite prediction ({np.min(finite_vals):.4g})"
    )


@pytest.mark.parametrize(
    "sentinel_kind", ["default_minus_inf", "substitute", "historical"]
)
def test_masked_points_do_not_depend_on_batch_composition(sentinel_kind):
    """
    `predict` early-returns when a batch is entirely masked. That path used to skip
    the upper clipper, so the same point could score differently depending on which
    other points shared its batch. It now clips on both paths, so the values agree
    for ANY `minus_inf_value` -- including the historical -1e-300, which is the case
    that used to disagree (raw -1e-300 alone vs. the clip ceiling when mixed).
    """
    rng = np.random.default_rng(2)
    sur, bounds = _masked_surrogate(seed=2)
    ok, masked = _split_by_mask(sur, bounds, rng)
    if len(masked) < 40 or len(ok) < 40:
        pytest.skip("need both masked and finite points to compare batches")

    sentinel = {
        "default_minus_inf": -np.inf,
        "substitute": sur.minus_inf_value_substitute,
        "historical": -1e-300,
    }[sentinel_kind]

    prev = sur.minus_inf_value
    sur.minus_inf_value = sentinel
    try:
        alone = np.ravel(sur.predict(masked[:32], validate=False))
        mixed = np.ravel(sur.predict(np.vstack([ok[:32], masked[:32]]),
                                     validate=False))[32:]
    finally:
        sur.minus_inf_value = prev

    assert np.array_equal(alone, mixed), (
        "masked points scored differently depending on batch composition: "
        f"all-masked batch -> {np.unique(alone)}, mixed batch -> {np.unique(mixed)}"
    )


def test_substitute_has_headroom_under_float32_and_squaring():
    """
    Why -1e30 and not something more extreme (-1e300 was the likely intent of the
    -1e-300 typo). The substitute has to survive the arithmetic that actually gets
    applied to log-posterior arrays:

    - a float32 cast, which turns -1e300 straight back into -inf -- silently
      restoring the value the substitute exists to avoid (nessai already uses
      `np.single` internally, and float32 chain storage is commonplace);
    - squaring, since `mc.py` writes these values into a `minuslogp` chain column
      that GetDist takes per-column variances over, and (1e300)**2 is inf.
    """
    v = MINUS_INF_SUBSTITUTE
    assert np.isfinite(v) and v < 0
    assert np.isfinite(np.float32(v)), (
        f"{v:.0e} casts to {np.float32(v)} in float32, i.e. back to the -inf the "
        "substitute exists to avoid"
    )
    with np.errstate(over="ignore"):
        assert np.isfinite(np.float64(v) ** 2), (
            f"{v:.0e} squares to inf; any variance over a chain column containing "
            "it would be inf"
        )
        # and a chain of ~1e8 such entries must not overflow when summed
        assert np.isfinite(np.float64(v) * 1e8)


def test_substitute_survives_infinite_training_targets():
    """
    A constant cannot pick up an infinite training target, but a data-anchored form
    can: the first version of this fix read `surrogate.y` (which *includes* points
    classified as infinite), so one -inf there would have made the substitute -inf.
    Kept as a guard in case the constant is ever replaced by an anchored value.
    """
    d = 4
    bounds = np.array([[-6.0, 6.0]] * d)
    rng = np.random.default_rng(5)
    sur = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor={"kernel": "RBF", "output_scale_prior": [1e-2, 1e3],
                   "length_scale_prior": [1e-3, 1e2], "noise_level": 1e-2,
                   "optimizer": "fmin_l_bfgs_b", "n_restarts_optimizer": 2},
        infinities_classifier={"svm": {"threshold": "20s"}},
        random_state=rng, verbose=0,
    )
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(40, d))
    y = -0.5 * np.sum(X**2, axis=1)
    y[0] = -np.inf
    sur.append(X, y)

    sentinel = sur.minus_inf_value_substitute
    assert np.isfinite(sentinel), (
        f"an infinite training target leaked into the -inf stand-in ({sentinel!r})"
    )
    assert sentinel < np.min(y[np.isfinite(y)])


def _force_training_y(sur, y):
    """
    Overwrite the regressor's training targets, to reach regimes no real fit reaches.

    Leaves `sur` internally inconsistent (`_X` no longer matches `_y`), which is fine
    here: `minus_inf_value_substitute` reads only `_y[_i_regress]` and never the GP.
    """
    sur._y = np.asarray(y, dtype=float)
    sur._i_regress = np.arange(len(sur._y))


@pytest.mark.parametrize(
    "y_min, y_max, activates",
    [
        (-560.0, -20.0, False),          # the campaign targets: the constant wins
        (-1e13, -1e12, False),           # still far above the constant
        (-1e30, -1e30 + 100.0, True),    # NB: absolute margins vanish here (see below)
        (-1e31, -1e29, True),
        (-1e-5, 1e30, True),        # |y_min| tiny, range huge: abs(y_min) useless
        (-1.0, 1e31, True),
    ],
)
def test_pathological_targets_fall_back_below_the_constant(y_min, y_max, activates):
    """
    The guard for a target whose log-posterior reaches -1e30 on its own. Its two scales
    fail in opposite regimes, so both parametrized extremes must stay strictly below
    the training minimum AND below a plausible one-range dip beneath it:

    - |y_min| ~ 1e30 with a narrow range: float64 spacing is ~1.4e14, so an absolute
      margin is a no-op (`y_min - 1000 == y_min`) and only `abs(y_min)` saves it;
    - |y_min| tiny with a huge range (unnormalised, large positive logp): `abs(y_min)`
      is negligible and only `100 * range` saves it.

    It must also stay silent, and return the plain constant, on ordinary targets --
    a warning that fires every acquisition step is noise, not information.
    """
    sur, _ = _masked_surrogate(d=3, n=20, seed=0)
    sur.verbose = 3  # the warning is gated on `verbose > 1`; the fixture builds silent
    _force_training_y(sur, [y_min, 0.5 * (y_min + y_max), y_max])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = sur.minus_inf_value_substitute

    assert np.isfinite(value)
    assert value < y_min, (
        f"stand-in {value:.6e} is not strictly below the training minimum "
        f"{y_min:.6e} (an absolute margin alone would have returned y_min here)"
    )
    if activates:
        # also below a plausible dip of one training range beneath the minimum
        assert value < y_min - (y_max - y_min), (
            f"stand-in {value:.6e} is not below a one-range dip under the training "
            f"minimum ({y_min - (y_max - y_min):.6e}); the GP mean can reach there"
        )
    if activates:
        assert value != MINUS_INF_SUBSTITUTE
        assert len(caught) == 1 and "unnormalised" in str(caught[0].message)
    else:
        assert value == MINUS_INF_SUBSTITUTE
        assert not caught, (
            "the pathological-target warning fired on an ordinary target; it must be "
            f"silent on the common path, got: {[str(w.message) for w in caught]}"
        )


def test_pathological_warning_respects_verbosity():
    """Follows the file's `verbose > 1` convention (cf. `_append_noise_level`)."""
    sur, _ = _masked_surrogate(d=3, n=20, seed=0)
    _force_training_y(sur, [-1e31, -5e30, -1e29])
    for verbose, expected in [(0, 0), (1, 0), (2, 1), (3, 1)]:
        sur.verbose = verbose
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = sur.minus_inf_value_substitute
        assert len(caught) == expected, f"verbose={verbose}: got {len(caught)} warnings"
        # the value itself must not depend on verbosity
        assert value == pytest.approx(-1e33, rel=1e-12)


def test_the_historical_sentinel_would_have_failed():
    """
    Guard the guard: demonstrate that -1e-300 really does invert the ordering, so
    a future refactor cannot quietly reintroduce it and still pass the tests above.
    """
    rng = np.random.default_rng(3)
    sur, bounds = _masked_surrogate(seed=3)
    ok, masked = _split_by_mask(sur, bounds, rng)
    if len(masked) < 20:
        pytest.skip("classifier masks too little of this box to exercise the path")

    finite_vals = np.ravel(sur.predict(ok[:200], validate=False))
    assert -1e-300 > np.max(finite_vals), (
        "expected the historical sentinel to sit ABOVE every finite value "
        "(that was the bug); if this no longer holds the fixture has drifted and "
        "the other tests in this file may no longer be exercising the failure"
    )
