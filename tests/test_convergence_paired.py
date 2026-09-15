"""The paired (common-sample) GaussianKL estimator.

The naive estimator compares moments of two INDEPENDENT MC samples, so even when
the two surrogates are identical it returns the sampling noise rather than zero.
That noise floor grows as d(d+3)/2N, which at d=8, N=2000 is 0.022 -- above the
default convergence limit of 0.02, so the criterion can never be satisfied no
matter how good the surrogate is. The paired estimator puts both surrogates on
the SAME sample, so the noise is common to both moment estimates and cancels.

Each test below pins an independent expectation: a closed form, an exact
analytic zero, or the scaling law -- not "runs and is finite".
"""
import numpy as np
import pytest

from gpry.convergence import GaussianKL
from gpry.tools import kl_norm


def _crit(**params):
    return GaussianKL([[-10, 10]] * 4, params)


class _FakeGP:
    """Stands in for a fitted GPR: carries the attributes _snapshot manipulates."""

    def __init__(self, n=64, d=4):
        rng = np.random.default_rng(0)
        self.X_train_ = rng.normal(size=(n, d))
        self.alpha_ = rng.normal(size=n)
        self.L_ = rng.normal(size=(n, n))  # the expensive thing _snapshot drops
        self.V_ = rng.normal(size=(n, n))


class _FakeSurrogate:
    """A surrogate whose predictive mean IS a given log-pdf."""

    def __init__(self, mean, cov):
        self.mean, self.cov = np.asarray(mean, float), np.asarray(cov, float)
        self._inv = np.linalg.inv(self.cov)
        self._norm = -0.5 * np.log(np.linalg.det(2 * np.pi * self.cov))
        self.gpr = _FakeGP()
        self.n_total = self.n_regress = 0

    def predict(self, X, validate=False, **kwargs):
        Xc = np.asarray(X, float) - self.mean
        return self._norm - 0.5 * np.einsum("ij,jk,ik->i", Xc, self._inv, Xc)


# --------------------------------------------------------------------------- #
#  _moments: must agree with numpy's weighted mean and unbiased covariance
# --------------------------------------------------------------------------- #
def test_moments_matches_numpy_for_uniform_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    mean, cov = GaussianKL._moments(X, np.ones(len(X)))
    assert np.allclose(mean, X.mean(axis=0), rtol=0, atol=1e-12)
    # w = 1/N each => cov / (1 - N * (1/N)^2) = cov * N/(N-1), i.e. ddof=1
    assert np.allclose(cov, np.cov(X, rowvar=False, ddof=1), rtol=1e-10)


def test_moments_matches_numpy_for_nonuniform_weights():
    rng = np.random.default_rng(2)
    X, w = rng.normal(size=(500, 4)), rng.exponential(size=500)
    mean, cov = GaussianKL._moments(X, w)
    assert np.allclose(mean, np.average(X, axis=0, weights=w), rtol=1e-12)
    assert np.allclose(cov, np.cov(X, rowvar=False, aweights=w, ddof=1), rtol=1e-10)


# --------------------------------------------------------------------------- #
#  the property the whole change exists for: identical surrogates -> exact zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d, n", [(2, 500), (4, 1000), (8, 2000)])
def test_identical_surrogates_give_exactly_zero(d, n):
    rng = np.random.default_rng(3)
    surr = _FakeSurrogate(np.zeros(d), np.eye(d))
    c = GaussianKL([[-10, 10]] * d, {})
    c._prev_surrogate = surr  # same object => logr == 0 identically
    X = rng.normal(size=(n, d))
    kl, ess = c._paired_kl(X, np.ones(n), surr)
    assert kl == pytest.approx(0.0, abs=1e-12)
    assert ess == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
#  ...and that this is NOT what the naive estimator does (the scaling law)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d, n", [(4, 1000), (8, 2000)])
def test_naive_estimator_floor_follows_d_d_plus_3_over_2n(d, n):
    """Two independent samples from the SAME distribution: the naive KL returns
    d(d+3)/2N, not 0. At d=8, N=2000 that is 0.022 -- above the default 0.02
    limit, so convergence is unreachable however good the surrogate is."""
    rng = np.random.default_rng(4)
    vals = []
    for _ in range(30):
        A, B = rng.normal(size=(n, d)), rng.normal(size=(n, d))
        vals.append(kl_norm(A.mean(0), np.cov(A, rowvar=False),
                            B.mean(0), np.cov(B, rowvar=False)))
    floor = d * (d + 3) / (2 * n)
    assert np.mean(vals) == pytest.approx(floor, rel=0.35)
    # the paired estimator on the same d, N is ~12 orders of magnitude below it
    surr = _FakeSurrogate(np.zeros(d), np.eye(d))
    c = GaussianKL([[-10, 10]] * d, {})
    c._prev_surrogate = surr
    kl, _ = c._paired_kl(rng.normal(size=(n, d)), np.ones(n), surr)
    assert abs(kl) < 1e-6 * floor


# --------------------------------------------------------------------------- #
#  closed form: the estimator must recover the analytic KL between two Gaussians
# --------------------------------------------------------------------------- #
def test_paired_kl_recovers_the_analytic_kl():
    d, n = 4, 400000
    rng = np.random.default_rng(5)
    m_new, C_new = np.zeros(d), np.eye(d)
    # C_old < C_new keeps the importance weights bounded (the reweighted target
    # is narrower than the sampling density), so ESS stays healthy.
    m_old = np.full(d, 0.3)
    C_old = 0.85 * np.eye(d)
    exact = kl_norm(m_new, C_new, m_old, C_old)

    c = _crit()
    c._prev_surrogate = _FakeSurrogate(m_old, C_old)
    surr = _FakeSurrogate(m_new, C_new)
    X = rng.multivariate_normal(m_new, C_new, size=n)  # sampled under the NEW surrogate
    kl, ess = c._paired_kl(X, np.ones(n), surr)

    assert ess > c.min_ess_frac
    assert kl == pytest.approx(exact, rel=0.02), f"got {kl}, exact {exact}"


def test_paired_kl_direction_matches_the_naive_branch():
    """kl_norm is asymmetric; the paired branch must keep main's KL(new||old)."""
    d, n = 3, 200000
    rng = np.random.default_rng(6)
    m_new, C_new = np.zeros(d), np.eye(d)
    m_old, C_old = np.full(d, 0.4), 0.8 * np.eye(d)
    c = GaussianKL([[-10, 10]] * d, {})
    c._prev_surrogate = _FakeSurrogate(m_old, C_old)
    kl, _ = c._paired_kl(rng.multivariate_normal(m_new, C_new, size=n),
                         np.ones(n), _FakeSurrogate(m_new, C_new))
    fwd = kl_norm(m_new, C_new, m_old, C_old)
    rev = kl_norm(m_old, C_old, m_new, C_new)
    assert abs(kl - fwd) < abs(kl - rev)
    assert kl == pytest.approx(fwd, rel=0.03)


# --------------------------------------------------------------------------- #
#  guards
# --------------------------------------------------------------------------- #
def test_collapsed_weights_report_infinity_not_a_number():
    """Surrogates far apart => reweighting is meaningless. That is itself 'not
    converged', so the criterion must say inf rather than a plausible number."""
    d, n = 4, 20000
    rng = np.random.default_rng(7)
    c = _crit()
    c._prev_surrogate = _FakeSurrogate(np.full(d, 12.0), np.eye(d))  # miles away
    kl, ess = c._paired_kl(rng.normal(size=(n, d)), np.ones(n),
                           _FakeSurrogate(np.zeros(d), np.eye(d)))
    assert not np.isfinite(kl)
    assert ess < c.min_ess_frac


def test_non_finite_predictions_are_rejected():
    d, n = 4, 1000
    rng = np.random.default_rng(8)
    surr = _FakeSurrogate(np.zeros(d), np.eye(d))
    c = _crit()
    c._prev_surrogate = surr
    w = np.ones(n)
    w[: n // 2 + 1] = np.nan  # majority unusable
    kl, _ = c._paired_kl(rng.normal(size=(n, d)), w, surr)
    assert not np.isfinite(kl)


# --------------------------------------------------------------------------- #
#  _snapshot must not corrupt the live GP it copies from
# --------------------------------------------------------------------------- #
def test_snapshot_restores_the_cholesky_on_the_original():
    surr = _FakeSurrogate(np.zeros(4), np.eye(4))
    L_before, V_before = surr.gpr.L_, surr.gpr.V_
    snap = GaussianKL._snapshot(surr)
    assert surr.gpr.L_ is L_before, "live GP left without its Cholesky factor"
    assert surr.gpr.V_ is V_before
    assert snap is not None and snap is not surr
    assert snap.gpr.L_ is None, "snapshot should not carry the n x n factor"
    # and the frozen copy still predicts identically
    X = np.random.default_rng(9).normal(size=(50, 4))
    assert np.allclose(snap.predict(X), surr.predict(X))


def test_snapshot_of_an_unfitted_surrogate_is_none():
    class _NoGP:
        gpr = None

    assert GaussianKL._snapshot(_NoGP()) is None


# --------------------------------------------------------------------------- #
#  opt-out and defaults
# --------------------------------------------------------------------------- #
def test_paired_is_the_default_and_can_be_switched_off():
    assert _crit().paired is True
    assert _crit(paired=False).paired is False
    assert _crit().min_ess_frac == 0.25
    assert _crit(min_ess_frac=0.5).min_ess_frac == 0.5


def test_no_snapshot_is_taken_when_paired_is_off():
    """With paired=False the snapshot is pure cost: deep-copied every iteration and
    pickled into the checkpoint, where it dominates con.pkl."""
    surr = _FakeSurrogate(np.zeros(4), np.eye(4))
    assert _crit(paired=False)._maybe_snapshot(surr) is None
    assert _crit(paired=True)._maybe_snapshot(surr) is not None


# --------------------------------------------------------------------------- #
#  the regression this port exists to prevent
# --------------------------------------------------------------------------- #
def test_paired_branch_is_actually_reached_with_nora(tmp_path):
    """A paired estimator that is never called is worth nothing.

    The previous paired implementation lived in ``_get_new_mean_and_cov_from_mc``
    and was unreachable: ``_get_new_mean_and_cov`` tries
    ``_get_new_mean_and_cov_from_acquisition`` first, and that only raises
    AttributeError when the acquisition holds no sample -- which, with NORA, it
    never does. No unit test of the estimator can catch that, so run the real
    Runner and assert the branch fires.
    """
    from gpry.run import Runner

    d = 4
    rng = np.random.default_rng(0)
    A = rng.normal(size=(d, d))
    cov = 0.05 * (A @ A.T + d * np.eye(d))
    inv = np.linalg.inv(cov)

    def loglike(x0, x1, x2, x3):
        v = np.array([x0, x1, x2, x3], float)
        return float(-0.5 * v @ inv @ v)

    fired = []
    original = GaussianKL._paired_kl

    def spy(self, X, w, surr):
        fired.append(True)
        return original(self, X, w, surr)

    GaussianKL._paired_kl = spy
    try:
        runner = Runner(
            loglike=loglike,
            bounds=np.array([[-5.0, 5.0]] * d),
            params=[f"x{i}" for i in range(d)],
            gp_acquisition="NORA",
            convergence_criterion={"GaussianKL": {"paired": True, "limit": 1e-3}},
            options={"n_points_per_acq": 4, "max_total": 180, "max_initial": 60},
            checkpoint=str(tmp_path / "ckpt"),
            load_checkpoint="overwrite",
            seed=3,
            verbose=1,
        )
        runner.run()
    finally:
        GaussianKL._paired_kl = original

    assert fired, "paired branch never reached -- the estimator is dead code again"
    assert runner.has_converged, (
        "did not converge on a 4-D Gaussian within 180 evaluations; with the naive "
        "estimator this run does not converge at all, which is the bug being fixed"
    )
