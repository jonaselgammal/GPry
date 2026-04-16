import numpy as np
from sklearn.base import clone

from gpry.gpr import GaussianProcessRegressor


def _make_gpr(use_jax):
    d = 2
    length_scale_prior = np.array([[0.01, 100.0]] * d)
    return GaussianProcessRegressor(
        kernel="RBF",
        output_scale_prior=[0.01, 100.0],
        length_scale_prior=length_scale_prior,
        noise_level=0.01,
        n_restarts_optimizer=1,
        random_state=0,
        use_jax=use_jax,
    )


def test_data_driven_hyperopt_starts_are_added_on_simple_refits(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(20, 2))
    y = -0.5 * np.sum((X - 0.5) ** 2 / np.array([0.04, 0.12]), axis=1)

    gpr = _make_gpr(use_jax=False)
    gpr.fit(X, y, validate=False)

    starts = []

    def fake_opt(obj_func, theta_initial, bounds):
        starts.append(np.array(theta_initial, copy=True))
        return np.array(theta_initial, copy=True), float(obj_func(theta_initial, eval_gradient=False))

    monkeypatch.setattr(gpr, "_constrained_optimization", fake_opt)
    gpr._fit_hyperparameters(start_from_current=True, n_restarts=1)

    assert len(starts) >= 3
    rounded = {tuple(np.round(theta, 8)) for theta in starts}
    assert len(rounded) >= 2


def test_fit_with_data_driven_starts_keeps_jax_and_numpy_paths_working():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(30, 3))
    center = np.array([0.25, 0.55, 0.75])
    widths = np.array([0.05, 0.09, 0.12])
    y = -0.5 * np.sum((X - center) ** 2 / widths, axis=1)

    for use_jax in (False, True):
        gpr = GaussianProcessRegressor(
            kernel="RBF",
            output_scale_prior=[0.01, 100.0],
            length_scale_prior=np.array([[0.01, 100.0]] * 3),
            noise_level=0.01,
            n_restarts_optimizer=2,
            random_state=0,
            use_jax=use_jax,
        )
        gpr.fit(X, y, validate=False)
        assert np.isfinite(gpr.log_marginal_likelihood_value_)
        preds = gpr.predict(X[:5], return_std=True, validate=False)
        assert np.all(np.isfinite(preds[0]))
        assert np.all(np.isfinite(preds[1]))
