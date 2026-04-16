"""
Experimental adaptive hyperparameter optimization scheduling for GPry.

This module is not part of the conservative GPry baseline. It is kept for
opt-in tests of cheaper or less frequent hyperparameter optimization.

Instead of re-optimizing GP hyperparameters on a fixed schedule, this module
tracks the log-marginal-likelihood (LML) history and adapts:
- Skip re-optimization when LML is stable (small change over recent fits)
- Reduce number of optimizer restarts as the GP stabilizes
- Switch to warm-start-only after initial exploration phase
- Reduce maxiter for warm-started runs
- Use a data subset for hyperparameter optimization when N is large

Usage
-----
Instantiated in Runner and replaces the fixed ``fit_full_every`` /
``fit_simple_every`` logic in ``_fit_surrogate_parallel()``.
"""

import numpy as np


class HyperparameterScheduler:
    """Decides when and how thoroughly to re-optimize GP hyperparameters.

    Parameters
    ----------
    fit_full_every : int or None
        Base interval for full (multi-restart) hyperparameter optimization.
    fit_simple_every : int or None
        Base interval for simple (single-restart, warm-started) optimization.
    n_restarts_base : int
        Base number of optimizer restarts for full fits.
    lml_tol : float or None
        If the LML change over the last ``lml_window`` fits is below this,
        skip the fit. If None, never skip based on LML.
    lml_window : int
        Number of recent LML values to compare for stability check.
    restart_decay : float
        Factor by which to reduce restarts as the GP stabilizes.
        After ``decay_period`` full fits, restarts are multiplied by this.
    min_restarts : int
        Minimum number of restarts (floor for decay).
    decay_period : int
        Number of full fits after which one decay step is applied.
    warm_start_only_after : int or None
        After this many full fits, switch to warm-start-only mode (1 restart
        from current params). None means never switch (decay handles it).
    maxiter_full : int
        Maximum L-BFGS-B iterations per restart for full fits.
    maxiter_warmstart : int
        Maximum L-BFGS-B iterations for warm-started runs.
        Since warm starts begin near the optimum, they converge faster.
    subset_threshold : int
        If n_training > this, use a subset for hyperparameter optimization.
    subset_size : int or None
        Size of the subset. If None, uses ``subset_threshold``.
    """

    def __init__(
        self,
        fit_full_every=None,
        fit_simple_every=1,
        n_restarts_base=5,
        lml_tol=None,
        lml_window=3,
        restart_decay=0.8,
        min_restarts=1,
        decay_period=3,
        warm_start_only_after=5,
        maxiter_full=200,
        maxiter_warmstart=50,
        subset_threshold=500,
        subset_size=None,
    ):
        self.fit_full_every = fit_full_every
        self.fit_simple_every = fit_simple_every
        self.n_restarts_base = n_restarts_base
        self.lml_tol = lml_tol
        self.lml_window = lml_window
        self.restart_decay = restart_decay
        self.min_restarts = min_restarts
        self.decay_period = decay_period
        self.warm_start_only_after = warm_start_only_after
        self.maxiter_full = maxiter_full
        self.maxiter_warmstart = maxiter_warmstart
        self.subset_threshold = subset_threshold
        self.subset_size = subset_size or subset_threshold
        # State
        self._lml_history = []
        self._n_full_fits = 0
        self._n_total_fits = 0
        self._n_skipped = 0

    def should_fit(self, iteration, n_points):
        """Decide whether to fit hyperparameters at this iteration.

        Parameters
        ----------
        iteration : int
            Current iteration index.
        n_points : int
            Current number of training points.

        Returns
        -------
        dict or False
            False if no fit should happen, or a dict with keys:
            ``start_from_current``, ``n_restarts``, ``maxiter``.
        """
        is_full = (
            self.fit_full_every is not None
            and iteration % self.fit_full_every == self.fit_full_every - 1
        )
        is_simple = (
            self.fit_simple_every is not None
            and iteration % self.fit_simple_every == self.fit_simple_every - 1
        )
        if not is_full and not is_simple:
            return False
        # Check LML stability: skip if LML hasn't changed much.
        # This applies to both simple and warm-start-only fits (not full fits).
        if self.lml_tol is not None and len(self._lml_history) >= self.lml_window:
            recent = self._lml_history[-self.lml_window:]
            lml_range = max(recent) - min(recent)
            if lml_range < self.lml_tol and not is_full:
                self._n_skipped += 1
                return False
        # After warm_start_only_after full fits, demote full fits to warm-start-only
        if (self.warm_start_only_after is not None
                and self._n_full_fits >= self.warm_start_only_after
                and is_full):
            return {
                "start_from_current": True,
                "n_restarts": 1,
                "maxiter": self.maxiter_warmstart,
            }
        if is_full:
            n_restarts = self._compute_restarts()
            self._n_full_fits += 1
            return {
                "start_from_current": True,
                "n_restarts": n_restarts,
                "maxiter": self.maxiter_full,
            }
        else:
            # Simple fit: warm-started, reduced maxiter
            return {
                "start_from_current": True,
                "n_restarts": 1,
                "maxiter": self.maxiter_warmstart,
            }

    def _compute_restarts(self):
        """Compute number of restarts with decay."""
        n_decay_steps = self._n_full_fits // self.decay_period
        n_restarts = int(self.n_restarts_base * self.restart_decay ** n_decay_steps)
        return max(self.min_restarts, n_restarts)

    def record_lml(self, lml):
        """Record a log-marginal-likelihood value after a fit."""
        self._lml_history.append(float(lml))
        self._n_total_fits += 1

    def get_training_subset(self, X, y, noise_level=None):
        """Select a subset of training data for hyperparameter optimization.

        Keeps the points near the mode (highest y) and adds a random sample
        of the rest, so the optimizer sees both the peak structure and the
        overall landscape.

        Parameters
        ----------
        X : array, shape (N, d)
        y : array, shape (N,)
        noise_level : array or float, optional

        Returns
        -------
        X_sub, y_sub, noise_sub : arrays
            Subset of training data. If N <= subset_threshold, returns
            the original arrays unchanged.
        """
        N = len(y)
        if N <= self.subset_threshold:
            return X, y, noise_level
        M = self.subset_size
        # Keep top 50% of subset from highest-y points (near the mode)
        n_top = M // 2
        n_random = M - n_top
        sorted_idx = np.argsort(y)[::-1]  # highest y first
        top_idx = sorted_idx[:n_top]
        remaining_idx = sorted_idx[n_top:]
        rng = np.random.default_rng()
        random_idx = rng.choice(remaining_idx, size=min(n_random, len(remaining_idx)),
                                replace=False)
        subset_idx = np.sort(np.concatenate([top_idx, random_idx]))
        X_sub = X[subset_idx]
        y_sub = y[subset_idx]
        noise_sub = None
        if noise_level is not None:
            if np.iterable(noise_level):
                noise_sub = np.asarray(noise_level)[subset_idx]
            else:
                noise_sub = noise_level
        return X_sub, y_sub, noise_sub

    @property
    def stats(self):
        """Return a summary dict of scheduling statistics."""
        return {
            "n_full_fits": self._n_full_fits,
            "n_total_fits": self._n_total_fits,
            "n_skipped": self._n_skipped,
            "n_lml_recorded": len(self._lml_history),
            "current_restarts": self._compute_restarts(),
            "lml_history": list(self._lml_history[-10:]),
        }
