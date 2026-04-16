"""
Experimental TuRBO (Trust Region Bayesian Optimization) support for GPry.

This module is not part of the conservative GPry baseline. It is kept for
opt-in experiments with local trust-region acquisition.

This module implements a trust-region approach where the GP acquisition is focused on a
local region around the best point. The trust region expands on consecutive successes
(new best points found) and shrinks on consecutive failures, restarting when the region
becomes too small.

Reference:
    Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization",
    NeurIPS 2019.

Usage:
    The :class:`TuRBOManager` is designed to be attached to a
    :class:`surrogate.SurrogateModel`. It provides ``get_trust_bounds()`` which returns
    bounds narrower than the prior, and ``update()`` which adjusts the trust region based
    on new evaluations.
"""

import numpy as np


class TrustRegion:
    """A single trust region for TuRBO.

    Parameters
    ----------
    center : array, shape (d,)
        Initial center of the trust region.
    L_init : float
        Initial side length of the trust region.
    bounds : array, shape (d, 2)
        Prior bounds (used for clipping).
    n_success : int
        Consecutive successes before expanding. Default: 3.
    n_failure : int or None
        Consecutive failures before shrinking. Default: max(4, d).
    d : int or None
        Dimensionality. Inferred from ``center`` if None.
    """

    def __init__(self, center, L_init, bounds, n_success=3, n_failure=None, d=None):
        self.center = np.copy(center)
        self.L = L_init
        self.L_init = L_init
        self.L_min = L_init * 0.5**7  # minimum before restart
        self.bounds = np.array(bounds)
        self.d = d or len(center)
        self.n_success = n_success
        self.n_failure = n_failure or max(4, self.d)
        self.success_count = 0
        self.failure_count = 0
        self.best_y = -np.inf
        self.n_restarts = 0

    @property
    def trust_bounds(self):
        """Compute trust region bounds, clipped to prior bounds."""
        half_L = self.L / 2.0
        lb = np.maximum(self.center - half_L, self.bounds[:, 0])
        ub = np.minimum(self.center + half_L, self.bounds[:, 1])
        return np.column_stack([lb, ub])

    def update(self, new_y, new_x):
        """Update trust region based on new evaluation result.

        Parameters
        ----------
        new_y : float
            Best y value from the new evaluations within this region.
        new_x : array, shape (d,)
            Corresponding x location.

        Returns
        -------
        action : str
            One of 'expand', 'shrink', 'restart', or 'none'.
        """
        if new_y > self.best_y:
            self.best_y = new_y
            self.center = np.copy(new_x)
            self.success_count += 1
            self.failure_count = 0
        else:
            self.failure_count += 1
            self.success_count = 0

        action = "none"
        if self.success_count >= self.n_success:
            self.L = min(self.L * 2.0, self._max_L())
            self.success_count = 0
            action = "expand"
        elif self.failure_count >= self.n_failure:
            self.L /= 2.0
            self.failure_count = 0
            if self.L < self.L_min:
                action = "restart"
            else:
                action = "shrink"
        return action

    def _max_L(self):
        """Maximum trust region side length (full prior range)."""
        return np.max(self.bounds[:, 1] - self.bounds[:, 0])

    def restart(self, new_center, new_best_y):
        """Restart the trust region at a new location."""
        self.center = np.copy(new_center)
        self.L = self.L_init
        self.success_count = 0
        self.failure_count = 0
        self.best_y = new_best_y
        self.n_restarts += 1


class TuRBOManager:
    """Manages TuRBO trust regions for the GPry surrogate model.

    This is designed to be used from the Runner's main loop. The Runner
    calls ``get_trust_bounds()`` before each acquisition step to get the
    current trust region bounds, and ``update()`` after each evaluation
    to update the trust region state.

    Parameters
    ----------
    bounds : array, shape (d, 2)
        Prior bounds.
    L_init : float or None
        Initial trust region side length. If None, uses 0.8 * max(prior range).
    n_regions : int
        Number of independent trust regions (TuRBO-m). Default: 1.
    n_success : int
        Consecutive successes before expanding. Default: 3.
    n_failure : int or None
        Consecutive failures before shrinking. Default: max(4, d).
    """

    def __init__(self, bounds, L_init=None, n_regions=1, n_success=3, n_failure=None):
        self.bounds = np.array(bounds)
        self.d = len(bounds)
        if L_init is None:
            L_init = 0.8 * np.max(self.bounds[:, 1] - self.bounds[:, 0])
        self.L_init = L_init
        self.n_regions = n_regions
        self.n_success = n_success
        self.n_failure = n_failure
        self.regions = []  # initialized when first training data available
        self._initialized = False

    def initialize(self, X, y):
        """Initialize trust regions from existing training data.

        Places the first region at the best point. Additional regions
        (for TuRBO-m) are placed at diverse high-y points.

        Parameters
        ----------
        X : array, shape (n, d)
            Training inputs.
        y : array, shape (n,)
            Training targets.
        """
        finite = np.isfinite(y)
        X_f, y_f = X[finite], y[finite]
        if len(y_f) == 0:
            return

        # Sort by y descending
        order = np.argsort(y_f)[::-1]

        self.regions = []
        used_centers = []

        for i in range(min(self.n_regions, len(y_f))):
            if i == 0:
                idx = order[0]
            else:
                # Find the highest-y point that is far from existing centers
                best_idx = None
                best_min_dist = -1
                for j in order:
                    x_cand = X_f[j]
                    min_dist = min(np.linalg.norm(x_cand - c) for c in used_centers)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = j
                idx = best_idx

            center = X_f[idx]
            region = TrustRegion(
                center=center,
                L_init=self.L_init,
                bounds=self.bounds,
                n_success=self.n_success,
                n_failure=self.n_failure,
                d=self.d,
            )
            region.best_y = y_f[idx]
            self.regions.append(region)
            used_centers.append(center)

        self._initialized = True

    def get_trust_bounds(self, region_idx=0):
        """Get the trust region bounds for acquisition.

        Returns the trust bounds clipped to the prior, or the full prior
        bounds if not yet initialized.

        Parameters
        ----------
        region_idx : int
            Index of the trust region to query. Default: 0.

        Returns
        -------
        bounds : array, shape (d, 2)
            Trust region bounds.
        """
        if not self._initialized or region_idx >= len(self.regions):
            return self.bounds
        return self.regions[region_idx].trust_bounds

    def update(self, new_X, new_y, all_X=None, all_y=None):
        """Update trust regions after new evaluations.

        Parameters
        ----------
        new_X : array, shape (n_new, d)
            Newly evaluated points.
        new_y : array, shape (n_new,)
            Corresponding y values.
        all_X : array, shape (n_total, d), optional
            Full training set X (for restart placement).
        all_y : array, shape (n_total,), optional
            Full training set y.

        Returns
        -------
        actions : list of str
            Action taken for each region ('expand', 'shrink', 'restart', 'none').
        """
        if not self._initialized:
            if all_X is not None and all_y is not None:
                self.initialize(all_X, all_y)
            return ["init"]

        actions = []
        for region in self.regions:
            # Find the best new point that falls within this region's bounds
            tb = region.trust_bounds
            in_region = np.all((new_X >= tb[:, 0]) & (new_X <= tb[:, 1]), axis=1)

            if np.any(in_region):
                best_in_region = np.argmax(np.where(in_region, new_y, -np.inf))
                action = region.update(new_y[best_in_region], new_X[best_in_region])
            else:
                # No points in this region -- count as failure
                action = region.update(-np.inf, region.center)

            if action == "restart" and all_X is not None and all_y is not None:
                # Restart at a random high-y point far from other regions
                self._restart_region(region, all_X, all_y)

            actions.append(action)

        return actions

    def _restart_region(self, region, all_X, all_y):
        """Restart a region at a diverse high-y point."""
        finite = np.isfinite(all_y)
        X_f, y_f = all_X[finite], all_y[finite]
        if len(y_f) == 0:
            return

        # Try to find a point far from other region centers
        other_centers = [r.center for r in self.regions if r is not region]

        if other_centers:
            order = np.argsort(y_f)[::-1]
            best_idx = order[0]
            best_score = -1
            for idx in order[: max(20, len(y_f) // 5)]:
                min_dist = min(np.linalg.norm(X_f[idx] - c) for c in other_centers)
                score = y_f[idx] + 0.1 * min_dist  # balance quality and diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            region.restart(X_f[best_idx], y_f[best_idx])
        else:
            best_idx = np.argmax(y_f)
            region.restart(X_f[best_idx], y_f[best_idx])

    @property
    def summary(self):
        """Human-readable summary of trust region states."""
        lines = []
        for i, r in enumerate(self.regions):
            lines.append(
                f"  TR{i}: center={np.array2string(r.center, precision=2)}, "
                f"L={r.L:.3f}, best_y={r.best_y:.2f}, "
                f"succ={r.success_count}, fail={r.failure_count}, "
                f"restarts={r.n_restarts}"
            )
        return "\n".join(lines)
