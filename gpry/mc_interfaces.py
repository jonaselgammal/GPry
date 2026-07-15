"""
Gradient-based (HMC / NUTS) acquisition sampling for GPry -- "Lever 1".

Monte-Carlo interfaces for the *sampling* step of the acquisition, seeded from
the GP training points, as an alternative to NORA's nested sampling (see
``GPry_NORA_hyperparam_handoff.md``, Lever 1). This module is the MC counterpart
of :mod:`gpry.ns_interfaces`. It provides two backends:

- a self-contained **numpy HMC** with reflective boundaries and an identity mass
  matrix (``hmc_sample_gp_mean`` / ``hmc_acquire``) -- no extra dependencies, but
  a weak sampler on anisotropic targets;
- **BlackJAX NUTS** with window adaptation (dual-averaging step size + diagonal
  mass matrix) on a JAX re-implementation of the GP mean (``nuts_sample_gp_mean``
  / ``nuts_acquire``). JAX/BlackJAX are imported lazily, so plain GPry imports do
  not pull them in.

Rationale
---------
For a stiff, high-d target NORA's nested sampler must fill and compress the
whole prior volume, spending many surrogate (GP) evaluations. HMC instead
follows the typical set using the *analytic* gradient of the GP mean
``grad_x mu(x) = grad_x k(x, X) . alpha`` and can reach comparable acquisition
quality with far fewer GP evaluations -- in the within-mode, gradient-guided
regime. HMC does NOT hop between separated modes, so chains are **seeded from
the GP's training points**, which already sit inside the discovered modes; this
makes the sampler mode-complete over *discovered* modes (it does not discover
new ones -- keep a separate discovery mechanism and NS for the final evidence).

Coordinate conventions
-----------------------
The sampler runs entirely in the GP's *transformed* (pre-processed) input
space, where ``kernel_``, ``X_train_`` and ``alpha_`` live. There the analytic
mean-gradient returned by
:meth:`gpry.gpr.GaussianProcessRegressor.predict_mean_grad_batch` needs no chain
rule, and the training points ``gpr.X_train_`` are natural seeds. Prior bounds
are transformed with ``preprocessing_X.transform_bounds`` and samples are mapped
back to the original parameter space with ``preprocessing_X.inverse_transform``.
For a bare (Runner-less) GPR the preprocessing is the identity, so transformed
space == original space.

The numpy HMC chains are fully vectorized: all walkers advance together as
``(n_chains, d)`` arrays, using the vectorized kernel gradient
(``kernel_.gradient_x_batch``), with reflective ("billiard") box boundaries.
The NUTS backend instead maps the box to R^d with a logit bijector (adding the
log-Jacobian to the target) and lets BlackJAX autodiff the JAX GP mean.
"""

import numpy as np


def _make_logp_grad(gpr, beta):
    """
    Build the (vectorized) log-target and its gradient in transformed space.

    The target matches NORA's nested-sampling target, i.e. the GP posterior
    mean used as a log-likelihood over a uniform box prior:
    ``log p(u) = beta * mu(u)`` (``beta`` acts as an inverse temperature; the
    default ``beta=1`` reproduces ``exp(mu)``). Points that the GP's SVM
    infinities-classifier flags as ``-inf`` get ``log p = -inf`` and zero
    gradient, so the reflective HMC simply rejects moves into them.
    """
    has_svm = getattr(gpr, "infinities_classifier", None) is not None

    def logp_grad(U):
        U = np.atleast_2d(U)
        mean, grad = gpr.predict_mean_grad_batch(U)
        logp = beta * mean
        grad = beta * grad
        if has_svm:
            try:
                finite = gpr.infinities_classifier.predict(
                    np.ascontiguousarray(U), validate=False
                )
                finite = np.asarray(finite, dtype=bool)
                if not np.all(finite):
                    logp = np.where(finite, logp, -np.inf)
                    grad = np.where(finite[:, None], grad, 0.0)
            except Exception:  # pragma: no cover - classifier is best-effort here
                pass
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return logp, grad

    return logp_grad


def _reflect(U, p, lo, hi):
    """
    Reflect positions that left the box ``[lo, hi]``, flipping the corresponding
    momentum components. The loop handles multiple reflections within one step.
    """
    for _ in range(8):
        below = U < lo
        above = U > hi
        out = below | above
        if not out.any():
            break
        U = np.where(below, 2 * lo - U, U)
        U = np.where(above, 2 * hi - U, U)
        p = np.where(out, -p, p)
    # Final safety clamp against pathological overshoots.
    U = np.clip(U, lo, hi)
    return U, p


def _leapfrog(U, p, grad, logp_grad, eps, n_steps, lo, hi):
    """One reflective leapfrog trajectory of ``n_steps`` steps (vectorized)."""
    p = p + 0.5 * eps * grad
    logp = None
    for s in range(n_steps):
        U = U + eps * p
        U, p = _reflect(U, p, lo, hi)
        logp, grad = logp_grad(U)
        if s < n_steps - 1:
            p = p + eps * grad
    p = p + 0.5 * eps * grad
    return U, p, logp, grad


def _hmc_step(U, logp, grad, logp_grad, eps, n_steps, lo, hi, rng):
    """One HMC transition for every chain; returns updated state and accepts."""
    m = U.shape[0]
    p0 = rng.standard_normal(size=U.shape)
    U_new, p_new, logp_new, grad_new = _leapfrog(
        U, p0.copy(), grad, logp_grad, eps, n_steps, lo, hi
    )
    H0 = -logp + 0.5 * np.sum(p0 ** 2, axis=1)
    H_new = -logp_new + 0.5 * np.sum(p_new ** 2, axis=1)
    with np.errstate(invalid="ignore"):
        log_accept = H0 - H_new
    finite = np.isfinite(logp_new) & np.isfinite(H_new)
    accept = np.log(rng.uniform(size=m)) < log_accept
    accept &= finite
    a = accept[:, None]
    U_out = np.where(a, U_new, U)
    logp_out = np.where(accept, logp_new, logp)
    grad_out = np.where(a, grad_new, grad)
    return U_out, logp_out, grad_out, accept


def hmc_sample_gp_mean(
    gpr,
    lo,
    hi,
    seeds=None,
    rng=None,
    max_chains=256,
    n_warmup=50,
    n_samples=100,
    n_leapfrog=15,
    step_size=None,
    beta=1.0,
    thin=1,
    target_accept=0.7,
):
    """
    Run training-point-seeded reflective HMC on the GP mean, in transformed space.

    Parameters
    ----------
    gpr : GaussianProcessRegressor
        A *fitted* GPry GP. Its ``X_train_`` (transformed space) are the default
        chain seeds.
    lo, hi : array, shape=(d,)
        Box bounds in the *transformed* input space.
    seeds : array, shape=(n_chains, d), optional
        Explicit chain seeds (transformed space). Defaults to ``gpr.X_train_``,
        subsampled to at most ``max_chains``.
    max_chains : int
        Cap on the number of parallel chains (= number of seeds used).
    n_warmup, n_samples : int
        Number of warmup (step-size adaptation) and sampling iterations.
    n_leapfrog : int
        Leapfrog steps per HMC iteration.
    step_size : float, optional
        Initial leapfrog step size. Defaults to a heuristic based on the box
        size and dimension; adapted during warmup toward ``target_accept``.
    beta : float
        Inverse temperature on the GP mean (``beta=1`` -> target ``exp(mu)``).
    thin : int
        Keep one sample per chain every ``thin`` sampling iterations.
    target_accept : float
        Target Metropolis acceptance for step-size adaptation.

    Returns
    -------
    dict with keys
        ``X`` : (n_kept, d) accepted positions (transformed space),
        ``logp`` : (n_kept,) their ``beta * mu`` values,
        ``n_eval`` : total GP evaluations consumed by this call,
        ``n_eval_sampling`` : GP evaluations during the sampling phase only,
        ``accept_rate`` : mean acceptance during sampling,
        ``step_size`` : adapted step size,
        ``n_chains`` : number of chains.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = lo.shape[0]

    if seeds is None:
        if not hasattr(gpr, "X_train_"):
            raise ValueError("GPR is not fitted: no training points to seed HMC.")
        seeds = np.asarray(gpr.X_train_, dtype=float)
    seeds = np.atleast_2d(np.asarray(seeds, dtype=float))
    if seeds.shape[0] > max_chains:
        idx = rng.choice(seeds.shape[0], size=max_chains, replace=False)
        seeds = seeds[idx]
    # Keep seeds strictly inside the box.
    U = np.clip(seeds, lo, hi)
    n_chains = U.shape[0]

    if step_size is None:
        # Heuristic: a fraction of the box width, shrinking with dimension.
        step_size = 0.25 * float(np.mean(hi - lo)) / max(1.0, d ** 0.25)
    eps = float(step_size)

    logp_grad = _make_logp_grad(gpr, beta)
    logp, grad = logp_grad(U)

    n_eval_start = int(getattr(gpr, "n_eval", 0))

    # --- Warmup: adapt the step size toward target_accept (simple, robust) ---
    acc_ema = target_accept
    for _ in range(n_warmup):
        U, logp, grad, accept = _hmc_step(
            U, logp, grad, logp_grad, eps, n_leapfrog, lo, hi, rng
        )
        acc = float(np.mean(accept))
        acc_ema = 0.7 * acc_ema + 0.3 * acc
        if acc_ema > target_accept + 0.05:
            eps *= 1.1     # accepting too much -> bolder steps
        elif acc_ema < target_accept - 0.05:
            eps /= 1.1     # rejecting too much -> smaller steps
        eps = float(np.clip(eps, 1e-6, float(np.mean(hi - lo))))

    # --- Sampling ---
    n_eval_sampling_start = int(getattr(gpr, "n_eval", 0))
    kept = []
    kept_logp = []
    accepts = []
    for t in range(n_samples):
        U, logp, grad, accept = _hmc_step(
            U, logp, grad, logp_grad, eps, n_leapfrog, lo, hi, rng
        )
        accepts.append(float(np.mean(accept)))
        if (t % thin) == 0:
            kept.append(U.copy())
            kept_logp.append(logp.copy())

    X = np.concatenate(kept, axis=0) if kept else np.empty((0, d))
    logp_out = np.concatenate(kept_logp, axis=0) if kept_logp else np.empty((0,))
    # Chain-structured trace (n_kept, n_chains, d) for ESS / diagnostics.
    trace = np.stack(kept, axis=0) if kept else np.empty((0, n_chains, d))
    n_eval_end = int(getattr(gpr, "n_eval", 0))
    return {
        "X": X,
        "logp": logp_out,
        "trace": trace,
        "n_eval": n_eval_end - n_eval_start,
        "n_eval_sampling": n_eval_end - n_eval_sampling_start,
        "accept_rate": float(np.mean(accepts)) if accepts else 0.0,
        "step_size": eps,
        "n_chains": n_chains,
    }


def hmc_acquire(gpr, bounds, rng=None, return_info=False, **hmc_kwargs):
    """
    NORA-compatible driver: seeded HMC on the GP mean over an original-space box.

    Transforms ``bounds`` into the GP's transformed space, runs
    :func:`hmc_sample_gp_mean`, and maps the accepted samples back to the
    original parameter space.

    Parameters
    ----------
    gpr : GaussianProcessRegressor
        A fitted GPry GP.
    bounds : array, shape=(d, 2)
        Box bounds in the *original* parameter space.
    rng : numpy Generator, optional
    return_info : bool
        If True, also return the raw diagnostics dict from
        :func:`hmc_sample_gp_mean`.
    **hmc_kwargs
        Forwarded to :func:`hmc_sample_gp_mean`.

    Returns
    -------
    X : (n, d) samples in the original parameter space.
    y : None
        Placeholder (kept ``None`` to mirror the NS samplers, which recompute y
        downstream). Use ``info['logp']`` for the sampled ``beta * mu`` values.
    sigma_y : None
    w : (n,) uniform weights (HMC samples are equally weighted).
    (info : dict, only if ``return_info``.)
    """
    bounds = np.asarray(bounds, dtype=float)
    tb = np.asarray(gpr.preprocessing_X.transform_bounds(bounds), dtype=float)
    lo, hi = tb[:, 0], tb[:, 1]
    # transform_bounds may flip lo/hi if the transform reverses an axis.
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    info = hmc_sample_gp_mean(gpr, lo, hi, rng=rng, **hmc_kwargs)
    X_norm = info["X"]
    X = gpr.preprocessing_X.inverse_transform(X_norm) if len(X_norm) else X_norm
    w = np.ones(X.shape[0]) if len(X) else np.empty((0,))
    if return_info:
        return X, None, None, w, info
    return X, None, None, w


# =========================================================================== #
# BlackJAX NUTS backend
# =========================================================================== #
def _extract_stationary_kernel(kernel):
    """
    Pull ``(amplitude, length_scale, family, nu)`` out of GPry's default kernel
    ``ConstantKernel * (RBF|Matern)`` (possibly nested products). ``amplitude``
    is the product of the constant factors; the predictive mean ignores any
    additive WhiteKernel (zero off the training set).
    """
    from gpry.kernels import RBF, Matern, ConstantKernel, WhiteKernel

    amp = [1.0]
    stat = [None]

    def walk(k):
        if isinstance(k, ConstantKernel):
            amp[0] *= float(k.constant_value)
        elif isinstance(k, WhiteKernel):
            pass  # zero contribution to k(x*, X_train) for x* not in the set
        elif isinstance(k, (RBF, Matern)):
            stat[0] = k
        elif hasattr(k, "k1") and hasattr(k, "k2"):
            walk(k.k1)
            walk(k.k2)
        elif hasattr(k, "length_scale"):
            stat[0] = k
        else:
            raise ValueError(f"Unsupported kernel factor for NUTS: {k!r}")

    walk(kernel)
    if stat[0] is None:
        raise ValueError(f"Could not extract a stationary kernel from {kernel!r}")
    ell = np.atleast_1d(np.asarray(stat[0].length_scale, dtype=float))
    if isinstance(stat[0], Matern):
        return amp[0], ell, "matern", float(stat[0].nu)
    return amp[0], ell, "rbf", None


def _build_jax_logdensity(gpr, lo, hi, beta):
    """
    Build a JAX log-density on the unconstrained space u in R^d, where the
    normalized input is ``x = lo + (hi - lo) * sigmoid(u)`` (logit bijector), so
    NUTS is unconstrained yet samples never leave the box. The target matches
    NORA's NS target (``beta * mu`` on a uniform-box prior) plus the bijector's
    log-Jacobian. Returns ``(logdensity_fn, x_of_u, u_of_x)``.
    """
    import jax
    import jax.numpy as jnp

    amp, ell, family, nu = _extract_stationary_kernel(gpr.kernel_)
    # y-normalization scale (Dummy -> 1.0, Normalize_y -> std_y). The additive
    # offset is an irrelevant constant for sampling, so only the scale is needed.
    std_y = float(np.ravel(gpr.preprocessing_y.inverse_transform_scale(np.ones(1)))[0])

    ell_j = jnp.asarray(ell)
    Xtr_j = jnp.asarray(np.asarray(gpr.X_train_, dtype=float))
    alpha_j = jnp.asarray(np.ravel(np.asarray(gpr.alpha_, dtype=float)))
    lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi)
    width = hi_j - lo_j

    def k_vec(xn):
        diff = (xn - Xtr_j) / ell_j            # (n_train, d)
        d2 = jnp.sum(diff ** 2, axis=1)        # (n_train,)
        if family == "rbf":
            return amp * jnp.exp(-0.5 * d2)
        dist = jnp.sqrt(jnp.clip(d2, 1e-30, None))
        if nu == 0.5:
            kk = jnp.exp(-dist)
        elif nu == 1.5:
            a = jnp.sqrt(3.0) * dist
            kk = (1.0 + a) * jnp.exp(-a)
        elif nu == 2.5:
            a = jnp.sqrt(5.0) * dist
            kk = (1.0 + a + (5.0 / 3.0) * d2) * jnp.exp(-a)
        else:
            raise ValueError(f"Matern nu={nu} not supported in the JAX backend.")
        return amp * kk

    def gp_mean_norm(xn):
        return jnp.dot(k_vec(xn), alpha_j)

    def x_of_u(u):
        return lo_j + width * jax.nn.sigmoid(u)

    def logdensity_fn(u):
        xn = x_of_u(u)
        log_jac = jnp.sum(jnp.log(width) + jax.nn.log_sigmoid(u)
                          + jax.nn.log_sigmoid(-u))
        return beta * std_y * gp_mean_norm(xn) + log_jac

    def u_of_x(xn):
        # inverse bijector, clipped away from the boundaries to keep u finite
        s = np.clip((np.asarray(xn) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        return np.log(s) - np.log1p(-s)

    return logdensity_fn, x_of_u, u_of_x


_NUTS_RUNNER = None


def _get_nuts_runner():
    """
    Build (once) and return the module-level JIT-compiled multi-chain NUTS
    runner. Reusing the SAME function object lets JAX cache the compiled
    executable across acquisition calls: it is re-used whenever the static args
    (``n_warmup``, ``n_samples``, kernel ``family``/``nu``, ``target_accept``)
    and the input SHAPES (padded training capacity, ``n_chains``, ``d``) match.

    All values that change between iterations (``X_train``, ``alpha``, length
    scales, amplitude, bounds, seeds, key) flow in as *arguments*, so only their
    values change -- not the traced computation -- and no recompile is
    triggered. This removes the ~1.7 s/call build+trace overhead that dominated
    the previous version (which rebuilt the whole BlackJAX pipeline every call).
    """
    global _NUTS_RUNNER
    if _NUTS_RUNNER is not None:
        return _NUTS_RUNNER
    import jax
    import jax.numpy as jnp
    import blackjax
    from functools import partial

    @partial(jax.jit, static_argnames=("n_warmup", "n_samples", "family", "nu",
                                       "target_accept", "max_num_doublings"))
    def _run(Xpad, alpha_pad, ell, amp, std_y, beta, lo, hi, u0, key,
             n_warmup, n_samples, family, nu, target_accept, max_num_doublings):
        width = hi - lo

        def k_vec(xn):
            diff = (xn - Xpad) / ell                # (capacity, d)
            d2 = jnp.sum(diff ** 2, axis=1)         # (capacity,)
            if family == "rbf":
                return amp * jnp.exp(-0.5 * d2)
            dist = jnp.sqrt(jnp.clip(d2, 1e-30, None))
            if nu == 0.5:
                kk = jnp.exp(-dist)
            elif nu == 1.5:
                a = jnp.sqrt(3.0) * dist
                kk = (1.0 + a) * jnp.exp(-a)
            elif nu == 2.5:
                a = jnp.sqrt(5.0) * dist
                kk = (1.0 + a + (5.0 / 3.0) * d2) * jnp.exp(-a)
            else:
                raise ValueError(f"Matern nu={nu} not supported in the JAX backend.")
            return amp * kk

        def logdensity_fn(u):
            xn = lo + width * jax.nn.sigmoid(u)
            log_jac = jnp.sum(jnp.log(width) + jax.nn.log_sigmoid(u)
                              + jax.nn.log_sigmoid(-u))
            # padded rows have alpha_pad == 0, so they contribute nothing here
            return beta * std_y * jnp.dot(k_vec(xn), alpha_pad) + log_jac

        def run_chain(k, u_init):
            wkey, skey = jax.random.split(k)
            # NB: do NOT pass progress_bar here -- it defaults to False and some
            # BlackJAX versions don't accept it on window_adaptation (they then
            # forward it to the NUTS kernel, which raises). Keep this call to the
            # arguments that are stable across versions.
            warmup = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn, is_mass_matrix_diagonal=True,
                target_acceptance_rate=target_accept,
                max_num_doublings=max_num_doublings)
            (state, params), _ = warmup.run(wkey, u_init, num_steps=n_warmup)
            # params already carries max_num_doublings (an extra_parameter of the
            # adaptation), so it does not need to be passed again here.
            step = blackjax.nuts(logdensity_fn, **params).step

            def one(st, kk):
                st, info = step(kk, st)
                return st, (st.position, info.num_integration_steps,
                            info.acceptance_rate, info.is_divergent)

            skeys = jax.random.split(skey, n_samples)
            _, out = jax.lax.scan(one, state, skeys)
            return out  # (pos, nsteps, acc, divs), each leading axis n_samples

        keys = jax.random.split(key, u0.shape[0])
        return jax.vmap(run_chain)(keys, u0)

    _NUTS_RUNNER = _run
    return _run


def nuts_sample_gp_mean(
    gpr,
    lo,
    hi,
    seeds=None,
    rng=None,
    max_chains=64,
    n_warmup=200,
    n_samples=200,
    beta=1.0,
    target_accept=0.8,
    thin=1,
    pad_multiple=64,
    max_num_doublings=10,
):
    """
    BlackJAX NUTS on the GP mean, seeded from training points, in normalized
    input space. Each chain runs its own window adaptation (dual-averaging step
    size + diagonal mass matrix), which is exactly what the naive identity-mass
    HMC lacks.

    Uses the persistent, JIT-compiled runner from :func:`_get_nuts_runner`. To
    keep the traced shapes stable across acquisition calls (so JAX compiles
    once and re-uses the executable), the training set is padded to a fixed
    capacity (``ceil(n_train / pad_multiple) * pad_multiple``; padded rows get
    ``alpha = 0`` and so contribute nothing to the GP mean), and exactly
    ``n_chains = max_chains`` chains are used (seeds sampled with replacement
    when there are fewer training points). A recompile happens only when the
    capacity tier changes (every ``pad_multiple`` new points).

    Returns a dict analogous to :func:`hmc_sample_gp_mean`, with ``X`` and
    ``trace`` (normalized space), ``n_eval`` (total leapfrog/gradient evals of
    the GP mean, warmup + sampling), ``n_eval_sampling``, ``accept_rate``,
    ``n_chains``, ``divergences`` and the padded ``capacity``.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng() if rng is None else rng
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = lo.shape[0]

    if not hasattr(gpr, "X_train_"):
        raise ValueError("GPR is not fitted: no training points to seed NUTS.")
    Xtr = np.asarray(gpr.X_train_, dtype=float)
    alpha = np.ravel(np.asarray(gpr.alpha_, dtype=float))
    n_train = Xtr.shape[0]

    # Chains: fixed at max_chains for shape stability (sample seeds with
    # replacement when there are fewer training points than chains).
    seeds_all = Xtr if seeds is None else np.atleast_2d(np.asarray(seeds, dtype=float))
    n_seed = seeds_all.shape[0]
    n_chains = int(max_chains)
    idx = rng.choice(n_seed, size=n_chains, replace=(n_seed < n_chains))
    seeds_sel = np.clip(seeds_all[idx], lo, hi)

    # Pad training set to a fixed capacity (padded rows get alpha = 0).
    capacity = max(int(pad_multiple),
                   int(np.ceil(n_train / pad_multiple) * pad_multiple))
    Xpad = np.zeros((capacity, d))
    Xpad[:n_train] = Xtr
    alpha_pad = np.zeros(capacity)
    alpha_pad[:n_train] = alpha

    amp, ell, family, nu = _extract_stationary_kernel(gpr.kernel_)
    std_y = float(np.ravel(gpr.preprocessing_y.inverse_transform_scale(np.ones(1)))[0])

    # Seeds -> unconstrained u-space (logit bijector).
    s = np.clip((seeds_sel - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    u0 = np.log(s) - np.log1p(-s)

    runner = _get_nuts_runner()
    key = jax.random.PRNGKey(int(rng.integers(2 ** 31 - 1)))
    pos, nsteps, acc, divs = runner(
        jnp.asarray(Xpad), jnp.asarray(alpha_pad), jnp.asarray(ell),
        jnp.asarray(float(amp)), jnp.asarray(std_y), jnp.asarray(float(beta)),
        jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(u0), key,
        n_warmup=int(n_warmup), n_samples=int(n_samples),
        family=family, nu=nu, target_accept=float(target_accept),
        max_num_doublings=int(max_num_doublings),
    )

    # u-space -> normalized x-space (sigmoid), then reshape.
    pos = np.asarray(pos)                                   # (n_chains, n_samples, d)
    pos_x = lo + (hi - lo) / (1.0 + np.exp(-pos))
    nsteps = np.asarray(nsteps)                             # (n_chains, n_samples)
    mean_tree = float(np.mean(nsteps))
    n_eval_sampling = int(nsteps.sum() + nsteps.size)
    n_eval_warmup = int(round(n_warmup * n_chains * (mean_tree + 1)))

    trace = np.swapaxes(pos_x, 0, 1)                        # (n_samples, n_chains, d)
    if thin > 1:
        trace = trace[::thin]
    X = trace.reshape(-1, d)
    return {
        "X": X,
        "trace": trace,
        "n_eval": n_eval_warmup + n_eval_sampling,
        "n_eval_sampling": n_eval_sampling,
        "n_eval_warmup": n_eval_warmup,
        "accept_rate": float(np.mean(acc)),
        "divergences": int(np.sum(np.asarray(divs))),
        "n_chains": n_chains,
        "mean_tree_size": mean_tree,
        "capacity": capacity,
    }


def nuts_acquire(gpr, bounds, rng=None, return_info=False, **nuts_kwargs):
    """
    NORA-compatible driver for BlackJAX NUTS (mirrors :func:`hmc_acquire`).
    Returns ``(X, None, None, w)`` in the original parameter space, plus the
    diagnostics dict if ``return_info``.
    """
    bounds = np.asarray(bounds, dtype=float)
    tb = np.asarray(gpr.preprocessing_X.transform_bounds(bounds), dtype=float)
    lo, hi = np.minimum(tb[:, 0], tb[:, 1]), np.maximum(tb[:, 0], tb[:, 1])
    info = nuts_sample_gp_mean(gpr, lo, hi, rng=rng, **nuts_kwargs)
    X_norm = info["X"]
    X = gpr.preprocessing_X.inverse_transform(X_norm) if len(X_norm) else X_norm
    w = np.ones(X.shape[0]) if len(X) else np.empty((0,))
    if return_info:
        return X, None, None, w, info
    return X, None, None, w
