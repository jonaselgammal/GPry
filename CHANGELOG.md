# 4.0 – 2026-xx-yy (unreleased)

- API changes for the surrogate model: now a `surrogate.SurrogateModel` super-object manages both a `gpr.GaussianProcessRegressor` and an `infinities_classifier.InfinitiesClassifier` (also the preprocessors and clipping). The infinities classifier itself has been revamped: SVM and trust bounds check are now work at the same level.
- NORA promoted to default acquisition engine.
- Log-evidence can now be retrieved from the last MC sampler run if a nested sampler is used (default). NB: the associated uncertainty is that of the nested sampler integration, not including surrogate modelling errors.
- Added `run.Runner` kwarg `mc` to pass options for the final and diagnosis MC samplers.
- Kernel white noise can now easily be varied (as opposed to a fixed `alpha` diagonal term) with kwargs, without needing to define a custom kernel (still fixed by default).
- Documentation has been reworked, separating explanations and module documentation.
- [Cobaya](https://cobaya.readthedocs.io) wrapper updated, and better defaults added.
- Improved [UltraNest](https://johannesbuchner.github.io/UltraNest/) interface defaults: switched to slice sampler to mimic [PolyChord](https://github.com/PolyChord/PolyChordLite) (more stable).

# 3.0 – 2025-03-30

- In general implements necessary changes for inference runs in [arXiv:2503.21871](https://arxiv.org/abs/2503.21871).
- Moved to `pyproject.toml`.
- Improved documentation.
- MPI dependence made optional.
- Hard [Cobaya](https://cobaya.readthedocs.io) dependency dropped.
- Convergence:
  - Implemented convergence policies: `n` (necessary) and `s` (sufficient).
  - Reworked `GaussianKL` criterion and created `GaussianKLTrain` criterion for alignment between current MS sample and training set.
- Acquisition engines:
  - `bounds` can now be passed to `multi_add()` method to restrict candidates search.
  - Interfaces to [UltraNest](https://johannesbuchner.github.io/UltraNest/) and [nessai](https://nessai.readthedocs.io) added to NORA.
- Trust region prior restriction added to `GaussianProcessRegressor`.
- The `SVM` is now passed pre-processed points and thresholds, which simplifies its code.
- Changed treatment of true model: created `Truth` class that abstracts Cobaya dependence.
- Added interfaces for [UltraNest](https://johannesbuchner.github.io/UltraNest/) and [nessai](https://nessai.readthedocs.io) for NORA. Both of them and [PolyChord](https://github.com/PolyChord/PolyChordLite) can now also be used for MC runs of the surrogate posterior.
- Created plots for logp and parameter traces and surrogate model parameter slices.
- Changes to `Runner`:
  - Can specify a dict of options for `plots` kwarg.
  - If `NORA` is used, `GaussianKL` and `TrainAlignment` are added as default convergence criteria.
  - Added methods to interface methods of `Truth` instance attribute and the `GaussianProcessRegressor` (e.g. `logp_truth()` vs `logp()`).
  - Add more robust diagnosis at convergence; avoids almost 100% of overshooting cases.
  - Adds `set_fiducial_point` and `set_fiducial_MC` for plots.
  - Nested Sampling is now the default MC sampler for diagnosis and final MC run.
- Improved interface for [Cobaya](https://cobaya.readthedocs.io).

# 2.0 – 2023-09-14

- Added the NORA `GPAcquisition` engine, as described in [arXiv:2305.19267](https://arxiv.org/abs/2305.19267), using [PolyChord](https://github.com/PolyChord/PolyChordLite).
- Custom `hyperparameter_bounds` can be passed to the `GaussianProcessRegressor.fit()` method.
- Performance improvements in `GaussianProcessRegressor.fit()` and `SVM.predict()`.
- Reworked input for the `Runner`, and added diagnosis method for overshoots.
- Early interface for running within [Cobaya](https://cobaya.readthedocs.io).

# 1.1 – 2022-11-04

- Catches up with the published paper: [arXiv:2211.02045](https://arxiv.org/abs/2211.02045).
- Relicensed from `MIT` to `LGPL`.
- Improved documentation.
- Updated default `z_scaling` to `0.85`, following [arXiv:2211.02045](https://arxiv.org/abs/2211.02045).
- Updated default noise level of `GaussianProcessRegressor` to `1e-2`.
- Renamed some `GaussianProcessRegressor` methods to retrieve the number of evaluations, and added methods to retrieve only the last-iteration ones.
- Added dimensionality scaling to the rel and abs differences in `CorrectCounter`, following [arXiv:2211.02045](https://arxiv.org/abs/2211.02045).
- Created `ReferenceProposer` and `PriorProposer` for initialisation and sequential acquisition.
- Changes to `Runner`:
  - It can now take a callable log-likelihood as `model`, together with prior`bounds`, instead of `cobaya.Model`.
  - It can now take an `initial_proposer`, a random `seed`, and a boolean `plots` kwarg.
  - Reworked `options`: some renamed, and can now take `zeta_scaling`.
  - More verbose logging by default.
- Standardized some class names (e.g. `GP_Aquisition` to `GPAcquisition`).
- Added more tests.

# 1.0 – 2022-06-23

Early implementation of the functionalities described in the [first GPry paper](https://arxiv.org/abs/2211.02045).
