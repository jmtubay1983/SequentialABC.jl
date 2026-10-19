# Changelog

## v0.4.0 — 2026-10-19

Adaptive distance scaling for ABC-SMC, with checkpoint/restart support, hardened
schema validation, and expanded documentation and regression tests.

- Add adaptive distance scaling: per-key inverse-distance weights with three
  strategies (`NoAdaptation`, `FixedScale`, `StandardDeviationScale`), a single
  `NumericalPolicy` (floor, constant-summary policy, max weight), and the immutable
  `AdaptiveDistanceConfig` / runtime `AdaptiveDistanceState` split, enabled via the
  `adaptive` keyword of `run_smc` / `run_smc!` (`AdaptiveDistanceConfig(NoAdaptation())`
  normalizes to the plain non-adaptive path).
- Tolerance schedules now accept a distance vector: `QuantileSchedule` /
  `FixedSchedule` take `(state, t, distances)`, and in adaptive mode ε is derived
  from the preceding population's recomputed `epsilon_selection_distances`.
- Add checkpoint schema v2 carrying the five restart-critical adaptive runtime
  fields (key order, weights, `gen_applied`, epsilon-selection distances,
  ε); schema versioning rejects mismatched payloads, and cross-field validation
  requires `1 <= gen_applied <= state.gen`, `epsilon_selection_distances` length
  equal to the particle population, finite non-negative adaptive ε, and a
  `NamedTuple` record (non-record payloads raise a clear `ArgumentError`).
- Documentation: corrected `epsilon_quantile` monotonicity explanation (the
  `min(..., ε_cap)` guard is load-bearing in adaptive mode), composition with
  weighted base distances (`WeightedL2`, Mahalanobis), `run_inference` keyword
  tables, `mean_summary` M = 1 edge cases, N = 1 / zero-scale / floor / clamp /
  `max_weight` behavior, and regression adjustment dropping adaptive runtime state.
- Tests: adaptive checkpoint round-trip and uninterrupted-vs-resumed equivalence,
  malformed/missing-field checkpoint payloads, one-shot `run_smc` with adaptive,
  adaptive resume without re-supplied config (error path), N = 1 under the
  default constant-summary policy, and cross-field schema validation cases.
- Deferred: per-generation observed-summary scaling (Issue #42 item 10) remains a
  later performance optimization.

## Unreleased

## v0.3.0 — 2026-10-17

The Phase B helpers: offline Slurm job generation and checkpoint collection, plus
kernel bandwidth and scaling utilities.

- Add Slurm job templates and launcher helpers (`SlurmTemplate`, `render_slurm_script`, `write_slurm_script`, `make_generation_job`, `collect_checkpoints!`, `simulate_slurm_submission_preview`).
- Add kernel bandwidth and scaling helpers (`auto_kernel_scale`, `shrinkage_covariance`, `bandwidth_from_cov`, `make_adaptive_kernel`).

## v0.2.0 — 2026-10-17

The Phase A engine suite: three ABC engines behind one entry point, adaptive
kernels, named summaries and distances, and the recipes that collect the
recommended defaults.

- `run_inference`: one entry point for all three engines (`:smc`, `:pmc`, `:mcmc`).
- Add the ABC-MCMC engine (`run_mcmc`) and a one-call `run_smc` entry point mirroring it.
- Add the ABC-PMC engine (`run_pmc`) with per-generation adaptive perturbation kernels.
- Add adaptive multivariate kernels (`estimate_covariance`, `adaptive_scale`, `scaling_factor` with the `:rosenthal` and `:beaumont` rules) and the streaming `OnlineCovEstimator`.
- Add the composable summary statistics framework (`FunctionSummary`, `CompositeSummary`, `aggregate`, `compose`, `scale`, `pca_reduce`, `namedtuple_to_vector`).
- Add common distance functions: `l2_distance`, `weighted_l2`, `mahalanobis_distance` and the `make_mahalanobis` shrinkage estimator.
- Add recipes for common workflows: `default_schedule`, `default_kernel_scaling` and `preset_summaries`.
- Add the engine benchmark harness (`bench/compare_engines.jl`) and the plotting API with examples.
- Add regression adjustment postprocessing (`adjusted_posterior`).

## v0.1.0 — 2026-10-16

- Documentation and CI infrastructure updates (docs env, autodocs, CI workflows, .gitignore).
- Add MIT license.
- Add CITATION.cff and CITATION.md.
- Add reproducibility bundle in repro/ (run_all.sh, README).
- Add SECURITY.md and CODE_OF_CONDUCT.md.
