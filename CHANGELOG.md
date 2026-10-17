# Changelog

## Unreleased

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
