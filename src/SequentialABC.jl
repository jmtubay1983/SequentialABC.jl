module SequentialABC

using LinearAlgebra
using Random
using Distributions
using StatsBase
using Statistics
using LogExpFunctions

include("simulator.jl")
include("config.jl")
include("adaptive_distance.jl")
include("state.jl")
include("schedules.jl")
include("rng.jl")
include("kernels.jl")
include("kernels/AdaptiveKernel.jl")
# kernel tuning after the adaptive kernel: it reuses `scaling_factor`, `cov`
# and `_kernel_matrix` from the kernels above.
include("kernel_tuning/KernelTuning.jl")
include("weights.jl")
include("smc.jl")
include("checkpoint.jl")
# HPC helpers after checkpointing: `collect_checkpoints!` matches its naming.
include("hpc/launcher.jl")
include("engines/MCMC.jl")
include("engines/PMC.jl")
# summary types first: summary_utils' validate_summary_keys dispatches on AbstractSummary
include("summaries/Summaries.jl")
include("summaries/summary_utils.jl")
# distances after summaries: named summaries are compared via namedtuple_to_vector
include("distances/Distances.jl")
# recipes last: they compose schedules, kernels, summaries and distances
include("recipes/Recipes.jl")
include("postprocess/RegressionAdjustment.jl")
include("plotting/PlotsAPI.jl")

"""
    run_inference(engine::Symbol, cfg, s_obs; kwargs...)

Unified front door for the three inference engines. Keywords are forwarded
**unchanged** to the engine's own entrypoint, and the engine's own result type
comes back unwrapped — a named forwarder, not a new API level:

| `engine` | calls | result | keywords |
|---|---|---|---|
| `:smc`  | [`run_smc`](@ref)  | [`ABCSMCState`](@ref) | `N`, `T`, `schedule`, `ε0`, `M`, `base_seed`, `adaptive`, `checkpoint_dir` |
| `:mcmc` | [`run_mcmc`](@ref) | [`ABCMCMCChain`](@ref) | `n_samples`, `ε`, `thin`, `M`, `base_seed` |
| `:pmc`  | [`run_pmc`](@ref)  | [`ABCSMCState`](@ref) | `N`, `T`, `εs`, `schedule`, `kernel_rule`, `M`, `base_seed` |

The `adaptive` keyword (an [`AdaptiveDistanceConfig`](@ref) for adaptive
distance scaling) is forwarded by `:smc` exactly like any other keyword;
`:mcmc` and `:pmc` do not accept it in this release.

```julia
state = run_inference(:smc,  cfg, s_obs; N = 1000, T = 5, base_seed = 2026)
chain = run_inference(:mcmc, cfg, s_obs; n_samples = 2000, ε = 0.3, base_seed = 2026)
pmc   = run_inference(:pmc,  cfg, s_obs; N = 1000, T = 5,
                      εs = [4.0, 2.0, 1.0, 0.6, 0.4], base_seed = 2026)
```

Keywords are **not** inspected here: each engine validates its own, so
`run_inference(:mcmc, cfg, s_obs; N = 1000)` fails with the same error
`run_mcmc` would raise — ABC-MCMC takes `n_samples`, not `N`. An unsupported
symbol throws an `ArgumentError`.

Every call starts a **fresh** run. Resuming a checkpointed population is the
state-based drivers' job: take an [`ABCSMCState`](@ref) from
[`load_latest_state`](@ref) and continue with `run_smc!` / `run_pmc!`, which
work directly on the result this function returns.
"""
function run_inference(engine::Symbol, cfg, s_obs; kwargs...)
    if engine === :smc
        return run_smc(cfg, s_obs; kwargs...)
    elseif engine === :mcmc
        return run_mcmc(cfg, s_obs; kwargs...)
    elseif engine === :pmc
        return run_pmc(cfg, s_obs; kwargs...)
    else
        throw(ArgumentError("Unknown engine: $(engine). Supported: :smc, :mcmc, :pmc"))
    end
end

export ABCSimulator,
       AgentsSimulator,
       ContinuousSimulator,
       ABCSMCConfig,
       ABCSMCState,
       # Adaptive distance scaling — stable configuration types only.
       # AdaptiveDistanceState and the lifecycle helpers (adaptive_distance,
       # update_adaptive!, estimate_weights) are internal; state is inspected
       # read-only through `state.adaptive`.
       AbstractScaleStrategy,
       NoAdaptation,
       FixedScale,
       StandardDeviationScale,
       PreviousGenAccepted,
       UnitInit,
       NumericalPolicy,
       AdaptiveDistanceConfig,
       GaussianKernel,
       AdaptiveKernel,
       OnlineCovEstimator,
       update!,
       commit!,
       reset!,
       cov,
       estimate_covariance,
       adaptive_scale,
       scaling_factor,
       with_covariance,
       make_state,
       make_particle_seed,
       initialize!,
       run_generation!,
       run_smc,
       run_smc!,
       run_pmc,
       run_pmc!,
       run_inference,
       importance_weight,
       ess,
       systematic_resample,
       save_checkpoint,
       list_checkpoints,
       load_checkpoint,
       load_latest_state,
       SlurmTemplate,
       render_slurm_script,
       write_slurm_script,
       make_generation_job,
       collect_checkpoints!,
       simulate_slurm_submission_preview,
       auto_kernel_scale,
       shrinkage_covariance,
       bandwidth_from_cov,
       make_adaptive_kernel,
       epsilon_quantile,
       QuantileSchedule,
       FixedSchedule,
       run_mcmc,
       ABCMCMCChain,
       acceptance_rate,
       abc_mcmc_accept,
       mcmc_propose,
       mcmc_log_proposal,
       ols_adjust,
       local_linear_adjust,
       adjusted_posterior,
       RegressionAdjustment,
       plot_marginals,
       plot_pairs,
       plot_trace,
       plot_diagnostics,
       plot_ppc,
       PlotSpec,
       AbstractSummary,
       FunctionSummary,
       CompositeSummary,
       ScaledSummary,
       PCAProjection,
       summary_keys,
       compute_summary,
       compose,
       scale,
       pca_reduce,
       aggregate,
       flatten_namedtuple,
       namedtuple_to_vector,
       vector_to_namedtuple,
       validate_summary_keys,
       l2_distance,
       weighted_l2,
       mahalanobis_distance,
       make_mahalanobis,
       default_schedule,
       default_kernel_scaling,
       preset_summaries

end
