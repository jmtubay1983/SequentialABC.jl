module SequentialABC

using LinearAlgebra
using Random
using Distributions
using StatsBase
using Statistics
using LogExpFunctions

include("simulator.jl")
include("config.jl")
include("state.jl")
include("schedules.jl")
include("rng.jl")
include("kernels.jl")
include("weights.jl")
include("smc.jl")
include("checkpoint.jl")
include("plotting/PlotsAPI.jl")

export ABCSimulator,
       AgentsSimulator,
       ContinuousSimulator,
       ABCSMCConfig,
       ABCSMCState,
       GaussianKernel,
       make_state,
       make_particle_seed,
       initialize!,
       run_generation!,
       run_smc!,
       importance_weight,
       ess,
       systematic_resample,
       save_checkpoint,
       list_checkpoints,
       load_checkpoint,
       load_latest_state,
       epsilon_quantile,
       QuantileSchedule,
       FixedSchedule,
       plot_marginals,
       plot_pairs,
       plot_trace,
       plot_diagnostics,
       plot_ppc,
       PlotSpec

end
