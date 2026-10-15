module SequentialABC

using LinearAlgebra
using Random
using Distributions
using StatsBase
using LogExpFunctions

include("simulator.jl")
include("config.jl")
include("state.jl")
include("rng.jl")
include("kernels.jl")
include("weights.jl")
include("smc.jl")

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
       systematic_resample

end
