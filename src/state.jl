"""
    ABCSMCState{C,Tθ,Tsum}

Mutable sampler state for Toni et al. (2009) ABC‑SMC.

Fields:
  - cfg::C                   — ABCSMCConfig
  - θ::Vector{Tθ}            — particles at current generation
  - w::Vector{Float64}       — normalized importance weights
  - summaries::Vector{Tsum}  — cached summaries
  - ε::Float64               — current tolerance
  - gen::Int                 — current generation index
  - ess::Float64             — effective sample size
  - base_seed::Int           — base RNG seed for deterministic replication
  - n_attempts::Int          — accepted‑particle attempts this generation (diagnostic)
"""
mutable struct ABCSMCState{C,Tθ,Tsum}
    cfg::C
    θ::Vector{Tθ}
    w::Vector{Float64}
    summaries::Vector{Tsum}
    ε::Float64
    gen::Int
    ess::Float64
    base_seed::Int
    n_attempts::Int
end

"""
    make_state(cfg::ABCSMCConfig, ::Type{Tθ}, N::Int, ε0::Float64, base_seed::Int)

Construct an empty generation‑0 state. `Tθ` is the raw parameter type from the prior;
`Tsum` starts as `Any` and gets retyped in `initialize!` from the first simulation result.
"""
function make_state(cfg::ABCSMCConfig, ::Type{Tθ}, N::Int, ε0::Float64, base_seed::Int) where {Tθ}
    θ = Vector{Tθ}(undef, N)
    w = fill(1.0 / N, N)
    summaries = Vector{Any}(undef, N)
    return ABCSMCState{typeof(cfg), Tθ, Any}(cfg, θ, w, summaries, ε0, 0, Float64(N), base_seed, 0)
end