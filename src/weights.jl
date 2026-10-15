using LogExpFunctions

"""
    importance_weight(cfg::ABCSMCConfig, θ_prop::AbstractVector,
                      θ_prev::Vector, w_prev::Vector{Float64}) -> Float64

Toni et al. (2009) importance weight:

w ∝ π(θ_prop) / Σ_j w_prev[j] · K(θ_prop | θ_prev[j], Σ)

Implemented in log‑space for numerical stability:
log w = log π(θ_prop) - logsumexp( log(w_prev[j]) + log K(θ_prop | θ_prev[j], Σ) )

The prior is a log‑density on the raw parameter space; the transform is
applied only at simulation time.
"""
function importance_weight(cfg::ABCSMCConfig, θ_prop::AbstractVector,
                           θ_prev::Vector, w_prev::Vector{Float64})
    logprior = cfg.prior(θ_prop)

    denom = logsumexp([
        log(w_prev[j]) + cfg.kernel(θ_prop, θ_prev[j], cfg.Σ_inv, cfg.logdetΣ)
        for j in eachindex(θ_prev)
    ])

    return exp(logprior - denom)
end

"""
    ess(w::AbstractVector{<:Real}) -> Float64

Effective sample size:
ESS = 1 / Σ_i w_i²

For N=1, ESS is defined as 1.0 (guard against division issues).
"""
function ess(w::AbstractVector{<:Real})
    return 1.0 / sum(w .^ 2)
end

"""
    systematic_resample(w::AbstractVector{<:Real}, rng::AbstractRNG) -> Vector{Int}

Low‑variance systematic resampling (Toni et al. 2009 use resampling when the
effective sample size falls too low; systematic resampling has the smallest
variance among standard schemes).

Algorithm:
  1. Draw u ~ U(0, 1/N)
  2. Place N equally spaced positions u + (i−1)/N on the cumulative weight staircase
  3. Return the bin indices those positions fall into

The RNG is passed explicitly — never the global RNG — so resampling is as
deterministic and replayable as every other stochastic step.
"""
function systematic_resample(w::AbstractVector{<:Real}, rng::AbstractRNG)
    N = length(w)
    u = rand(rng) / N
    positions = u .+ (0:N-1) ./ N

    cumw = cumsum(w)
    cumw[end] = 1.0  # guard against floating-point drift

    idx = Vector{Int}(undef, N)
    j = 1
    for i in 1:N
        while j < N && positions[i] > cumw[j]
            j += 1
        end
        idx[i] = j
    end
    return idx
end