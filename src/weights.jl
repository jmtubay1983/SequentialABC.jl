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