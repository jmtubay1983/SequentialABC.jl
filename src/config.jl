using Distributions
using LogExpFunctions

"""
    ABCSMCConfig{P,Tr,D,K,S}

Configuration struct for the Toni et al. (2009) ABC‑SMC sampler.

Fields:
  - prior::P              — log‑density function π(θ_raw) on the raw parameter space
  - sample_prior::Function — sampler: (rng) → θ_raw from the prior (for gen‑1 init)
  - transform::Tr         — θ_raw → θ_model (project‑specific parameter map)
  - distance::D           — (summary_sim, summary_obs) → Real
  - kernel::K             — perturbation kernel: log K(θ_new | θ_old, Σ)
  - Σ::Matrix{Float64}    — perturbation covariance
  - Σ_inv::Matrix{Float64} — precomputed inverse covariance
  - simulator::S          — simulator backend (AgentsSimulator or ContinuousSimulator)

Constructor computes Σ_inv automatically from Σ.
"""
struct ABCSMCConfig{P,Tr,D,K,S}
    prior::P
    sample_prior::Function
    transform::Tr
    distance::D
    kernel::K
    Σ::Matrix{Float64}
    Σ_inv::Matrix{Float64}
    logdetΣ::Float64
    simulator::S
end

function ABCSMCConfig(prior, sample_prior, transform, distance, kernel, Σ, simulator)
    C = convert(Matrix{Float64}, Σ)
    return ABCSMCConfig{typeof(prior),typeof(transform),typeof(distance),typeof(kernel),typeof(simulator)}(
        prior, sample_prior, transform, distance, kernel, C, inv(C), logdet(C), simulator
    )
end