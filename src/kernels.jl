using LinearAlgebra
using Distributions

"""
    GaussianKernel

Perturbation kernel for Toni et al. (2009) ABC‑SMC:
θ_new = θ_old + ε,  ε ~ 𝒩(0, Σ)

Implements the exact log‑pdf and perturbation steps used in the paper.
"""
struct GaussianKernel end

"""
    (K::GaussianKernel)(θ_new, θ_old, Σ_inv, logdetΣ) -> Float64

Full log‑density of 𝒩(θ_old, Σ) at θ_new, including the normalizing constant:

log 𝒩(x | μ, Σ) = -½ (x-μ)' Σ⁻¹ (x-μ) - ½ (k log 2π + logdetΣ)

Numerically identical to `Distributions.logpdf(MvNormal(μ, Σ), x)` (unit-tested).
"""
function (K::GaussianKernel)(θ_new::AbstractVector, θ_old::AbstractVector,
                             Σ_inv::Matrix{Float64}, logdetΣ::Float64)
    diff = θ_new .- θ_old
    k = length(diff)
    return -0.5 * diff' * Σ_inv * diff - 0.5 * (k * log(2π) + logdetΣ)
end

"""
    perturb(K::GaussianKernel, θ_old::AbstractVector, Σ::Matrix{Float64}, rng::AbstractRNG) -> Vector{Float64}

Draw θ_new ~ 𝒩(θ_old, Σ) using Cholesky factorization for efficiency and stability:
θ_new = θ_old + L * z,  where LL' = Σ and z ~ 𝒩(0, I)
"""
function perturb(K::GaussianKernel, θ_old::AbstractVector, Σ::Matrix{Float64}, rng::AbstractRNG)
    L = cholesky(Σ).L
    z = randn(rng, length(θ_old))
    return θ_old .+ L * z
end