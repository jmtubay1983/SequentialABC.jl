using SequentialABC
using Test
using Distributions
using LinearAlgebra
using Random
using LogExpFunctions

@testset "Importance weights" begin
    prior(θ) = logpdf(Normal(0.0, 1.0), θ[1])
    kernel = GaussianKernel()
    Σ = [0.5;;]

    θ_prev = [[0.0], [1.0]]
    w_prev = [0.5, 0.5]

    cfg = ABCSMCConfig(prior, identity, identity, identity, kernel, Σ, nothing)

    w = importance_weight(cfg, [0.5], θ_prev, w_prev)
    @test w > 0
    @test isfinite(w)

    # Weight should be proportional to prior / kernel-mixture density
    # Recompute independently to cross-check
    denom = logsumexp([
        log(w_prev[j]) + kernel([0.5], θ_prev[j], inv(Σ), logdet(Σ))
        for j in eachindex(θ_prev)
    ])
    expected = exp(prior([0.5]) - denom)
    @test isapprox(w, expected; rtol=1e-12)
end

@testset "ESS" begin
    # Uniform weights → ESS = N
    @test ess([0.5, 0.5]) == 2.0
    @test ess(fill(0.25, 4)) == 4.0

    # Point mass → ESS = 1
    @test ess([1.0, 0.0]) == 1.0
    @test ess([1.0]) == 1.0

    # Intermediate
    w = [0.7, 0.2, 0.1]
    @test 1.0 < ess(w) < 3.0
    @test isapprox(ess(w), 1.0 / sum(w .^ 2))

    # ESS returns Float64
    @test ess([0.5, 0.5]) isa Float64
end
