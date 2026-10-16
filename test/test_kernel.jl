using SequentialABC
using Test
using Distributions
using LinearAlgebra

@testset "Gaussian kernel" begin
    K = GaussianKernel()

    # 1-D case
    Σ1 = [0.5;;]
    Σ1_inv = inv(Σ1)
    ld1 = logdet(Σ1)
    θ_new = [1.2]
    θ_old = [1.0]
    ours = K(θ_new, θ_old, Σ1_inv, ld1)
    ref = logpdf(MvNormal(θ_old, Σ1), θ_new)
    @test isapprox(ours, ref; atol=1e-12)

    # 2-D case (correlated)
    Σ2 = [0.8 0.1; 0.1 0.6]
    Σ2_inv = inv(Σ2)
    ld2 = logdet(Σ2)
    for (x, μ) in [([1.2, -0.4], [0.7, 0.3]), ([0.0, 0.0], [0.0, 0.0])]
        ours = K(Vector{Float64}(x), Vector{Float64}(μ), Σ2_inv, ld2)
        ref = logpdf(MvNormal(Vector{Float64}(μ), Σ2), Vector{Float64}(x))
        @test isapprox(ours, ref; atol=1e-12)
    end

    # Kernel type is a singleton
    @test K isa GaussianKernel
end
