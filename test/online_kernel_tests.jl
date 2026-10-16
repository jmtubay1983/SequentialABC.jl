using Test
using SequentialABC
using Distributions
using Random
using Statistics
using LinearAlgebra

const _I2 = Matrix{Float64}(I, 2, 2)

@testset "online covariance estimator" begin
    @testset "matches the batch estimator" begin
        rng = MersenneTwister(2027)
        Σtrue = [1.0 0.6; 0.6 2.0]
        X = permutedims(rand(rng, MvNormal([1.0, -2.0], Σtrue), 5_000))
        w = abs.(randn(rng, 5_000)) .+ 0.1

        for (samples, weights) in ((X, nothing), (X, w))
            est = OnlineCovEstimator(2)
            if weights === nothing
                for i in 1:size(samples, 1)
                    update!(est, samples[i, :])
                end
                @test est.W ≈ 5_000.0
            else
                for i in 1:size(samples, 1)
                    update!(est, samples[i, :], weights[i])
                end
                @test est.W ≈ sum(weights)
            end

            Σ_batch = estimate_covariance(samples, weights; shrink = 0.0)
            Σ_online = cov(est; shrink = 0.0)

            @test est.n == 5_000
            @test Σ_online ≈ Σ_batch rtol = 1e-6
            @test Σ_online ≈ permutedims(Σ_online)
        end

        # the same data in the vector-of-vectors layout
        V = [collect(X[i, :]) for i in 1:size(X, 1)]
        est = OnlineCovEstimator(2)
        for (x, wi) in zip(V, w)
            update!(est, x, wi)
        end
        @test cov(est; shrink = 0.0) ≈ estimate_covariance(V, w; shrink = 0.0) rtol = 1e-6
    end

    @testset "one dimension" begin
        rng = MersenneTwister(4)
        x = 5.0 .+ 2.0 .* randn(rng, 10_000)
        est = OnlineCovEstimator(1)
        for xi in x
            update!(est, [xi])
        end
        Σ = cov(est; shrink = 0.0)
        @test size(Σ) == (1, 1)
        @test Σ[1, 1] ≈ var(x; corrected = true) rtol = 1e-6
        @test est.μ[1] ≈ mean(x) rtol = 1e-9
    end

    @testset "order independence and copy semantics" begin
        rng = MersenneTwister(9)
        X = randn(rng, 500, 3)
        a = OnlineCovEstimator(3)
        b = OnlineCovEstimator(3)
        for i in 1:500
            update!(a, X[i, :])
        end
        for i in 500:-1:1
            update!(b, X[i, :])
        end
        @test cov(a) ≈ cov(b) rtol = 1e-8

        c = copy(a)
        update!(c, [10.0, 10.0, 10.0])
        @test c.n == a.n + 1
        @test a.n == 500                       # the original is untouched
        @test !(cov(c; shrink = 0.0) ≈ cov(a; shrink = 0.0))
    end

    @testset "shrinkage and the eigenvalue floor guarantee positive definiteness" begin
        # a single sample: no variance information at all
        est = OnlineCovEstimator(3)
        update!(est, [1.0, 2.0, 3.0])
        Σ = cov(est)
        @test size(Σ) == (3, 3)
        @test Σ ≈ Σ' && isposdef(Symmetric(Σ))
        @test minimum(eigvals(Symmetric(Σ))) ≥ est.shrink - 1e-12

        # every sample identical: raw variance is exactly zero
        est = OnlineCovEstimator(3)
        for _ in 1:50
            update!(est, [1.0, 1.0, 1.0])
        end
        @test LinearAlgebra.norm(SequentialABC._raw_cov(est)) == 0.0
        Σz = cov(est)
        @test isposdef(Symmetric(Σz))
        @test tr(Σz) > 0

        # rank deficient: 3-D samples confined to a line
        est = OnlineCovEstimator(3)
        for t in 1.0:50.0
            update!(est, [t, 2t, 3t])
        end
        Σl = cov(est; shrink = 0.0)             # even with no shrinkage
        @test isposdef(Symmetric(Σl))
        @test minimum(eigvals(Symmetric(Σl))) > 0

        # an empty estimator is still positive definite
        empty = cov(OnlineCovEstimator(2))
        @test isposdef(Symmetric(empty))
    end

    @testset "reset!" begin
        est = OnlineCovEstimator(2)
        for _ in 1:10
            update!(est, [1.0, 2.0])
        end
        @test est.n == 10
        @test reset!(est) === est
        @test est.n == 0
        @test est.W == 0.0 && est.W2 == 0.0
        @test all(==(0.0), est.μ)
        @test all(==(0.0), est.M2)
        @test SequentialABC._raw_cov(est) === nothing
    end

    @testset "validation" begin
        @test_throws ArgumentError OnlineCovEstimator(0)
        @test_throws ArgumentError OnlineCovEstimator(2; shrink = 1.5)

        est = OnlineCovEstimator(2)
        @test_throws DimensionMismatch update!(est, [1.0, 2.0, 3.0])
        @test_throws ArgumentError update!(est, [1.0, 2.0], -1.0)
        @test_throws ArgumentError update!(est, [1.0, 2.0], Inf)
        @test_throws ArgumentError cov(est; shrink = 2.0)

        # a zero weight contributes nothing
        @test update!(est, [1.0, 2.0], 0.0) === est
        @test est.n == 0 && est.W == 0.0

        # integer samples are accepted
        update!(est, [1, 2])
        @test est.n == 1
    end
end

@testset "AdaptiveKernel with the online estimator" begin
    rng = MersenneTwister(11)
    Σtrue = [0.25 0.0; 0.0 0.16]
    X = permutedims(rand(rng, MvNormal([1.0, 2.0], Σtrue), 2_000))
    Σ0 = [1.0 0.0; 0.0 1.0]

    @testset "streaming and batch updates agree" begin
        kb = AdaptiveKernel(Σ0; rule = :beaumont, shrink = 0.0)
        update!(kb, X, fill(1.0, size(X, 1)))
        @test kb.n_updates == 1
        @test kb.est.n == 0                    # the batch path clears the accumulator

        ks = AdaptiveKernel(Σ0; rule = :beaumont, shrink = 0.0)
        for x in eachrow(X)
            update!(ks, x)                     # single parameter vector ⇒ streaming
        end
        @test ks.est.n == 2_000
        @test ks.n_updates == 0                # nothing adopted until commit!
        commit!(ks)

        @test ks.n_updates == 1
        @test ks.est.n == 0                    # reset after committing
        @test cov(ks) ≈ cov(kb) rtol = 1e-6
        @test cov(ks) ≈ 2.0 .* estimate_covariance(X; shrink = 0.0) rtol = 1e-5
        @test cov(ks) ≈ 2.0 .* Σtrue rtol = 0.1
        @test isposdef(Symmetric(cov(ks)))
    end

    @testset "weights are honoured while streaming" begin
        w = abs.(randn(rng, size(X, 1))) .+ 0.05
        ks = AdaptiveKernel(Σ0; rule = :unit, shrink = 0.0)
        for i in 1:size(X, 1)
            update!(ks, X[i, :], w[i])
        end
        commit!(ks)
        @test cov(ks) ≈ estimate_covariance(X, w; shrink = 0.0) rtol = 1e-5
    end

    @testset "degenerate accumulations are ignored" begin
        k = AdaptiveKernel(Σ0)
        Σ_before = cov(k)
        for _ in 1:10
            update!(k, [1.0, 1.0])             # all particles identical
        end
        @test k.est.n == 10
        commit!(k)
        @test cov(k) == Σ_before
        @test k.n_updates == 0
        @test k.est.n == 0                     # accumulator cleared regardless

        # committing with nothing accumulated is a no-op
        k2 = AdaptiveKernel(Σ0)
        @test commit!(k2) === k2
        @test k2.n_updates == 0
        @test cov(k2) == Σ0

        # a single particle is not enough variance information either
        k3 = AdaptiveKernel(Σ0)
        update!(k3, [3.0, -1.0])
        commit!(k3)
        @test k3.n_updates == 0
        @test cov(k3) == Σ0
    end

    @testset "adopting an external estimator" begin
        est = OnlineCovEstimator(2)
        for x in eachrow(X)
            update!(est, x)
        end

        k = AdaptiveKernel(Σ0; rule = :unit, shrink = 0.0)
        update!(k, est)
        @test k.n_updates == 1
        @test cov(k) ≈ Σtrue rtol = 0.1
        @test est.n == 2_000                   # the caller's estimator is untouched
        @test k.est.n == 0

        @test_throws DimensionMismatch update!(k, OnlineCovEstimator(3))
    end

    @testset "batch vs streaming dispatch" begin
        k = AdaptiveKernel(Σ0)
        update!(k, [0.5, -0.5])                # Vector{Float64} ⇒ one particle
        @test k.est.n == 1
        @test k.n_updates == 0

        V = [randn(rng, 2) for _ in 1:200]     # Vector{Vector{Float64}} ⇒ batch
        update!(k, V, fill(1.0, 200))
        @test k.n_updates == 1
        @test k.est.n == 0
        @test cov(k) ≈ scaling_factor(2) .* estimate_covariance(V) rtol = 1e-6
    end
end
