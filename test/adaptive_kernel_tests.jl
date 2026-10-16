using Test
using SequentialABC
using Distributions
using Random
using Statistics
using LinearAlgebra

const _I2 = Matrix{Float64}(I, 2, 2)

@testset "adaptive kernels" begin
    @testset "scaling rules" begin
        @test scaling_factor(1; rule = :rosenthal) ≈ 2.38^2
        @test scaling_factor(4; rule = :rosenthal) ≈ 2.38^2 / 4
        @test scaling_factor(4; rule = :gelman) ≈ 2.38^2 / 4
        @test scaling_factor(3; rule = :beaumont) == 2.0
        @test scaling_factor(3; rule = :abc) == 2.0
        @test scaling_factor(3; rule = :unit) == 1.0
        @test scaling_factor(3; rule = :none) == 1.0
        @test scaling_factor(3; rule = 0.75) == 0.75
        @test_throws ArgumentError scaling_factor(0)
        @test_throws ArgumentError scaling_factor(2; rule = :nope)
    end

    @testset "adaptive_scale" begin
        Σ = [2.0 0.5; 0.5 1.0]
        @test adaptive_scale(Σ, 2; rule = :unit) ≈ Σ
        @test adaptive_scale(Σ, 2; rule = :rosenthal) ≈ (2.38^2 / 2) .* Σ
        @test adaptive_scale(Σ, 2; rule = :beaumont) ≈ 2.0 .* Σ
        @test adaptive_scale(Σ; rule = :unit) ≈ Σ          # d defaults to size(Σ, 1)
        @test_throws ArgumentError adaptive_scale([1.0 2.0 3.0], 3)
        @test_throws DimensionMismatch adaptive_scale(Σ, 3)
    end

    @testset "estimate_covariance recovers a known covariance" begin
        rng = MersenneTwister(2026)
        Σtrue = [1.0 0.6; 0.6 2.0]
        X = permutedims(rand(rng, MvNormal(zeros(2), Σtrue), 200_000))

        Σhat = estimate_covariance(X)
        @test size(Σhat) == (2, 2)
        @test Σhat ≈ Σtrue rtol = 0.05
        @test Σhat ≈ permutedims(Σhat)
        @test isposdef(Symmetric(Σhat))
    end

    @testset "estimate_covariance: weights, layouts, validation" begin
        rng = MersenneTwister(7)
        X = vcat(randn(rng, 1000, 2) .+ 5.0, randn(rng, 1000, 2))
        w = [ones(1000); zeros(1000)]

        Σw = estimate_covariance(X, w)
        @test Σw ≈ _I2 rtol = 0.2            # only the first cluster: ~N(0, I)

        # vector-of-vectors layout (exactly `ABCSMCState.θ`)
        V = [collect(X[i, :]) for i in 1:size(X, 1)]
        @test estimate_covariance(V, w) ≈ Σw

        # uniform weights ≡ no weights
        @test estimate_covariance(X) ≈ estimate_covariance(X, fill(1.0, 2000))

        @test_throws DimensionMismatch estimate_covariance(X, ones(10))
        @test_throws ArgumentError estimate_covariance(X, -ones(2000))
        @test_throws ArgumentError estimate_covariance(X, zeros(2000))
        @test_throws ArgumentError estimate_covariance(X; shrink = 2.0)
        @test_throws ArgumentError estimate_covariance(Vector{Vector{Float64}}())
    end

    @testset "regularisation keeps estimates positive definite" begin
        # rank deficient: 50 points on a line in ℝ³ (N ≫ d but rank 1)
        t = collect(1.0:50.0)
        X = hcat(t, 2 .* t, 3 .* t)
        Σ = estimate_covariance(X)
        @test size(Σ) == (3, 3)
        @test isposdef(Symmetric(Σ))
        @test minimum(eigvals(Symmetric(Σ))) > 0

        # every particle identical ⇒ shrink · I
        @test estimate_covariance(ones(10, 2); shrink = 1e-3) ≈ 1e-3 .* _I2

        # a single sample does not throw
        @test estimate_covariance(reshape([1.0, 2.0], 1, 2)) ≈ 1e-6 .* _I2

        # stronger shrinkage pulls a rank-deficient estimate towards isotropy
        Σ_upsilon = estimate_covariance(X; shrink = 0.5)
        @test isposdef(Symmetric(Σ_upsilon))
        @test tr(Σ_upsilon) ≈ tr(Σ) rtol = 1e-6
    end

    @testset "perturb preserves the mean in expectation" begin
        Σ = [0.04 0.01; 0.01 0.09]
        k = AdaptiveKernel(Σ; rule = :unit)
        θ0 = [1.0, -2.0]

        # deterministic for a given seed
        @test SequentialABC.perturb(k, θ0, MersenneTwister(3)) ==
              SequentialABC.perturb(k, θ0, MersenneTwister(3))

        rng = MersenneTwister(11)
        N = 100_000
        P = Matrix{Float64}(undef, N, 2)
        for i in 1:N
            P[i, :] = SequentialABC.perturb(k, θ0, rng)
        end

        @test vec(mean(P, dims = 1)) ≈ θ0 atol = 0.02
        @test estimate_covariance(P) ≈ Σ rtol = 0.05
    end

    @testset "kernel log density" begin
        Σ = [0.04 0.01; 0.01 0.09]
        k = AdaptiveKernel(Σ; rule = :unit)
        a, b = [0.3, 0.7], [-0.2, 0.1]

        @test k(a, b) ≈ logpdf(MvNormal(b, Σ), a)
        @test k(a, b) ≈ k(b, a)             # symmetric random walk
        @test k(a, b) ≈ -0.5 * (a - b)' * inv(Σ) * (a - b) -
                        0.5 * (2 * log(2π) + logdet(Σ))
    end

    @testset "construction validation" begin
        @test_throws ArgumentError AdaptiveKernel([1.0 2.0; 2.0 1.0])   # not PD
        @test_throws ArgumentError AdaptiveKernel([1.0 0.0 0.0])        # not square
        @test_throws ArgumentError AdaptiveKernel([1.0;;]; shrink = -0.1)

        k = AdaptiveKernel([2.0;;])
        @test k.Σ == [2.0;;]
        @test k.Σ_inv ≈ [0.5;;]
        @test k.logdetΣ ≈ log(2.0)
        @test k.n_updates == 0
        @test k.rule === :rosenthal

        k2 = AdaptiveKernel([2.0;;]; rule = 0.25)
        @test k2.rule == 0.25
    end

    @testset "update! re-estimates the covariance" begin
        rng = MersenneTwister(5)
        Σtrue = [0.25 0.0; 0.0 0.16]
        X = permutedims(rand(rng, MvNormal([1.0, 2.0], Σtrue), 20_000))

        k = AdaptiveKernel([1.0 0.0; 0.0 1.0]; rule = :unit)
        @test update!(k, X) === k
        @test k.n_updates == 1
        @test k.Σ ≈ Σtrue rtol = 0.1
        @test k.Σ_inv ≈ inv(k.Σ)
        @test k.logdetΣ ≈ logdet(k.Σ)

        # the scaling rule is applied to the estimate
        k2 = AdaptiveKernel([1.0 0.0; 0.0 1.0]; rule = :beaumont)
        update!(k2, X)
        @test k2.Σ ≈ 2.0 .* estimate_covariance(X)

        # a degenerate population is ignored, not adopted
        Σ_before = copy(k.Σ)
        update!(k, ones(10, 2))
        @test k.Σ == Σ_before
        @test k.n_updates == 1

        @test_throws DimensionMismatch update!(k, ones(5, 3))
    end

    @testset "engine integration: with_covariance and an adaptive SMC loop" begin
        prior(θ) = logpdf(Normal(0.0, 2.0), θ[1]) + logpdf(Normal(0.0, 2.0), θ[2])
        sample_prior(rng) = 2.0 .* randn(rng, 2)
        sim = ContinuousSimulator(model_builder = θ -> θ,
                                  evolve = (θ, rng) -> sum(θ) + 0.3 * randn(rng),
                                  summary = identity)
        cfg = ABCSMCConfig(prior, sample_prior, identity, (s, so) -> abs(s - so),
                           GaussianKernel(), [0.5 0.0; 0.0 0.5], sim)

        cfg2 = with_covariance(cfg, [0.1 0.0; 0.0 0.2])
        @test cfg2.Σ == [0.1 0.0; 0.0 0.2]
        @test cfg2.Σ_inv ≈ inv(cfg2.Σ)
        @test cfg2.logdetΣ ≈ logdet(cfg2.Σ)
        @test cfg2.prior === cfg.prior            # everything else is carried over
        @test cfg2.simulator === cfg.simulator
        @test cfg2.kernel === cfg.kernel

        k = AdaptiveKernel([0.5 0.0; 0.0 0.5]; rule = :beaumont)
        @test with_covariance(cfg, k).Σ == k.Σ

        # a tiny adaptive SMC run: adapt after every generation
        N = 50
        state = make_state(cfg, Vector{Float64}, N, 4.0, 42)
        initialize!(state, 1.0)
        for _ in 2:4
            update!(k, state.θ, state.w)
            state.cfg = with_covariance(state.cfg, k)
            run_generation!(state, 1.0; ε = epsilon_quantile(state, 0.6))
            @test state.ess >= 1.0
            @test state.ess <= N + 1e-9
        end

        @test k.n_updates == 3
        @test state.gen == 4
        @test all(isfinite, state.dists)
        @test all(p -> all(isfinite, p), state.θ)
        @test size(k.Σ) == (2, 2)
        @test isposdef(Symmetric(k.Σ))
    end

    @testset "sampler consistency on a Gaussian model" begin
        # s = θ + N(0, 0.5²) with a N(0, 2²) prior. The ABC posterior is centred
        # on the *observed summary*, so a run at a tight tolerance must land
        # there; if the perturbation/weighting were wrong it would not.
        prior(θ) = logpdf(Normal(0.0, 2.0), θ[1]) + logpdf(Normal(0.0, 2.0), θ[2])
        sample_prior(rng) = 2.0 .* randn(rng, 2)
        sim = ContinuousSimulator(model_builder = θ -> θ,
                                  evolve = (θ, rng) -> θ .+ 0.5 .* randn(rng, 2),
                                  summary = identity)
        s_obs = [0.8, -0.4]
        cfg = ABCSMCConfig(prior, sample_prior, identity,
                           (s, so) -> sqrt(sum(abs2, s .- so)),
                           GaussianKernel(), [0.25 0.0; 0.0 0.25], sim)

        kernel = AdaptiveKernel([0.25 0.0; 0.0 0.25]; rule = :beaumont)
        state = make_state(cfg, Vector{Float64}, 200, 1.0, 7)
        initialize!(state, s_obs)
        for _ in 2:5
            update!(kernel, state.θ, state.w)
            state.cfg = with_covariance(state.cfg, kernel)
            run_generation!(state, s_obs; ε = epsilon_quantile(state, 0.5))
        end

        m = [sum(state.w .* [p[i] for p in state.θ]) for i in 1:2]
        @test m ≈ s_obs atol = 0.2
        @test state.ε < 1.0
    end
end
