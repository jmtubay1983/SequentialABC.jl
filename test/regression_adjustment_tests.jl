using Test
using SequentialABC
using Random
using Statistics
using LinearAlgebra

@testset "Regression adjustment" begin
    rng = MersenneTwister(2024)
    N = 400

    @testset "ols_adjust recovers known linear coefficients" begin
        s = randn(rng, N)
        a, b = 1.5, -2.0
        S = reshape(s, :, 1)

        # deterministic  θ = a + b s  ⇒  the adjustment returns exactly a
        θ_exact = a .+ b .* s
        adj = ols_adjust(reshape(θ_exact, :, 1), S, [0.0])
        @test adj isa Matrix{Float64}
        @test size(adj) == (N, 1)
        @test all(isfinite, adj)
        @test maximum(abs.(adj .- a)) < 1e-8

        # the fitted slope b shows up as a shift when s_obs ≠ 0
        adj_shift = ols_adjust(reshape(θ_exact, :, 1), S, [1.0])
        @test maximum(abs.(adj_shift .- (a + b))) < 1e-8

        # noisy  θ = a + b s + ε  ⇒  residual dependence on s disappears
        θ_noisy = a .+ b .* s .+ 0.1 .* randn(rng, N)
        adjn = ols_adjust(reshape(θ_noisy, :, 1), S, [0.0])
        @test all(isfinite, adjn)
        @test abs(cor(vec(adjn), s)) < 1e-8
        @test isapprox(mean(adjn), a; atol = 0.05)

        # coercion: vector-of-vector samples, vector summaries, 2-D summaries
        @test size(ols_adjust([reshape(θ_exact, :, 1)[i, :] for i in 1:N], vec(s), 0.0)) == (N, 1)
        @test size(ols_adjust(reshape(θ_exact, :, 1), hcat(s, randn(rng, N)), [0.0, 0.0])) == (N, 1)
    end

    @testset "local_linear_adjust" begin
        s = randn(rng, N)
        S = reshape(s, :, 1)
        Θ = reshape(1.0 .+ 2.0 .* s .+ 0.05 .* randn(rng, N), :, 1)

        for kernel in (:epanechnikov, :gaussian, :uniform)
            adj = local_linear_adjust(Θ, S, [0.0]; kernel = kernel)
            @test size(adj) == (N, 1)
            @test all(isfinite, adj)
        end
        @test all(isfinite, local_linear_adjust(Θ, S, [0.0]; kernel = :gaussian, h = 0.25))
        @test_throws ArgumentError local_linear_adjust(Θ, S, [0.0]; kernel = :bogus)
        @test_throws ArgumentError local_linear_adjust(Θ, S, [0.0]; h = -1.0)
    end

    @testset "input validation" begin
        Θ = randn(rng, 12, 2)
        @test_throws DimensionMismatch ols_adjust(Θ, randn(rng, 11, 1), [0.0])
        @test_throws DimensionMismatch ols_adjust(Θ, randn(rng, 12, 1), [0.0, 0.0])
        @test_throws DimensionMismatch ols_adjust(Θ, randn(rng, 12, 1), [0.0]; w = ones(3))
        @test_throws ArgumentError adjusted_posterior((θ = [[1.0]],); observed_summary = 0.0)
    end

    @testset "adjusted_posterior returns finite values of the right shape" begin
        d = 2
        Θ = randn(rng, N, d)
        s = randn(rng, N)
        st = ABCSMCState(nothing, [Θ[i, :] for i in 1:N], fill(1 / N, N),
                         collect(s), abs.(s), 0.5, 1, Float64(N), 7, 0, 0)

        for method in (:ols, :local_linear)
            adj = adjusted_posterior(st; summaries_fn = x -> [x],
                                     observed_summary = 0.0, method = method)
            @test adj isa ABCSMCState
            @test length(adj.θ) == N
            @test all(p -> length(p) == d && all(isfinite, p), adj.θ)
            @test adj.w == st.w
            @test adj.ε == st.ε && adj.gen == st.gen
        end

        # type-unstable summaries (Vector{Any}), as produced by make_state states
        st_any = ABCSMCState(nothing, [Θ[i, :] for i in 1:N], fill(1 / N, N),
                             Any[Float64(x) for x in s], abs.(s), 0.5, 1, Float64(N), 7, 0, 0)
        adj_any = adjusted_posterior(st_any; summaries_fn = x -> [x], observed_summary = 0.0)
        @test adj_any isa ABCSMCState
        @test all(p -> length(p) == d && all(isfinite, p), adj_any.θ)

        # a NamedTuple result is rebuilt in kind, and the default summaries_fn works
        nt = (θ = [Θ[i, :] for i in 1:N], summaries = collect(s), w = fill(1 / N, N))
        adj_nt = adjusted_posterior(nt; observed_summary = 0.0)
        @test adj_nt isa NamedTuple
        @test all(p -> length(p) == d && all(isfinite, p), adj_nt.θ)
        @test adj_nt.summaries === nt.summaries

        @test_throws ArgumentError adjusted_posterior(nt; observed_summary = 0.0, method = :bogus)
        @test_throws ArgumentError adjusted_posterior(nt; method = :ols)   # no observed_summary
        @test_throws ArgumentError adjusted_posterior(nt; summaries_fn = x -> [x],
                                                     observed_summary = 0.0, method = :ols, h = 0.5)
    end

    @testset "array-level adjusted_posterior" begin
        s = randn(rng, N)
        S = reshape(s, :, 1)
        Θ = reshape(1.0 .+ 2.0 .* s .+ 0.05 .* randn(rng, N), :, 1)

        r = adjusted_posterior(Θ, S, [0.0]; method = :ols)
        @test r isa RegressionAdjustment
        @test r.method == :ols
        @test r.bandwidth === nothing
        @test size(r.θ) == (N, 1)
        @test size(r.slope) == (1, 1)
        @test all(isfinite, r.θ)
        @test isapprox(sum(r.weights), 1.0)
        @test r.θ ≈ ols_adjust(Θ, S, [0.0])

        r2 = adjusted_posterior(Θ, S, [0.0]; method = :local_linear, kernel = :gaussian)
        @test r2.method == :local_linear
        @test r2.kernel == :gaussian
        @test r2.bandwidth isa Float64 && r2.bandwidth > 0
        @test r2.θ ≈ local_linear_adjust(Θ, S, [0.0]; kernel = :gaussian)
        @test_throws ArgumentError adjusted_posterior(Θ, S, [0.0]; method = :bogus)
    end
end
