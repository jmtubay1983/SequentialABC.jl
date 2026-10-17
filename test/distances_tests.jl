using SequentialABC
using Test
using Distributions
using Random
using Statistics
using LinearAlgebra

# ── tiny deterministic reference problem (same 2-D family as the other suites) ─
# θ ~ N(0, 2²I), s(θ) = θ + N(0, I). The `_dis` suffix keeps this file's globals
# apart from the other suites sharing one Main namespace.
_prior_dis(θ) = sum(logpdf(Normal(0.0, 2.0), x) for x in θ)
_sample_prior_dis(rng) = 2.0 .* randn(rng, 2)
_simulator_dis = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = identity,
)
# The ad-hoc Euclidean one-liner the new `l2_distance` replaces.
_distance_ref(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
CFG_L2 = ABCSMCConfig(_prior_dis, _sample_prior_dis, identity, l2_distance,
                      GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_dis)
CFG_REF = ABCSMCConfig(_prior_dis, _sample_prior_dis, identity, _distance_ref,
                       GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_dis)
OBS_DIS = [1.0, -1.0]
SEED_DIS = 20260118

@testset "distance functions" begin

    # ── l2_distance ──────────────────────────────────────────────────────────
    @testset "l2_distance: scalars, arrays, named summaries" begin
        @test l2_distance(3.0, 1.5) == 1.5
        @test l2_distance(2, 5) == 3.0                    # Int inputs promote
        @test l2_distance(5.0, 2.0) == l2_distance(2.0, 5.0)

        @test l2_distance([1.0, 2.0], [4.0, 6.0]) == 5.0
        @test l2_distance([0, 0], [3, 4]) == 5.0
        @test l2_distance([1.0, 2.0], [1.0, 2.0]) == 0.0
        @test l2_distance(Float32[0, 0], Float32[3, 4]) == 5.0
        # matrices: the Frobenius norm of the difference
        @test l2_distance(zeros(2, 2), Matrix{Float64}(I, 2, 2)) ≈ sqrt(2)
        @test_throws DimensionMismatch l2_distance([1.0], [1.0, 2.0])
        @test_throws DimensionMismatch l2_distance([1.0 2.0], [1.0, 2.0])

        # named summaries compare by key, so declaration order is irrelevant
        a = (count = 4.0, mean_age = 3.75)
        b = (count = 2.0, mean_age = 4.75)
        @test l2_distance(a, b) ≈ sqrt(5)
        @test l2_distance(a, b) ≈ _distance_ref([4.0, 3.75], [2.0, 4.75])
        @test l2_distance(a, NamedTuple{(:mean_age, :count)}((4.75, 2.0))) ≈ sqrt(5)
        @test l2_distance(a, a) == 0.0

        err = try
            l2_distance((count = 1.0,), (n = 1.0,))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("count", sprint(showerror, err))
        @test occursin("n", sprint(showerror, err))

        # a non-Real named value has no single numeric distance; key is named
        err2 = try
            l2_distance((a = [1.0],), (a = [1.0],))
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("`a`", sprint(showerror, err2))

        # mixed forms are rejected, never silently coerced
        @test_throws MethodError l2_distance((a = 1.0,), [1.0])
        @test_throws MethodError l2_distance([1.0], 1.0)
    end

    # ── weighted_l2 ──────────────────────────────────────────────────────────
    @testset "weighted_l2: weights, named weights, factories" begin
        @test weighted_l2([1.0, 1.0], [0.0, 0.0], [4.0, 9.0]) == sqrt(13)
        @test weighted_l2([1.0, 100.0], [0.0, 0.0], [1.0, 0.0]) == 1.0  # muted
        @test weighted_l2([2.0], [0.0], [9.0]) == 6.0
        @test weighted_l2(3.0, 1.0, 4.0) == 4.0                          # scalar form
        @test_throws ArgumentError weighted_l2(3.0, 1.0, -1.0)
        @test_throws ArgumentError weighted_l2(3.0, 1.0, Inf)

        @test_throws DimensionMismatch weighted_l2([1.0], [1.0, 2.0], [1.0, 1.0])
        @test_throws DimensionMismatch weighted_l2([1.0, 2.0], [1.0, 2.0], [1.0])
        @test_throws ArgumentError weighted_l2([1.0, 2.0], [1.0, 2.0], [-1.0, 1.0])
        @test_throws ArgumentError weighted_l2([1.0, 2.0], [1.0, 2.0], [0.0, 0.0])
        @test_throws ArgumentError weighted_l2([1.0, 2.0], [1.0, 2.0], [Inf, 1.0])

        # named weights: unlisted keys keep weight 1, unknown keys are rejected
        a = (count = 2000.0, mean_age = 30.0)
        b = (count = 1900.0, mean_age = 31.0)
        @test weighted_l2(a, b, (count = 1e-3,)) ≈ sqrt(11)
        @test weighted_l2(a, b, (count = 1e-3, mean_age = 1.0)) ≈ sqrt(11)
        @test weighted_l2(a, b, [1e-3, 1.0]) ≈ sqrt(11)
        err = try
            weighted_l2(a, b, (cnt = 1e-3,))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("cnt", sprint(showerror, err))       # names the typo
        @test occursin("count", sprint(showerror, err))     # and the valid keys
        @test_throws DimensionMismatch weighted_l2(a, b, [1.0])

        # named values and named weights keep agreeing when pairs are reordered
        b2 = NamedTuple{(:mean_age, :count)}((31.0, 1900.0))
        @test weighted_l2(a, b2, (count = 1e-3,)) ≈ sqrt(11)

        # the factory validates once and returns a callable
        d = weighted_l2([16.0, 9.0])
        @test d isa SequentialABC.WeightedL2
        @test d([1.0, 1.0], [0.0, 0.0]) == 5.0
        @test d.weights == [16.0, 9.0]
        @test occursin("WeightedL2", sprint(show, d))

        # … and a callable config distance is what the engines want
        dn = weighted_l2((count = 1e-3,))
        @test dn(a, b2) ≈ sqrt(11)
        @test occursin("WeightedL2", sprint(show, dn))
        # a key muted to zero weight contributes nothing; unlisted keys stay at 1
        dz = weighted_l2((count = 0.0,))
        @test dz((count = 1.0, mean_age = 10.0), (count = 5.0, mean_age = 13.0)) == 3.0

        # construction-time errors
        @test_throws ArgumentError weighted_l2([1.0, -1.0])
        @test_throws ArgumentError weighted_l2([0.0, 0.0])
        @test_throws ArgumentError weighted_l2(Float64[])
        @test_throws ArgumentError weighted_l2((;))
        @test_throws ArgumentError weighted_l2((a = -1.0,))
        @test_throws ArgumentError weighted_l2((a = [1.0],))
        # an all-zero final weight vector is caught at call time
        @test_throws ArgumentError weighted_l2((a = 0.0,))((a = 1.0,), (a = 0.0,))

        # mutating the source vector afterwards cannot change the metric
        w = [16.0, 9.0]
        d2 = weighted_l2(w)
        w[1] = 0.0
        @test d2([1.0, 1.0], [0.0, 0.0]) == 5.0
    end

    # ── mahalanobis_distance ─────────────────────────────────────────────────
    @testset "mahalanobis_distance: explicit covariance" begin
        # identity covariance is Euclidean
        @test mahalanobis_distance([1.0, 2.0], [4.0, 6.0], Matrix{Float64}(I, 2, 2)) == 5.0
        @test mahalanobis_distance([1.0, 2.0], [4.0, 6.0], Matrix{Float64}(I, 2, 2)) ==
              l2_distance([1.0, 2.0], [4.0, 6.0])
        # diagonal covariance matches weighted_l2 with w = 1 ./ diag
        Σd = [4.0 0.0; 0.0 9.0]
        @test mahalanobis_distance([1.0, 1.0], [0.0, 0.0], Σd) ≈
              weighted_l2([1.0, 1.0], [0.0, 0.0], [1 / 4, 1 / 9])
        # hand-computed 2×2: x' Σ⁻¹ x = 4/7
        Σ = [4.0 1.0; 1.0 2.0]
        @test mahalanobis_distance([1.0, 1.0], [0.0, 0.0], Σ) ≈ sqrt(4 / 7)
        @test mahalanobis_distance([1.0, 1.0], [0.0, 0.0], Σ) ==
              mahalanobis_distance([0.0, 0.0], [1.0, 1.0], Σ)
        # translation invariance
        @test mahalanobis_distance([11.0, 1.0], [10.0, 0.0], Σ) ≈ sqrt(4 / 7)
        # scalar variance form
        @test mahalanobis_distance(3.0, 1.0, 4.0) == 1.0
        @test_throws ArgumentError mahalanobis_distance(3.0, 1.0, 0.0)
        @test_throws ArgumentError mahalanobis_distance(3.0, 1.0, -1.0)

        # a Cholesky factor is accepted directly — the no-re-factorisation path
        F = cholesky(Symmetric(Σ))
        @test mahalanobis_distance([1.0, 1.0], [0.0, 0.0], F) ≈ sqrt(4 / 7)
        @test_throws DimensionMismatch mahalanobis_distance([1.0], [1.0], F)

        # named summaries align by key here too: (count, mean_age) difference is (1, 1)
        @test mahalanobis_distance((count = 1.0, mean_age = 1.0),
                                   NamedTuple{(:mean_age, :count)}((0.0, 0.0)),
                                   Σ) ≈ sqrt(4 / 7)
        err = try
            mahalanobis_distance((count = 1.0,), (n = 1.0,), Σ)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("count", sprint(showerror, err))

        # dimension and definiteness errors
        @test_throws DimensionMismatch mahalanobis_distance([1.0], [1.0], Σ)
        @test_throws DimensionMismatch mahalanobis_distance([1.0, 2.0], [1.0], Σ)
        @test_throws DimensionMismatch mahalanobis_distance([1.0, 2.0], [1.0, 2.0],
                                                            [1.0 0.0; 0.0 1.0; 0.0 0.0])
        err2 = try
            mahalanobis_distance([1.0, 1.0], [0.0, 0.0], [1.0 1.0; 1.0 1.0])
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("positive definite", sprint(showerror, err2))
        @test occursin("make_mahalanobis", sprint(showerror, err2))
    end

    # ── make_mahalanobis ─────────────────────────────────────────────────────
    @testset "make_mahalanobis: pilot estimation and shrinkage" begin
        rng = MersenneTwister(4242)
        X = randn(rng, 200, 3)
        X[:, 3] = X[:, 1] .+ 0.1 .* randn(rng, 200)     # correlated columns
        C = cov(X; dims = 1)

        d0 = make_mahalanobis(X; shrinkage = 0.0)
        @test d0.shrinkage == 0.0
        @test d0.n == 200
        @test size(d0.Σ) == (3, 3)
        @test d0.Σ ≈ C                                   # λ = 0 is the raw sample covariance
        @test d0.Σ ≈ d0.Σ'                               # symmetric
        @test d0.chol isa Cholesky

        d1 = make_mahalanobis(X; shrinkage = 1.0)
        @test d1.Σ ≈ Matrix(Diagonal(diag(C)))           # λ = 1 ignores correlations
        @test d1([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) ≈
              weighted_l2([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 1 ./ diag(C))

        dm = make_mahalanobis(X; shrinkage = 0.25)
        @test dm.Σ ≈ 0.75 .* C .+ 0.25 .* Diagonal(diag(C))

        # the default is nonzero shrinkage; λ = 0 recovers the explicit formula
        ddef = make_mahalanobis(X)
        @test ddef.shrinkage == 0.1
        @test ddef.Σ ≈ 0.9 .* C .+ 0.1 .* Diagonal(diag(C))
        @test d0([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) ≈
              mahalanobis_distance([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], C)

        # a callable distance on named summaries, and its length guard
        nt1 = (a = 1.0, b = 2.0, c = 3.0)
        nt2 = (a = 0.0, b = 0.0, c = 0.0)
        @test dm(nt1, NamedTuple{(:c, :b, :a)}((0.0, 0.0, 0.0))) ≈ dm(nt1, nt2)
        @test_throws DimensionMismatch dm((a = 1.0,), (a = 0.0,))
        @test_throws DimensionMismatch dm([1.0, 2.0], [0.0, 0.0])

        # single-key fits accept scalars
        ds = make_mahalanobis(X[:, 1:1]; shrinkage = 0.5)
        @test ds(3.0, 1.0) ≈ 2.0 / std(X[:, 1])
        @test_throws DimensionMismatch dm(1.0, 0.0)

        # integer summaries (counts) are a normal input
        dint = make_mahalanobis([1 2; 3 5; 2 4]; shrinkage = 0.5)
        @test isfinite(dint([1.0, 2.0], [0.0, 0.0]))

        # duplicating a key is exactly what shrinkage exists for: λ = 0.5 stays
        # usable. (The raw λ = 0 fit is singular only up to floating point, so
        # it is not asserted to throw — LAPACK flags it when the pivot rounds
        # to ≤ 0, and rounding decides that, not the data.)
        Xs = hcat(X[:, 1], X[:, 1])
        dhalf = make_mahalanobis(Xs; shrinkage = 0.5)
        @test isfinite(dhalf([1.0, 1.0], [0.0, 0.0]))

        # a constant key has zero variance and cannot be regularised at any λ:
        # the factorisation fails deterministically and the message says why
        for lam in (0.0, 0.5)
            err2 = try
                make_mahalanobis(hcat(X[:, 1], ones(200)); shrinkage = lam)
                nothing
            catch e
                e
            end
            @test err2 isa ArgumentError
            @test occursin("zero variance", sprint(showerror, err2))
        end

        # input validation
        @test_throws ArgumentError make_mahalanobis(X; shrinkage = -0.1)
        @test_throws ArgumentError make_mahalanobis(X; shrinkage = 1.5)
        @test_throws ArgumentError make_mahalanobis(X; shrinkage = NaN)
        @test_throws ArgumentError make_mahalanobis(reshape([1.0, 2.0], 1, 2))
        @test_throws ArgumentError make_mahalanobis(zeros(3, 0))
        Xb = copy(X)
        Xb[1, 1] = NaN
        @test_throws ArgumentError make_mahalanobis(Xb)

        # deterministic: same pilot, same shrunk covariance and distances
        @test make_mahalanobis(X).Σ == ddef.Σ
        @test make_mahalanobis(X)([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) ==
              ddef([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        @test occursin("MahalanobisDistance", sprint(show, ddef))
    end

    # ── integration with the samplers ────────────────────────────────────────
    @testset "distances drive the engines; l2_distance is a drop-in" begin
        # the ad-hoc distance and l2_distance produce bit-identical runs
        st_new = run_inference(:smc, CFG_L2, OBS_DIS; N = 40, T = 3, base_seed = SEED_DIS)
        st_ref = run_inference(:smc, CFG_REF, OBS_DIS; N = 40, T = 3, base_seed = SEED_DIS)
        @test st_new.gen == 3
        @test st_new.θ == st_ref.θ
        @test st_new.w == st_ref.w
        @test st_new.dists == st_ref.dists
        @test maximum(st_new.dists) <= st_new.ε

        # a weighted factory as the config distance, replayed deterministically
        dw = weighted_l2([1.0, 4.0])
        cfg_w = ABCSMCConfig(_prior_dis, _sample_prior_dis, identity, dw,
                             GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_dis)
        st_w = run_inference(:smc, cfg_w, OBS_DIS; N = 40, T = 3, base_seed = SEED_DIS)
        @test st_w.gen == 3
        @test maximum(st_w.dists) <= st_w.ε
        again = run_inference(:smc, cfg_w, OBS_DIS; N = 40, T = 3, base_seed = SEED_DIS)
        @test again.θ == st_w.θ
        @test again.dists == st_w.dists

        # Mahalanobis fitted on a pilot ensemble drives SMC and MCMC
        rng = MersenneTwister(999)
        pilot = [_simulator_dis.evolve(_simulator_dis.model_builder(_sample_prior_dis(rng)), rng)
                 for _ in 1:60]
        Xp = reduce(vcat, permutedims.(pilot))          # 60 × 2
        dm = make_mahalanobis(Xp; shrinkage = 0.5)
        cfg_m = ABCSMCConfig(_prior_dis, _sample_prior_dis, identity, dm,
                             GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_dis)
        st_m = run_inference(:smc, cfg_m, OBS_DIS; N = 40, T = 3, base_seed = SEED_DIS)
        @test st_m.gen == 3
        @test all(isfinite, st_m.dists)
        @test maximum(st_m.dists) <= st_m.ε

        chain = run_inference(:mcmc, cfg_m, OBS_DIS; n_samples = 20, ε = 3.0,
                              base_seed = SEED_DIS)
        @test length(chain) == 21
        @test all(θ -> all(isfinite, θ), chain.state.θ)
    end
end
