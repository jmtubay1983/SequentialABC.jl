using SequentialABC
using Test
using Distributions
using LinearAlgebra
using Random
using Statistics

# ── tiny 2-D config for the `make_adaptive_kernel` dimension plumbing ─────────
# The `_tune` suffix keeps this file's globals apart from the other suites that
# share one Main namespace.
_tune_prior(θ) = 0.0
_tune_sample_prior(rng) = randn(rng, 2)
_tune_simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ,
    summary = identity,
)
CFG_TUNE = ABCSMCConfig(_tune_prior, _tune_sample_prior, identity, l2_distance,
                        GaussianKernel(), [1.0 0.0; 0.0 1.0], _tune_simulator)

@testset "Kernel tuning helpers" begin

    # ── auto_kernel_scale ────────────────────────────────────────────────────
    @testset "auto_kernel_scale: rules and factor" begin
        @test auto_kernel_scale(4) ≈ 2.38^2 / 4
        @test auto_kernel_scale(4; rule = :rosenthal) == auto_kernel_scale(4; rule = :gelman)
        @test auto_kernel_scale(3; rule = :beaumont) == 2.0
        @test auto_kernel_scale(3; rule = :unit) == 1.0
        @test auto_kernel_scale(3; rule = 5.0) == 5.0
        @test auto_kernel_scale(1) == 2.38^2

        # factor multiplies whichever rule was chosen
        @test auto_kernel_scale(4; factor = 2.5) ≈ 2.5 * (2.38^2 / 4)

        # Silverman: (4/(d+2))^(1/(d+4))
        @test auto_kernel_scale(4; rule = :silverman) ≈ (4 / (4 + 2))^(1 / (4 + 4))
        @test auto_kernel_scale(1; rule = :silverman) ≈ (4 / 3)^(1 / 5)

        # median heuristic: 2·median(χ²_d)/d, and 2·log 2 at d = 2
        @test auto_kernel_scale(2; rule = :median) ≈ 2 * log(2)
        @test auto_kernel_scale(5; rule = :median) ≈ 2 * quantile(Chisq(5), 0.5) / 5

        @test_throws ArgumentError auto_kernel_scale(0)
        @test_throws ArgumentError auto_kernel_scale(4; rule = :nope)
        @test_throws ArgumentError auto_kernel_scale(4; factor = 0.0)
        @test_throws ArgumentError auto_kernel_scale(4; factor = -1.0)
    end

    # ── shrinkage_covariance ─────────────────────────────────────────────────
    @testset "shrinkage_covariance: symmetric and PD" begin
        rng = MersenneTwister(11)
        X = randn(rng, 300, 3)
        C = shrinkage_covariance(X)
        @test C isa Matrix{Float64}
        @test size(C) == (3, 3)
        @test C ≈ C'                                   # symmetric
        @test isposdef(Symmetric(C))
        # a tiny λ leaves the sample covariance essentially unchanged
        @test C ≈ cov(X; dims = 1) rtol = 1e-3

        # repeated rows carry no variance information → λ·I, still PD
        Y = repeat([1.0 2.0 3.0], 10, 1)
        Cy = shrinkage_covariance(Y; λ = 1e-3)
        @test Cy ≈ 1e-3 * Matrix(I, 3, 3)
        @test isposdef(Symmetric(Cy))

        # a single row is the same degenerate case
        Cs = shrinkage_covariance(reshape([1.0, 2.0], 1, 2); λ = 1e-4)
        @test Cs ≈ 1e-4 * Matrix(I, 2, 2)

        # λ = 1 is the fully isotropic target (tr S / d) · I
        C1 = shrinkage_covariance(X; λ = 1.0)
        @test C1 ≈ (tr(cov(X; dims = 1)) / 3) * Matrix(I, 3, 3)

        @test_throws ArgumentError shrinkage_covariance(X; λ = -0.1)
        @test_throws ArgumentError shrinkage_covariance(X; λ = 1.5)
    end

    # ── bandwidth_from_cov ───────────────────────────────────────────────────
    @testset "bandwidth_from_cov: :diag, :full and the floor" begin
        C = [4.0 1.0; 1.0 9.0]
        @test bandwidth_from_cov(C) ≈ [2.0, 3.0]        # :diag is the default
        @test bandwidth_from_cov(C) isa Vector{Float64}

        L = cholesky(Symmetric(C)).L
        bwf = bandwidth_from_cov(C; method = :full)
        @test bwf ≈ diag(L)                              # conditional sds
        @test bwf[2] < bandwidth_from_cov(C)[2]          # correlation shrinks it

        # a diagonal covariance has no correlation → both methods agree
        D = Matrix(Diagonal([4.0, 9.0]))
        @test bandwidth_from_cov(D; method = :full) ≈ bandwidth_from_cov(D; method = :diag)

        # the floor lifts zero and near-zero bandwidths
        @test all(bandwidth_from_cov(zeros(3, 3); min_bandwidth = 1e-6) .== 1e-6)
        @test all(bandwidth_from_cov(Matrix(1e-30 * I(2)); method = :full,
                                     min_bandwidth = 1e-4) .== 1e-4)

        # a non-PD covariance has no Cholesky factor, but :diag still works
        bad = [1.0 2.0; 2.0 1.0]
        @test_throws ArgumentError bandwidth_from_cov(bad; method = :full)
        @test bandwidth_from_cov(bad) ≈ [1.0, 1.0]

        @test_throws ArgumentError bandwidth_from_cov(zeros(2, 3))
        @test_throws ArgumentError bandwidth_from_cov(C; method = :bogus)
        @test_throws ArgumentError bandwidth_from_cov(C; min_bandwidth = 0.0)
    end

    # ── make_adaptive_kernel ─────────────────────────────────────────────────
    @testset "make_adaptive_kernel: callable and deterministic" begin
        tuner = make_adaptive_kernel(CFG_TUNE)
        @test tuner isa SequentialABC.TunedKernel

        rng = MersenneTwister(29)
        X = randn(rng, 500, 2)
        bw = tuner(X)
        @test bw isa Vector{Float64}
        @test length(bw) == 2
        @test all(bw .> 0)
        @test all(bw .>= 1e-6)                 # the min_scale floor
        @test tuner(X) == tuner(X)             # pure function of the population

        # the vector-of-parameter-vectors layout (ABCSMCState.θ) agrees
        particles = [collect(x) for x in eachrow(X)]
        @test tuner(particles) == bw

        # it is exactly the documented composition (with an explicit method, since
        # the default `:auto` may choose `:full` for this population)
        tuner_diag = make_adaptive_kernel(CFG_TUNE; method = :diag)
        expected = auto_kernel_scale(2) .*
                   bandwidth_from_cov(shrinkage_covariance(X); method = :diag)
        @test tuner_diag(X) ≈ expected

        # explicit methods and rule are forwarded
        @test make_adaptive_kernel(CFG_TUNE; method = :diag)(X) ≈ expected
        @test all(make_adaptive_kernel(CFG_TUNE; method = :full)(X) .> 0)
        @test make_adaptive_kernel(CFG_TUNE; method = :diag, rule = :beaumont)(X) ≈
              2.0 .* bandwidth_from_cov(shrinkage_covariance(X); method = :diag)

        # a higher floor is respected
        @test all(make_adaptive_kernel(CFG_TUNE; min_scale = 1.0)(X) .>= 1.0)

        # dimension plumbing
        @test_throws DimensionMismatch tuner(randn(MersenneTwister(1), 10, 3))
        @test_throws ArgumentError make_adaptive_kernel(CFG_TUNE; method = :bogus)
        @test_throws ArgumentError make_adaptive_kernel(CFG_TUNE; min_scale = 0.0)
        @test_throws ArgumentError make_adaptive_kernel(CFG_TUNE; rule = :bogus)
        @test_throws ArgumentError make_adaptive_kernel((; x = 1))
    end

    @testset ":auto follows the correlation structure" begin
        # exactly uncorrelated columns → marginals suffice
        X_unc = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        @test make_adaptive_kernel(CFG_TUNE; method = :auto)(X_unc) ≈
              make_adaptive_kernel(CFG_TUNE; method = :diag)(X_unc)

        # strongly correlated columns → :auto switches to the full covariance
        rng = MersenneTwister(4)
        z = randn(rng, 400)
        X_cor = hcat(z, z .+ 0.05 .* randn(rng, 400))
        a = make_adaptive_kernel(CFG_TUNE; method = :auto)(X_cor)
        @test a ≈ make_adaptive_kernel(CFG_TUNE; method = :full)(X_cor)
        @test !isapprox(a, make_adaptive_kernel(CFG_TUNE; method = :diag)(X_cor))
    end
end
