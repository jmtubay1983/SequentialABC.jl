using SequentialABC
using Test
using Distributions
using Random
using Statistics

const PRIOR = Normal(0.0, 5.0)
const Y_OBS = 1.0

sample_prior = rng -> [rand(rng, PRIOR)]
prior_logpdf = θ -> logpdf(PRIOR, θ[1])
transform = θ -> θ
distance = (s, s_obs) -> abs(s - s_obs)

simulator = ContinuousSimulator(
    model_builder = μ -> μ[1],
    evolve = (μ, rng) -> μ + randn(rng),
    summary = y -> y,
)

Σ_mat = [0.5;;]
cfg = ABCSMCConfig(prior_logpdf, sample_prior, transform, distance,
                   GaussianKernel(), Σ_mat, simulator)

schedule = (state, t) -> [3.0, 1.0, 0.3][t]
N = 100

@testset "Full ABC-SMC pipeline (3 generations)" begin
    state = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T=3, schedule=schedule)

    @test state.gen == 3
    @test isapprox(sum(state.w), 1.0; atol=1e-10)
    @test all(isfinite, state.w)
    @test all(θ -> all(isfinite, θ), state.θ)
    @test all(>(0), state.w)
    @test 1.0 <= state.ess <= N + 1e-9

    # Weighted posterior mean should be near the analytic value ≈ 0.96
    θbar = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
    @test 0.4 < θbar < 1.6
end

@testset "Determinism — identical seeds give bit-identical results" begin
    state1 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state1, Y_OBS; T=3, schedule=schedule)

    state2 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state2, Y_OBS; T=3, schedule=schedule)

    @test state1.θ == state2.θ
    @test state1.w == state2.w
    @test state1.ε == state2.ε
    @test state1.ess == state2.ess
end

@testset "Different seeds give different populations" begin
    state1 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state1, Y_OBS; T=3, schedule=schedule)

    state3 = make_state(cfg, Vector{Float64}, N, 3.0, 9999)
    run_smc!(state3, Y_OBS; T=3, schedule=schedule)

    @test state1.θ != state3.θ
end

@testset "Generation 1: uniform weights and ESS = N" begin
    state = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    @test state.gen == 1
    @test all(w -> isapprox(w, 1.0 / N; atol=1e-12), state.w)
    @test state.ess == N
end

@testset "FixedSchedule == equivalent hand-rolled lambda" begin
    fs = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(fs, Y_OBS; T=3, schedule=FixedSchedule([3.0, 1.0, 0.3]))

    ref = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(ref, Y_OBS; T=3, schedule=schedule)

    @test fs.θ == ref.θ
    @test fs.w == ref.w
end

@testset "FixedSchedule rejects non-monotone input" begin
    @test_throws ArgumentError FixedSchedule([0.3, 1.0])
end

@testset "QuantileSchedule monotonicity and equivalence" begin
    eps_hist = Float64[]
    recording = (st, t) -> begin
        ε = epsilon_quantile(st, 0.5)
        push!(eps_hist, ε)
        return ε
    end
    s1 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(s1, Y_OBS; T=4, schedule=recording)

    @test length(eps_hist) == 3
    @test issorted(eps_hist; rev=true)
    @test all(>(0), eps_hist)
    @test s1.ε == eps_hist[end]

    # ε_2 must equal the median of gen-1's accepted distances
    s2 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    initialize!(s2, Y_OBS)
    ε2_expected = quantile(s2.dists, 0.5)
    run_generation!(s2, Y_OBS; ε=ε2_expected)
    @test isapprox(s2.ε, ε2_expected; rtol=1e-12)

    # QuantileSchedule object gives the identical trajectory
    s3 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(s3, Y_OBS; T=4, schedule=QuantileSchedule(0.5))
    @test s3.θ == s1.θ && s3.w == s1.w

    # Constructor validation
    @test_throws ArgumentError QuantileSchedule(0.0)
    @test_throws ArgumentError QuantileSchedule(1.5)
end

@testset "Restart equivalence — checkpointed run == uninterrupted run" begin
    dir = mktempdir()

    # Run to gen 2 with checkpointing
    qa = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(qa, Y_OBS; T=2, schedule=QuantileSchedule(0.5),
             checkpoint_dir=dir)

    # Reload and continue to gen 4
    qb = load_latest_state(dir, cfg)
    @test qb.dists == qa.dists
    run_smc!(qb, Y_OBS; T=4, schedule=QuantileSchedule(0.5),
             checkpoint_dir=dir)

    # Uninterrupted run to gen 4
    qc = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    run_smc!(qc, Y_OBS; T=4, schedule=QuantileSchedule(0.5))

    @test qb.θ == qc.θ
    @test qb.w == qc.w
    @test qb.ε == qc.ε
end
