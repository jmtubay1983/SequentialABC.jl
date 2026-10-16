using SequentialABC
using Test
using Distributions
using Random

PRIOR = Normal(0.0, 5.0)
Y_OBS = 1.0

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

N = 100

@testset "Checkpoint round-trip" begin
    dir = mktempdir()

    state = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    path = save_checkpoint(state, dir)
    @test isfile(path)
    @test endswith(path, "state_gen_001.jls")

    # Reload and verify data integrity
    state2 = load_checkpoint(path, cfg)
    @test state2.gen == state.gen
    @test state2.ε == state.ε
    @test state2.ess == state.ess
    @test state2.base_seed == state.base_seed
    @test state2.n_attempts == state.n_attempts
    @test state2.n_resamples == state.n_resamples
    @test state2.θ == state.θ
    @test state2.w == state.w
    @test state2.dists == state.dists
    @test state2.summaries == state.summaries
end

@testset "list_checkpoints ordering and filtering" begin
    dir = mktempdir()

    s = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    initialize!(s, Y_OBS)
    save_checkpoint(s, dir)        # gen 1

    s.gen = 1
    s.ε = 1.0
    save_checkpoint(s, dir)        # gen 1 again (overwrites)
    s.gen = 3
    s.ε = 0.3
    save_checkpoint(s, dir)        # gen 3

    # Touch an unrelated file and a stale .tmp — both must be ignored
    touch(joinpath(dir, "README.txt"))
    touch(joinpath(dir, "state_gen_002.jls.tmp"))

    cps = list_checkpoints(dir)
    @test length(cps) == 2
    @test cps == [(1, joinpath(dir, "state_gen_001.jls")),
                  (3, joinpath(dir, "state_gen_003.jls"))]
    @test issorted([g for (g, _) in cps])
end

@testset "load_latest_state gets highest generation" begin
    dir = mktempdir()

    s = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    initialize!(s, Y_OBS)
    save_checkpoint(s, dir)

    s.gen = 5
    s.ε = 0.1
    save_checkpoint(s, dir)

    latest = load_latest_state(dir, cfg)
    @test latest.gen == 5
    @test latest.ε == 0.1
end

@testset "load_latest_state errors on empty dir" begin
    empty_dir = mktempdir()
    @test_throws ErrorException load_latest_state(empty_dir, cfg)
end
