using SequentialABC
import SequentialABC: SEED_SCHEMA_VERSION
using Test
using Distributions
using Random
using Serialization

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

@testset "Checkpoint seed_schema_version validation" begin
    dir = mktempdir()

    state = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    path = save_checkpoint(state, dir)

    # A checkpoint saved with the current schema version loads cleanly.
    loaded = load_checkpoint(path, cfg)
    @test loaded.gen == state.gen

    # A checkpoint with a different seed_schema_version must be rejected.
    bad_version_dir = mktempdir()
    bad_path = joinpath(bad_version_dir, "state_gen_001.jls")
    open(bad_path, "w") do io
        serialize(io, (θ = state.θ,
                       w = state.w,
                       summaries = state.summaries,
                       dists = state.dists,
                       ε = state.ε,
                       gen = state.gen,
                       ess = state.ess,
                       base_seed = state.base_seed,
                       n_attempts = state.n_attempts,
                       n_resamples = state.n_resamples,
                       seed_schema_version = SEED_SCHEMA_VERSION + 1))
    end
    @test_throws ErrorException load_checkpoint(bad_path, cfg)

    # A checkpoint missing seed_schema_version entirely must be rejected.
    no_version_dir = mktempdir()
    no_version_path = joinpath(no_version_dir, "state_gen_001.jls")
    open(no_version_path, "w") do io
        serialize(io, (θ = state.θ,
                       w = state.w,
                       summaries = state.summaries,
                       dists = state.dists,
                       ε = state.ε,
                       gen = state.gen,
                       ess = state.ess,
                       base_seed = state.base_seed,
                       n_attempts = state.n_attempts,
                       n_resamples = state.n_resamples))
    end
    @test_throws ErrorException load_checkpoint(no_version_path, cfg)
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
