using SequentialABC
using Test
using Distributions
using Random
using Statistics

# Deterministic 2-D reference problem: θ ~ N(0, 4²I), s(θ) = θ + N(0, I),
# distance = Euclidean norm. Small N keeps the suite fast.
_prior_logpdf(θ) = sum(logpdf(Normal(0.0, 4.0), x) for x in θ)
_sample_prior(rng) = 4.0 .* randn(rng, 2)
_distance(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
_simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = identity,
)

CFG_RUNSMC = ABCSMCConfig(_prior_logpdf, _sample_prior, identity, _distance,
                          GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator)
OBS_RUNSMC = [1.0, -1.0]
N_RUNSMC = 60
SEED_RUNSMC = 20260116

@testset "run_smc — one-call entrypoint" begin

    @testset "equivalent to the manual make_state/initialize!/run_smc! path" begin
        ε0 = 6.0
        sched = QuantileSchedule(0.5)

        manual = make_state(CFG_RUNSMC, Vector{Float64}, N_RUNSMC, ε0, SEED_RUNSMC)
        initialize!(manual, OBS_RUNSMC)
        run_smc!(manual, OBS_RUNSMC; T = 4, schedule = sched)

        auto = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 4, schedule = sched,
                       ε0 = ε0, base_seed = SEED_RUNSMC)

        @test typeof(auto) === typeof(manual)
        @test auto.θ == manual.θ
        @test auto.w == manual.w
        @test auto.dists == manual.dists
        @test auto.ε == manual.ε
        @test auto.gen == manual.gen
        @test auto.ess == manual.ess
        @test auto.n_attempts == manual.n_attempts
        @test auto.n_resamples == manual.n_resamples
        @test eltype(auto.θ) === Vector{Float64}
        @test auto.gen == 4
        @test isapprox(sum(auto.w), 1.0; atol = 1e-12)
    end

    @testset "default ε0 = Inf ⇒ generation 1 is a plain prior sample" begin
        st = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 1, base_seed = SEED_RUNSMC)
        expected = [_sample_prior(MersenneTwister(make_particle_seed(1, i, 1, 0, SEED_RUNSMC)))
                    for i in 1:N_RUNSMC]

        @test st.θ == expected
        @test st.n_attempts == N_RUNSMC          # exactly one attempt per particle
        @test st.ε == Inf                        # no tolerance applied (T == 1)
        @test st.gen == 1
        @test st.w == fill(1.0 / N_RUNSMC, N_RUNSMC)
        @test st.ess == Float64(N_RUNSMC)

        # With T ≥ 2 the schedule replaces Inf at generation 2.
        st3 = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 3, base_seed = SEED_RUNSMC)
        @test st3.gen == 3
        @test isfinite(st3.ε)
        # every accepted particle of the final generation lies inside the tolerance
        @test maximum(st3.dists) <= st3.ε
    end

    @testset "determinism and seed sensitivity" begin
        a = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 3, base_seed = SEED_RUNSMC)
        b = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 3, base_seed = SEED_RUNSMC)
        c = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 3, base_seed = SEED_RUNSMC + 1)

        @test a.θ == b.θ
        @test a.w == b.w
        @test a.ε == b.ε
        @test a.θ != c.θ
    end

    @testset "explicit schedules" begin
        ladder = [6.0, 4.0, 2.0, 1.5]
        st = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 4,
                     schedule = FixedSchedule(ladder), ε0 = ladder[1],
                     base_seed = SEED_RUNSMC)
        @test st.gen == 4
        @test st.ε == 1.5

        # a plain (state, t) -> ε closure is accepted too
        st2 = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 3,
                      schedule = (state, t) -> 2.0, ε0 = 6.0, base_seed = SEED_RUNSMC)
        @test st2.ε == 2.0
        @test st2.gen == 3
    end

    @testset "M replicates and explicit Tθ" begin
        st = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 2, M = 2,
                     base_seed = SEED_RUNSMC, Tθ = Vector{Float64})
        @test st.gen == 2
        @test eltype(st.θ) === Vector{Float64}
        @test all(θ -> length(θ) == 2, st.θ)
        @test isapprox(sum(st.w), 1.0; atol = 1e-12)
        @test all(θ -> all(isfinite, θ), st.θ)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError run_smc(CFG_RUNSMC, OBS_RUNSMC; N = 0, T = 3)
        @test_throws ArgumentError run_smc(CFG_RUNSMC, OBS_RUNSMC; N = 10, T = 0)
        @test_throws ArgumentError run_smc(CFG_RUNSMC, OBS_RUNSMC; N = 10, T = 3, M = 0)
    end

    @testset "checkpointing writes and resumes bit-identically" begin
        dir = mktempdir()
        st = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 3,
                     base_seed = SEED_RUNSMC, checkpoint_dir = dir)

        @test length(list_checkpoints(dir)) == 3      # generations 1, 2, 3

        loaded = load_latest_state(dir, CFG_RUNSMC)
        @test loaded.gen == 3
        @test loaded.θ == st.θ
        @test loaded.w == st.w

        # Continuing to the same T is a no-op that writes no extra checkpoint.
        run_smc!(loaded, OBS_RUNSMC; T = 3, schedule = QuantileSchedule(0.5),
                 checkpoint_dir = dir)
        @test loaded.θ == st.θ
        @test length(list_checkpoints(dir)) == 3

        # A resumed extension matches the uninterrupted run exactly.
        run_smc!(loaded, OBS_RUNSMC; T = 4, schedule = QuantileSchedule(0.5),
                 checkpoint_dir = dir)
        direct = run_smc(CFG_RUNSMC, OBS_RUNSMC; N = N_RUNSMC, T = 4,
                         base_seed = SEED_RUNSMC)
        @test loaded.gen == 4
        @test length(list_checkpoints(dir)) == 4
        @test loaded.θ == direct.θ
        @test loaded.w == direct.w
        @test loaded.ε == direct.ε
    end
end
