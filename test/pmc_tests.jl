using SequentialABC
using Test
using Distributions
using Random
using Statistics

# Same deterministic 2-D reference problem as the run_smc suite:
# θ ~ N(0, 4²I), s(θ) = θ + N(0, I), distance = Euclidean norm.
_prior_pmc(θ) = sum(logpdf(Normal(0.0, 4.0), x) for x in θ)
_sample_prior_pmc(rng) = 4.0 .* randn(rng, 2)
_distance_pmc(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
_simulator_pmc = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = identity,
)

CFG_PMC = ABCSMCConfig(_prior_pmc, _sample_prior_pmc, identity, _distance_pmc,
                       GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_pmc)
OBS_PMC = [1.0, -1.0]
N_PMC = 60
SEED_PMC = 20260117
LADDER_PMC = [6.0, 4.0, 2.0, 1.5, 1.0]

# Same problem, but the prior puts (numerically) zero density on θ₁ ≤ 4 — so a
# majority of particles get importance weight 0 and the weights genuinely
# degenerate. That is the regime PMC and ABC-SMC differ in.
_cfg_degenerate = ABCSMCConfig(θ -> (θ[1] > 4.0 ? 0.0 : -1e6), _sample_prior_pmc, identity,
                               _distance_pmc, GaussianKernel(), [1.0 0.0; 0.0 1.0],
                               _simulator_pmc)

@testset "run_pmc — population Monte Carlo" begin

    @testset "fixed tolerance ladder εs" begin
        st = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 5, εs = LADDER_PMC,
                     base_seed = SEED_PMC)

        @test st.gen == 5
        @test st.ε == LADDER_PMC[5]                  # the last rung is the final tolerance
        @test isapprox(sum(st.w), 1.0; atol = 1e-12)
        @test all(w -> w >= 0, st.w)
        @test all(θ -> all(isfinite, θ), st.θ)
        @test maximum(st.dists) <= st.ε              # every accepted particle passed
        @test eltype(st.θ) === Vector{Float64}

        # εs may be any sized collection of Real — ranges included.
        range_ladder = 6.0 .- 1.0 .* (0:4)
        rng_run = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 5, εs = range_ladder,
                          base_seed = SEED_PMC)
        vec_run = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 5, εs = collect(range_ladder),
                          base_seed = SEED_PMC)
        @test rng_run.θ == vec_run.θ
        @test rng_run.w == vec_run.w
        @test rng_run.ε == range_ladder[5]
    end

    @testset "generation 1 is the prior sample filtered by ε₁" begin
        st = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 1, εs = [6.0],
                     base_seed = SEED_PMC)

        # A finite ε₁ rejects the part of the prior that is too far from the
        # data, so more than N draws were needed — but the survivors are plain
        # prior draws with the uniform prior-sample weights.
        @test st.ε == 6.0
        @test maximum(st.dists) <= 6.0                # every particle passed ε₁
        @test st.n_attempts > N_PMC                   # rejection happened
        @test st.w == fill(1.0 / N_PMC, N_PMC)        # uniform prior-sample weights
        @test st.ess == Float64(N_PMC)
        @test st.n_resamples == 0
        @test all(θ -> all(isfinite, θ), st.θ)

        # With ε₁ = Inf nothing is rejected: generation 1 is then bit-identical
        # to the documented per-particle seed roles.
        prior_run = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 1, base_seed = SEED_PMC)
        expected = [_sample_prior_pmc(MersenneTwister(make_particle_seed(1, i, 1, 0, SEED_PMC)))
                    for i in 1:N_PMC]
        @test prior_run.θ == expected
        @test prior_run.ε == Inf
        @test prior_run.n_attempts == N_PMC           # exactly one attempt per particle

        # εs takes precedence over ε0: the same ladder run either way.
        quiet = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 2, εs = [6.0, 4.0],
                        ε0 = 999.0, base_seed = SEED_PMC)
        plain = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 2, εs = [6.0, 4.0],
                        base_seed = SEED_PMC)
        @test quiet.θ == plain.θ
        @test quiet.ε == plain.ε

        # In schedule mode ε0 still governs generation 1.
        with_ε0 = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 1, ε0 = 6.0,
                          base_seed = SEED_PMC)
        @test with_ε0.ε == 6.0
        @test maximum(with_ε0.dists) <= 6.0
        @test with_ε0.θ != prior_run.θ                # the finite tolerance rejected some draws
    end

    @testset "adaptive tolerance ladder via schedule" begin
        st = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 5, base_seed = SEED_PMC)
        @test st.gen == 5
        @test isfinite(st.ε)
        @test maximum(st.dists) <= st.ε

        # The tolerance rule itself is shared with ABC-SMC: generation 1 takes the
        # same prior sample for both engines, so the quantile of that population
        # — the generation-2 tolerance — must agree.
        pmc2 = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 2, base_seed = SEED_PMC)
        smc2 = run_smc(CFG_PMC, OBS_PMC; N = N_PMC, T = 2, ε0 = Inf, base_seed = SEED_PMC)
        @test pmc2.ε == smc2.ε

        # A custom closure schedule is accepted, like run_smc's.
        const_sched = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3,
                              schedule = FixedSchedule([5.0, 3.0, 1.0]),
                              ε0 = 5.0, base_seed = SEED_PMC)
        @test const_sched.ε == 1.0
    end

    @testset "kernel covariance re-derived as 2Σ_{t-1} at every generation" begin
        for t in 1:3
            # The state after t generations is exactly the population generation
            # t+1 adapts from.
            pop = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = t, εs = LADDER_PMC[1:t],
                          base_seed = SEED_PMC)
            kern = AdaptiveKernel(CFG_PMC.Σ; rule = :beaumont)
            update!(kern, pop.θ, pop.w)

            run = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = t + 1,
                          εs = LADDER_PMC[1:(t + 1)], base_seed = SEED_PMC)
            @test run.cfg.Σ ≈ kern.Σ
            # PMC's rule is twice the weighted covariance of the population.
            @test kern.Σ ≈ 2.0 * estimate_covariance(pop.θ, pop.w; shrink = 1e-6)
        end

        # The Σ in the returned state is the one the final generation ran with,
        # and it is NOT the fixed Σ the user configured.
        st = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 5, εs = LADDER_PMC,
                     base_seed = SEED_PMC)
        @test st.cfg.Σ != CFG_PMC.Σ

        # rule choice: a literal factor 2.0 is :beaumont; :rosenthal differs.
        literal = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                          base_seed = SEED_PMC, kernel_rule = 2.0)
        beaumont = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                           base_seed = SEED_PMC, kernel_rule = :beaumont)
        rosenthal = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                            base_seed = SEED_PMC, kernel_rule = :rosenthal)
        @test literal.θ == beaumont.θ
        @test literal.cfg.Σ == beaumont.cfg.Σ
        @test rosenthal.cfg.Σ != beaumont.cfg.Σ
    end

    @testset "never resamples — and says so in the diagnostics" begin
        # In the degenerate-weight regime ABC-SMC fires its ESS-triggered
        # resampling (and its reported ESS is reset towards N, hiding the
        # degeneracy), while PMC keeps the weights and reports the truth.
        smc = run_smc(_cfg_degenerate, OBS_PMC; N = 120, T = 5, ε0 = 6.0,
                      base_seed = SEED_PMC)
        pmc = run_pmc(_cfg_degenerate, OBS_PMC; N = 120, T = 5,
                      εs = LADDER_PMC, base_seed = SEED_PMC)

        @test smc.n_resamples >= 1                   # the contrast exists …
        @test pmc.n_resamples == 0                   # … and PMC never took it
        @test pmc.ess == ess(pmc.w)                  # true ESS of the weighted population
        @test pmc.ess < 0.5 * 120                    # degeneracy is visible, not reset
        @test pmc.ess < smc.ess
        @test all(w -> w >= 0, pmc.w)
        @test isapprox(sum(pmc.w), 1.0; atol = 1e-12)
    end

    @testset "same mutation path as ABC-SMC when the covariance matches" begin
        # PMC at T = 2 must reproduce run_smc at T = 2 configured with the Σ PMC
        # would derive, as long as SMC does not resample along the way. This
        # pins down that PMC changes only the covariance rule and the resampling.
        ε1, ε2 = 6.0, 4.0
        gen1 = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 1, εs = [ε1],
                       base_seed = SEED_PMC)
        kern = AdaptiveKernel(CFG_PMC.Σ; rule = :beaumont)
        update!(kern, gen1.θ, gen1.w)

        smc = run_smc(with_covariance(CFG_PMC, kern), OBS_PMC; N = N_PMC, T = 2,
                      ε0 = ε1, schedule = FixedSchedule([ε1, ε2]), base_seed = SEED_PMC)
        pmc = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 2, εs = [ε1, ε2],
                      base_seed = SEED_PMC)

        @test smc.n_resamples == 0                   # prerequisite for exact equality
        @test pmc.n_resamples == 0
        @test smc.θ == pmc.θ
        @test smc.w == pmc.w
        @test smc.dists == pmc.dists
        @test smc.ε == pmc.ε
        @test smc.cfg.Σ == pmc.cfg.Σ
    end

    @testset "run_pmc! — state-based driver and equivalence" begin
        manual = make_state(CFG_PMC, Vector{Float64}, N_PMC, LADDER_PMC[1], SEED_PMC)
        run_pmc!(manual, OBS_PMC; T = 4, schedule = FixedSchedule(LADDER_PMC[1:4]))

        auto = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 4, εs = LADDER_PMC[1:4],
                       base_seed = SEED_PMC)

        @test typeof(auto) === typeof(manual)
        @test manual.gen == 4
        @test auto.θ == manual.θ
        @test auto.w == manual.w
        @test auto.dists == manual.dists
        @test auto.ε == manual.ε
        @test auto.ess == manual.ess
        @test auto.n_attempts == manual.n_attempts
        @test auto.n_resamples == manual.n_resamples == 0
        @test auto.cfg.Σ == manual.cfg.Σ
    end

    @testset "checkpointing and resume" begin
        dir = mktempdir()
        st = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                     base_seed = SEED_PMC, checkpoint_dir = dir)

        @test length(list_checkpoints(dir)) == 3
        loaded = load_latest_state(dir, CFG_PMC)
        @test loaded.gen == 3
        @test loaded.θ == st.θ
        @test loaded.w == st.w

        # Resuming must reproduce an uninterrupted run bit-for-bit, including the
        # kernel re-derivation (pass the same schedule/rule as the original run).
        run_pmc!(loaded, OBS_PMC; T = 4, schedule = FixedSchedule(LADDER_PMC[1:4]),
                 checkpoint_dir = dir)
        direct = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 4, εs = LADDER_PMC[1:4],
                         base_seed = SEED_PMC)
        @test loaded.gen == 4
        @test loaded.θ == direct.θ
        @test loaded.w == direct.w
        @test loaded.cfg.Σ == direct.cfg.Σ
        @test loaded.ess == direct.ess
    end

    @testset "determinism" begin
        a = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                    base_seed = SEED_PMC)
        b = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                    base_seed = SEED_PMC)
        c = run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3, εs = LADDER_PMC[1:3],
                    base_seed = SEED_PMC + 1)

        @test a.θ == b.θ
        @test a.w == b.w
        @test a.cfg.Σ == b.cfg.Σ
        @test a.n_attempts == b.n_attempts
        @test a.θ != c.θ
    end

    @testset "Tθ inference and override" begin
        inferred = run_pmc(CFG_PMC, OBS_PMC; N = 5, T = 1, εs = [6.0],
                           base_seed = SEED_PMC)
        @test eltype(inferred.θ) === Vector{Float64}

        # sample_prior returns a different container: the run must adapt.
        float_cfg = ABCSMCConfig(_prior_pmc, rng -> Float64.(2.0 .* randn(rng, 2)),
                                 identity, _distance_pmc, GaussianKernel(),
                                 [1.0 0.0; 0.0 1.0], _simulator_pmc)
        floats = run_pmc(float_cfg, OBS_PMC; N = 5, T = 1, εs = [6.0],
                         base_seed = SEED_PMC)
        @test eltype(floats.θ) === Vector{Float64}

        explicit = run_pmc(float_cfg, OBS_PMC; N = 5, T = 1, εs = [6.0],
                           Tθ = Vector{Float64}, base_seed = SEED_PMC)
        @test eltype(explicit.θ) === Vector{Float64}
        @test explicit.θ == floats.θ
    end

    @testset "argument validation" begin
        @test_throws ArgumentError run_pmc(CFG_PMC, OBS_PMC; N = 0, T = 1, εs = [6.0])
        @test_throws ArgumentError run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 0, εs = Float64[])
        @test_throws ArgumentError run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 1, εs = [6.0], M = 0)
        # ladder length must match T, and must not increase
        @test_throws ArgumentError run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3,
                                           εs = LADDER_PMC[1:2])
        @test_throws ArgumentError run_pmc(CFG_PMC, OBS_PMC; N = N_PMC, T = 3,
                                           εs = [2.0, 4.0, 6.0])
    end
end
