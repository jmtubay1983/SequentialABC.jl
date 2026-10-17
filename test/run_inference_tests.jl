using SequentialABC
using Test
using Distributions
using Random

# The same deterministic 2-D toy problem the other engine suites use, kept tiny:
# θ ~ N(0, 4²I), s(θ) = θ + N(0, I), Euclidean distance.
_prior_inf(θ) = sum(logpdf(Normal(0.0, 4.0), x) for x in θ)
_sample_prior_inf(rng) = 4.0 .* randn(rng, 2)
_distance_inf(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
_simulator_inf = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = identity,
)

CFG_INF = ABCSMCConfig(_prior_inf, _sample_prior_inf, identity, _distance_inf,
                       GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_inf)
OBS_INF = [1.0, -1.0]
SEED_INF = 20260118
LADDER_INF = [6.0, 4.0]

@testset "run_inference — unified entrypoint" begin

    @testset ":smc forwards to run_smc" begin
        got = run_inference(:smc, CFG_INF, OBS_INF; N = 10, T = 2, ε0 = 6.0,
                            base_seed = SEED_INF)
        direct = run_smc(CFG_INF, OBS_INF; N = 10, T = 2, ε0 = 6.0,
                         base_seed = SEED_INF)

        @test got isa ABCSMCState
        @test typeof(got) === typeof(direct)   # no wrapping, same concrete type
        @test got.θ == direct.θ                # kwargs really are forwarded
        @test got.w == direct.w
        @test got.gen == 2
        @test got.base_seed == SEED_INF
    end

    @testset ":mcmc forwards to run_mcmc" begin
        # ABC-MCMC's population keyword is n_samples (there is no N) and ε must
        # be a Float64 — the dispatcher passes them through untouched.
        got = run_inference(:mcmc, CFG_INF, OBS_INF; n_samples = 50, ε = 6.0,
                            base_seed = SEED_INF)
        direct = run_mcmc(CFG_INF, OBS_INF; n_samples = 50, ε = 6.0,
                          base_seed = SEED_INF)

        @test got isa ABCMCMCChain
        @test typeof(got) === typeof(direct)
        @test got.state.θ == direct.state.θ
        @test length(got) == length(direct)
        @test got.accepted == direct.accepted
        @test got.ε == direct.ε
    end

    @testset ":pmc forwards to run_pmc" begin
        got = run_inference(:pmc, CFG_INF, OBS_INF; N = 50, T = 2, εs = LADDER_INF,
                            base_seed = SEED_INF)
        direct = run_pmc(CFG_INF, OBS_INF; N = 50, T = 2, εs = LADDER_INF,
                         base_seed = SEED_INF)

        @test got isa ABCSMCState
        @test typeof(got) === typeof(direct)
        @test got.θ == direct.θ
        @test got.cfg.Σ == direct.cfg.Σ        # PMC's per-generation adaptation included
        @test got.n_resamples == 0
        @test got.gen == 2
    end

    @testset "determinism through the dispatcher" begin
        a = run_inference(:smc, CFG_INF, OBS_INF; N = 10, T = 2, ε0 = 6.0,
                          base_seed = SEED_INF)
        b = run_inference(:smc, CFG_INF, OBS_INF; N = 10, T = 2, ε0 = 6.0,
                          base_seed = SEED_INF)
        c = run_inference(:smc, CFG_INF, OBS_INF; N = 10, T = 2, ε0 = 6.0,
                          base_seed = SEED_INF + 1)
        @test a.θ == b.θ
        @test a.θ != c.θ
    end

    @testset "unknown engine throws ArgumentError" begin
        @test_throws ArgumentError run_inference(:unknown, CFG_INF, OBS_INF)
        @test_throws ArgumentError run_inference(:bsl, CFG_INF, OBS_INF; N = 10, T = 2)
        @test_throws ArgumentError run_inference(:neural, CFG_INF, OBS_INF)

        # The message names the rejected engine and the supported ones.
        msg = try
            run_inference(:unknown, CFG_INF, OBS_INF; N = 10, T = 2)
            ""
        catch err
            sprint(showerror, err)
        end
        @test occursin("Unknown engine: unknown", msg)
        @test occursin("Supported: :smc, :mcmc, :pmc", msg)

        # It is thrown before anything is simulated: no seed work happened.
        @test_throws ArgumentError run_inference(:unknown, CFG_INF, OBS_INF; N = 10, T = 2)
    end

    @testset "engine-specific keywords still reach their engine" begin
        # A wrong keyword for the chosen engine must fail the way the engine
        # itself fails — the dispatcher does not swallow or rename keywords.
        #
        # run_smc has no n_samples, so :smc reports the unknown keyword.
        @test_throws MethodError begin
            run_inference(:smc, CFG_INF, OBS_INF; N = 10, T = 2, n_samples = 5)
        end
        # run_mcmc has no N: it fails exactly where run_mcmc(cfg, obs; N = 10, …)
        # would — on its own missing n_samples — not somewhere in the dispatcher.
        @test_throws UndefKeywordError begin
            run_inference(:mcmc, CFG_INF, OBS_INF; N = 10, ε = 6.0)
        end
        @test_throws UndefKeywordError begin
            run_mcmc(CFG_INF, OBS_INF; N = 10, ε = 6.0)   # the same failure, directly
        end
    end
end
