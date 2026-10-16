using Test
using SequentialABC
using Distributions
using Random
using Statistics

# --- a tiny one-parameter toy problem ---------------------------------------
#      θ ~ N(0, 3²),   s = θ + N(0, 0.25²)
_mcmc_toy() = begin
    prior(θ) = logpdf(Normal(0.0, 3.0), θ[1])
    sample_prior(rng) = [3.0 * randn(rng)]
    sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ[1] + 0.25 * randn(rng),
        summary = identity,
    )
    ABCSMCConfig(prior, sample_prior, identity,
                 (s, so) -> abs(s - so), GaussianKernel(), [0.04;;], sim)
end

@testset "ABC-MCMC" begin
    cfg = _mcmc_toy()
    s_obs = 0.7

    @testset "acceptance logic (abc_mcmc_accept)" begin
        # outside the acceptance region: always rejected, whatever the prior ratio
        @test !abc_mcmc_accept(0.0, 0.0, 1.0, 0.5, 0.0)
        @test !abc_mcmc_accept(1e6, -1e6, 0.51, 0.5, 0.0)

        # inside the region: a non-decreasing prior ratio is always accepted
        @test abc_mcmc_accept(0.0, 0.0, 0.5, 0.5, 0.99)
        @test abc_mcmc_accept(-1.0, -2.0, 0.1, 0.5, 0.99)

        # ...and a decreasing one is accepted with probability exp(Δ)
        Δ = -log(4.0)                      # acceptance probability 0.25
        @test abc_mcmc_accept(Δ, 0.0, 0.1, 0.5, 0.10)
        @test !abc_mcmc_accept(Δ, 0.0, 0.1, 0.5, 0.30)

        # zero prior density ⇒ reject; NaN prior ratio (both -Inf) ⇒ reject
        @test !abc_mcmc_accept(-Inf, 0.0, 0.1, 0.5, 0.0)
        @test !abc_mcmc_accept(-Inf, -Inf, 0.1, 0.5, 0.0)

        # an explicit proposal ratio is honoured
        @test !abc_mcmc_accept(0.0, 0.0, 0.1, 0.5, 0.5; log_proposal_ratio = -log(10.0))
        @test abc_mcmc_accept(0.0, 0.0, 0.1, 0.5, 0.5; log_proposal_ratio = log(10.0))
    end

    @testset "proposal interface" begin
        rng = MersenneTwister(1)
        θ = [0.3]
        θp = mcmc_propose(cfg, θ, rng)
        @test θp isa Vector{Float64}
        @test length(θp) == 1

        # deterministic given the seed
        @test mcmc_propose(cfg, θ, MersenneTwister(1)) == θp

        # symmetric random walk: K(θ' | θ) == K(θ | θ')
        a, b = [0.1], [-0.4]
        @test mcmc_log_proposal(cfg, a, b) ≈ mcmc_log_proposal(cfg, b, a)

        # ...and it agrees with the Gaussian density of the proposal
        @test mcmc_log_proposal(cfg, a, b) ≈ logpdf(MvNormal(b, cfg.Σ), a)
    end

    @testset "run_mcmc: shape, weights, determinism" begin
        chain = run_mcmc(cfg, s_obs; n_samples = 200, ε = 0.6, base_seed = 7)

        @test chain isa ABCMCMCChain
        @test length(chain) == 201                      # initial state + one per iteration
        @test length(chain.accepted) == 200
        @test length(chain.state.θ) == 201
        @test all(p -> length(p) == 1 && all(isfinite, p), chain.state.θ)
        @test isapprox(sum(chain.state.w), 1.0)
        @test all(>(0.0), chain.state.w)

        # every stored sample lies inside the acceptance region
        @test all(isfinite, chain.state.dists)
        @test all(d -> d <= 0.6 + 1e-12, chain.state.dists)

        @test 0.0 <= acceptance_rate(chain) <= 1.0
        @test chain.ε == 0.6
        @test chain.Σ == cfg.Σ
        @test chain.n_simulations >= 200
        @test occursin("ABCMCMCChain", sprint(show, chain))

        # the chain reuses the SMC result container
        @test chain.state.cfg === cfg
        @test chain.state.ε == 0.6
        @test chain.state.gen == 1

        # reproducibility: same seed ⇒ identical chain, different seed ⇒ different
        same = run_mcmc(cfg, s_obs; n_samples = 200, ε = 0.6, base_seed = 7)
        @test chain.state.θ == same.state.θ
        @test chain.accepted == same.accepted

        other = run_mcmc(cfg, s_obs; n_samples = 200, ε = 0.6, base_seed = 8)
        @test chain.state.θ != other.state.θ
    end

    @testset "thinning and replicates" begin
        chain = run_mcmc(cfg, s_obs; n_samples = 50, ε = 0.6, thin = 4, M = 3,
                         base_seed = 11)
        @test length(chain) == 51
        @test length(chain.accepted) == 200             # n_samples * thin iterations
        @test chain.n_simulations >= 200 * 3
    end

    @testset "validation" begin
        @test_throws ArgumentError run_mcmc(cfg, s_obs; n_samples = 0, ε = 0.6)
        @test_throws ArgumentError run_mcmc(cfg, s_obs; n_samples = 10, ε = -0.1)
        @test_throws ArgumentError run_mcmc(cfg, s_obs; n_samples = 10, ε = 0.6, thin = 0)
        @test_throws ArgumentError run_mcmc(cfg, s_obs; n_samples = 10, ε = 0.6, M = 0)
        # ε = 0 is unreachable for a continuous summary ⇒ the start search fails
        @test_throws ErrorException run_mcmc(cfg, s_obs; n_samples = 10, ε = 0.0,
                                             max_attempts = 20)
    end
end
