# src/engines/MCMC.jl
#
# ABC‑MCMC (Marjoram, Molitor, Plagnol & Tavaré 2003).
#
# The chain targets the ABC posterior
#
#     π_ε(θ | s_obs) ∝ π(θ) · 1[ d(s(θ), s_obs) ≤ ε ],
#
# using a symmetric Gaussian random‑walk proposal θ' ~ 𝒩(θ, Σ). Because the
# proposal is symmetric the Metropolis–Hastings ratio collapses to the prior
# ratio π(θ') / π(θ), so a step is accepted when
#
#     d(s(θ'), s_obs) ≤ ε   and   u < min(1, exp(log π(θ') − log π(θ))).
#
# Every stochastic step draws from an explicitly seeded MersenneTwister whose
# seed is a pure function of `(iteration, role, base_seed)` (see
# `make_particle_seed`), so a chain replays exactly — across processes, restarts
# and Slurm nodes.
#
# The engine deliberately reuses the ABC‑SMC building blocks (`ABCSMCConfig`,
# `run_sim_once`, the `GaussianKernel` perturbation) and returns its samples in
# an `ABCSMCState`, so everything downstream of an SMC run — `plot_marginals`,
# `adjusted_posterior`, `save_checkpoint` — works on MCMC output unchanged.

using Random
using Statistics

# Seed roles. Mirrors the SMC convention (cf. `make_particle_seed`): distinct
# `m` sentinels guarantee the proposal / simulation / accept streams can never
# collide within a single iteration.
const _MCMC_M_PROPOSAL = -11
const _MCMC_M_ACCEPT   = -12

"""
    ABCMCMCChain

Output of [`run_mcmc`](@ref): a Markov chain whose stationary distribution is
the ABC posterior `π_ε(θ | s_obs)`.

Fields:
  - `state::ABCSMCState`     — stored samples (the initial state plus one snapshot
    every `thin` iterations) with uniform weights. This is the *same* container an
    ABC‑SMC run produces, so downstream tooling works unchanged.
  - `accepted::Vector{Bool}` — accept/reject decision of every chain iteration
  - `ε::Float64`             — tolerance the chain was run at
  - `Σ::Matrix{Float64}`     — random‑walk proposal covariance
  - `base_seed::Int`         — base seed the chain was run with
  - `n_simulations::Int`     — total number of simulator calls (diagnostic)

Use `length(chain)` for the number of stored samples and
[`acceptance_rate`](@ref) for the acceptance diagnostic.
"""
struct ABCMCMCChain
    state::ABCSMCState
    accepted::Vector{Bool}
    ε::Float64
    Σ::Matrix{Float64}
    base_seed::Int
    n_simulations::Int
end

Base.length(chain::ABCMCMCChain) = length(chain.state.θ)

"""
    acceptance_rate(chain::ABCMCMCChain) -> Float64

Fraction of proposed moves that were accepted — the first diagnostic of an
ABC‑MCMC run. Very low rates mean the proposal covariance `Σ` is too large for
the tolerance `ε`; rates close to 1 mean `Σ` is too small for the chain to
explore the posterior.
"""
acceptance_rate(chain::ABCMCMCChain) = count(chain.accepted) / length(chain.accepted)

function Base.show(io::IO, chain::ABCMCMCChain)
    print(io, "ABCMCMCChain(", length(chain), " samples, acceptance rate ",
          round(acceptance_rate(chain); digits = 3),
          ", ε = ", round(chain.ε; digits = 4), ")")
    return nothing
end

# --- proposal interface ------------------------------------------------------

"""
    mcmc_propose(cfg::ABCSMCConfig, θ, rng::AbstractRNG) -> Vector{Float64}

Draw a proposal `θ' ~ 𝒩(θ, Σ)` from the random‑walk proposal, where `Σ` is
`cfg.Σ`. Thin wrapper around the ABC‑SMC perturbation kernel, so an ABC‑MCMC and
an ABC‑SMC run move particles identically given the same seed and covariance.
"""
mcmc_propose(cfg::ABCSMCConfig, θ, rng::AbstractRNG) = perturb(cfg.kernel, θ, cfg.Σ, rng)

"""
    mcmc_log_proposal(cfg::ABCSMCConfig, θ_new, θ_old) -> Float64

Log density `log K(θ_new | θ_old)` of the random‑walk proposal, including the
normalising constant.

The Gaussian random walk is **symmetric**,
`mcmc_log_proposal(cfg, a, b) == mcmc_log_proposal(cfg, b, a)`, which is exactly
what reduces the Metropolis–Hastings ratio to the prior ratio.
"""
mcmc_log_proposal(cfg::ABCSMCConfig, θ_new, θ_old) =
    cfg.kernel(θ_new, θ_old, cfg.Σ_inv, cfg.logdetΣ)

# --- acceptance rule ---------------------------------------------------------

"""
    abc_mcmc_accept(logprior_prop, logprior_cur, d_prop, ε, u;
                    log_proposal_ratio=0.0) -> Bool

Metropolis–Hastings acceptance test for ABC‑MCMC, exposed as a pure function so
the sampling logic can be tested in isolation:

1. `d_prop > ε` ⇒ reject — the proposal falls outside the ABC acceptance region,
   whatever the prior ratio says;
2. `log α = (logprior_prop − logprior_cur) + log_proposal_ratio`; reject when
   `log α` is `NaN` (e.g. both prior densities are `-Inf`, i.e. outside support);
3. otherwise accept with probability `min(1, exp(log α))`, decided by the uniform
   draw `u ∈ [0, 1)`.

`u` must always be drawn by the caller — even when the distance test already
fails — so the random stream never depends on the data.
"""
function abc_mcmc_accept(logprior_prop, logprior_cur, d_prop, ε, u;
                         log_proposal_ratio::Real = 0.0)
    d_prop <= ε || return false
    logα = (logprior_prop - logprior_cur) + log_proposal_ratio
    isnan(logα) && return false
    logα >= 0 && return true
    return u < exp(logα)
end

# --- chain -------------------------------------------------------------------

# Draw a starting point from the prior that already satisfies d ≤ ε, using a
# fresh deterministic seed on every attempt. Returns (θ, summary, distance, n_attempts).
function _mcmc_initial(cfg::ABCSMCConfig, summary_obs, ε, base_seed, max_attempts, M)
    for attempt in 1:max_attempts
        rng = MersenneTwister(make_particle_seed(1, 0, attempt, _MCMC_M_PROPOSAL, base_seed))
        θ_raw = cfg.sample_prior(rng)
        sims = run_sim_once(cfg, 1, 0, attempt, θ_raw; M = M, base_seed = base_seed)
        s̄ = reduce(+, sims) / M
        d = cfg.distance(s̄, summary_obs)
        if d <= ε && isfinite(cfg.prior(θ_raw))
            return θ_raw, s̄, d, attempt
        end
    end
    return error("no initial θ with d ≤ ε = $ε found in $max_attempts attempts; " *
                 "increase ε or max_attempts")
end

"""
    run_mcmc(cfg::ABCSMCConfig, summary_obs; n_samples, ε, thin=1, M=1,
             base_seed=1, max_attempts=10_000) -> ABCMCMCChain

Run an ABC‑MCMC chain (Marjoram et al. 2003) of `n_samples * thin` iterations
and return the [`ABCMCMCChain`](@ref).

  - `n_samples::Int`  — number of snapshots to store (plus the initial state)
  - `ε::Float64`      — ABC tolerance; proposals with `d(s', s_obs) > ε` are rejected
  - `thin::Int`       — keep one sample every `thin` iterations (reduces autocorrelation)
  - `M::Int`          — simulator replicates per iteration (summaries are averaged)
  - `base_seed::Int`  — base seed; the same config and seed replay an identical chain
  - `max_attempts::Int` — attempts allowed when searching for a feasible start point

The chain is initialised from a prior draw that already satisfies `d ≤ ε` and
then follows the random‑walk Metropolis rule of [`abc_mcmc_accept`](@ref) at
fixed tolerance `ε`. Accepted particles are returned in an [`ABCSMCState`](@ref)
with uniform weights (length `n_samples + 1`).
"""
function run_mcmc(cfg::ABCSMCConfig, summary_obs;
                  n_samples::Int,
                  ε::Float64,
                  thin::Int = 1,
                  M::Int = 1,
                  base_seed::Int = 1,
                  max_attempts::Int = 10_000)
    n_samples >= 1 || throw(ArgumentError("n_samples must be ≥ 1, got $n_samples"))
    thin >= 1 || throw(ArgumentError("thin must be ≥ 1, got $thin"))
    ε >= 0 || throw(ArgumentError("ε must be ≥ 0, got $ε"))
    M >= 1 || throw(ArgumentError("M must be ≥ 1, got $M"))

    θ_cur, s_cur, d_cur, n_start = _mcmc_initial(cfg, summary_obs, ε, base_seed,
                                                 max_attempts, M)
    logprior_cur = cfg.prior(θ_cur)

    n_iters = n_samples * thin
    accepted = falses(n_iters)

    Tθ = typeof(θ_cur)
    Tsum = typeof(s_cur)
    samples = Vector{Tθ}(undef, n_samples + 1)
    summaries = Vector{Tsum}(undef, n_samples + 1)
    dists = Vector{Float64}(undef, n_samples + 1)
    samples[1], summaries[1], dists[1] = θ_cur, s_cur, d_cur

    stored = 1
    n_sim = n_start * M

    for iter in 1:n_iters
        rng_prop = MersenneTwister(make_particle_seed(1, iter, 1, _MCMC_M_PROPOSAL, base_seed))
        θ_prop = mcmc_propose(cfg, θ_cur, rng_prop)

        sims = run_sim_once(cfg, 1, iter, 1, θ_prop; M = M, base_seed = base_seed)
        s_prop = reduce(+, sims) / M
        d_prop = cfg.distance(s_prop, summary_obs)
        logprior_prop = cfg.prior(θ_prop)

        # always draw u, so the random stream is independent of the data
        rng_acc = MersenneTwister(make_particle_seed(1, iter, 1, _MCMC_M_ACCEPT, base_seed))
        u = rand(rng_acc)

        if abc_mcmc_accept(logprior_prop, logprior_cur, d_prop, ε, u)
            θ_cur, s_cur, d_cur = θ_prop, s_prop, d_prop
            logprior_cur = logprior_prop
            accepted[iter] = true
        end
        n_sim += M

        if iter % thin == 0
            stored += 1
            samples[stored] = θ_cur
            summaries[stored] = s_cur
            dists[stored] = d_cur
        end
    end

    N = length(samples)
    w = fill(1.0 / N, N)
    # gen = 1 marks "one ABC population"; n_attempts is reused as the simulation
    # count and n_resamples stays 0 — all three are diagnostics only.
    state = ABCSMCState(cfg, samples, w, summaries, dists, ε, 1, Float64(N),
                        base_seed, n_sim, 0, nothing)
    return ABCMCMCChain(state, collect(accepted), ε, Matrix{Float64}(cfg.Σ),
                        base_seed, n_sim)
end
