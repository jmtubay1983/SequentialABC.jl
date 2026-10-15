using Random
using StatsBase

"""
    initialize!(state::ABCSMCState, summary_obs;
                max_attempts::Int=10_000)

Generate generation‑1 from the prior. Follows Toni et al. (2009) exactly:

For i = 1…N:
  1. Sample θ_raw from prior (deterministic RNG seeded from (gen, i, attempt, base))
  2. Simulate transform(θ_raw) → s(θ)
  3. Compute distance d = distance(s, summary_obs)
  4. If d ≤ ε accept, else repeat from step 1

Accepted particles get uniform weights 1/N (proposal = prior ⇒ ratio cancels).
Particles are stored on the **raw** parameter space (where the prior is defined);
the transform is applied only inside simulation. The first accepted summary
determines the type Tsum; subsequent summaries are typed as Vector{Tsum}.
"""
function initialize!(state::ABCSMCState, summary_obs;
                     max_attempts::Int=10_000)
    cfg = state.cfg
    N = length(state.θ)
    ε = state.ε

    for i in 1:N
        attempts = 0
        accepted = false

        while !accepted && attempts < max_attempts
            attempts += 1

            # Sample from prior with deterministic RNG
            rng_prior = MersenneTwister(make_particle_seed(1, i, attempts, 0, state.base_seed))
            θ_raw = cfg.sample_prior(rng_prior)

            # Simulate with current generation and particle index
            sims = run_sim_once(cfg, 1, i, attempts, θ_raw; M=1, base_seed=state.base_seed)
            s̄ = sims[1]

            # Compute distance
            d = cfg.distance(s̄, summary_obs)

            if d <= ε
                # Accept: store on the raw space (prior's space)
                state.θ[i] = θ_raw
                state.summaries[i] = s̄
                accepted = true
                state.n_attempts += attempts
            end
        end

        if !accepted
            error("ε=$ε unreachable after $max_attempts attempts for particle $i")
        end
    end

    # Uniform weights for gen-1 ⇒ ESS = N
    fill!(state.w, 1.0 / N)
    state.gen = 1
    state.ess = Float64(N)

    # Re‑type summaries vector from Any to the inferred Tsum
    Tsum = typeof(state.summaries[1])
    state.summaries = convert(Vector{Tsum}, state.summaries)

    return state
end

"""
    run_generation!(state::ABCSMCState, summary_obs;
                    ε::Float64, M::Int=1, max_attempts::Int=10_000)

Run a single generation of ABC‑SMC with redraw‑until‑accepted.

For each particle i:
  1. Resample ancestor j according to current weights
  2. Perturb θ = perturb(θ_prev[j], Σ)
  3. Simulate (M replicates) → s̄
  4. Compute distance d
  5. If d ≤ ε accept, else redraw with new attempt counter

Ensures no zero‑weight particles ⇒ normalization and ESS always well‑defined.
"""
function run_generation!(state::ABCSMCState, summary_obs;
                         ε::Float64, M::Int=1, max_attempts::Int=10_000)
    cfg = state.cfg
    θ_prev, w_prev = state.θ, state.w
    N = length(θ_prev)

    θ_new = similar(θ_prev, N)
    w_new = zeros(Float64, N)
    sums = Vector{Any}(undef, N)  # will be re‑typed from first accept

    for i in 1:N
        attempts = 0
        accepted = false

        while !accepted && attempts < max_attempts
            attempts += 1

            # Resample ancestor (deterministic RNG; m=0 tags the ancestor draw,
            # simulation replicates use m ≥ 1 so seeds never collide)
            rng_anc = MersenneTwister(make_particle_seed(state.gen + 1, i, attempts, 0, state.base_seed))
            j = sample(rng_anc, 1:length(θ_prev), Weights(w_prev))

            # Perturb with current generation (gen + 1)
            rng = MersenneTwister(make_particle_seed(state.gen + 1, i, attempts, -1, state.base_seed))
            θ_prop = perturb(cfg.kernel, θ_prev[j], cfg.Σ, rng)

            # Simulate (M replicates)
            sims = run_sim_once(cfg, state.gen + 1, i, attempts, θ_prop; M=M, base_seed=state.base_seed)

            # Aggregate summaries (mean of M replicates)
            s̄ = reduce(+, sims) / M

            # Accept/reject
            d = cfg.distance(s̄, summary_obs)

            if d <= ε
                # Accept: compute importance weight
                w_new[i] = importance_weight(cfg, θ_prop, θ_prev, w_prev)
                θ_new[i] = θ_prop
                sums[i] = s̄
                accepted = true
                state.n_attempts += attempts
            end
        end

        if !accepted
            error("ε=$(ε) unreachable after $max_attempts attempts for particle $i in gen $(state.gen+1)")
        end
    end

    # Normalize weights
    w_new ./= sum(w_new)

    # ESS
    ess_val = ess(w_new)

    # Resample population if ESS < 0.5 * N (deterministic RNG; m = −2 tags the
    # population-level resampling draw, distinct from all other seed roles)
    if ess_val < 0.5N
        rng_resample = MersenneTwister(make_particle_seed(state.gen + 1, 0, 0, -2, state.base_seed))
        idx = systematic_resample(w_new, rng_resample)
        θ_new = θ_new[idx]
        sums = sums[idx]
        fill!(w_new, 1.0 / N)
        ess_val = N
        state.n_resamples += 1
    end

    # Update state
    state.θ = θ_new
    state.w = w_new

    # Re‑type summaries
    Tsum = typeof(sums[1])
    state.summaries = convert(Vector{Tsum}, sums)

    state.ε = ε
    state.gen += 1
    state.ess = ess_val

    return state
end

"""
    run_smc!(state::ABCSMCState, summary_obs;
             T::Int, schedule, M::Int=1, max_attempts::Int=10_000)

Run ABC‑SMC for T generations with adaptive tolerance schedule.

schedule should be a function: schedule(state, t) -> ε_t
"""
function run_smc!(state::ABCSMCState, summary_obs;
                  T::Int, schedule, M::Int=1, max_attempts::Int=10_000)

    # Run generation‑1 if not already present
    if state.gen == 0
        initialize!(state, summary_obs; max_attempts=max_attempts)
    end

    for t in (state.gen + 1):T
        ε_new = schedule(state, t)
        run_generation!(state, summary_obs; ε=ε_new, M=M, max_attempts=max_attempts)
    end

    return state
end