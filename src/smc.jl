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
                state.dists[i] = d
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
                    ε::Float64, M::Int=1, max_attempts::Int=10_000,
                    resample::Bool=true)

Run a single generation of ABC‑SMC with redraw‑until‑accepted.

For each particle i:
  1. Resample ancestor j according to current weights
  2. Perturb θ = perturb(θ_prev[j], Σ)
  3. Simulate (M replicates) → s̄
  4. Compute distance d
  5. If d ≤ ε accept, else redraw with new attempt counter

Ensures no zero‑weight particles ⇒ normalization and ESS always well‑defined.

`resample = false` skips the population resampling step that normally fires when
`ESS < 0.5 N`. Ancestor draws of the next generation already sample the weighted
population, which is exactly what population Monte Carlo relies on: the weights
stay informative instead of being reset to `1/N`. The current tolerance still
accepts particles, `state.ess` reports the true ESS and `state.n_resamples`
stays at its previous value. Used by [`run_pmc`](@ref).
"""
function run_generation!(state::ABCSMCState, summary_obs;
                         ε::Float64, M::Int=1, max_attempts::Int=10_000,
                         resample::Bool=true)
    cfg = state.cfg
    θ_prev, w_prev = state.θ, state.w
    N = length(θ_prev)

    θ_new = similar(θ_prev, N)
    w_new = zeros(Float64, N)
    sums = Vector{Any}(undef, N)  # will be re‑typed from first accept
    d_new = zeros(Float64, N)

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
                d_new[i] = d
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
    # population-level resampling draw, distinct from all other seed roles).
    # Disabled by `resample = false`, i.e. by population Monte Carlo.
    if resample && ess_val < 0.5N
        rng_resample = MersenneTwister(make_particle_seed(state.gen + 1, 0, 0, -2, state.base_seed))
        idx = systematic_resample(w_new, rng_resample)
        θ_new = θ_new[idx]
        sums = sums[idx]
        d_new = d_new[idx]
        fill!(w_new, 1.0 / N)
        ess_val = N
        state.n_resamples += 1
    end

    # Update state
    state.θ = θ_new
    state.w = w_new
    state.dists = d_new

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
             T::Int, schedule, M::Int=1, max_attempts::Int=10_000,
             checkpoint_dir::Union{Nothing,AbstractString}=nothing)

Run ABC‑SMC for T generations with adaptive tolerance schedule.

schedule should be a function: schedule(state, t) -> ε_t

If `checkpoint_dir` is given, an atomic checkpoint is written after
initialization and after every completed generation (`state_gen_XXX.jls`).
Restart by loading with `load_latest_state(checkpoint_dir, cfg)` and calling
`run_smc!` again with the same directory — the run resumes from `state.gen + 1`.
"""
function run_smc!(state::ABCSMCState, summary_obs;
                  T::Int, schedule, M::Int=1, max_attempts::Int=10_000,
                  checkpoint_dir::Union{Nothing,AbstractString}=nothing)

    # Run generation‑1 if not already present
    if state.gen == 0
        initialize!(state, summary_obs; max_attempts=max_attempts)
        checkpoint_dir === nothing || save_checkpoint(state, checkpoint_dir)
    end

    for t in (state.gen + 1):T
        ε_new = schedule(state, t)
        run_generation!(state, summary_obs; ε=ε_new, M=M, max_attempts=max_attempts)
        checkpoint_dir === nothing || save_checkpoint(state, checkpoint_dir)
    end

    return state
end

"""
    run_smc(cfg::ABCSMCConfig, summary_obs; N, T, schedule = QuantileSchedule(0.5),
            ε0 = Inf, M = 1, base_seed = 1, Tθ = nothing, max_attempts = 10_000,
            checkpoint_dir = nothing) -> ABCSMCState

One‑call ABC‑SMC: build the state, generate generation 1 from the prior and
advance to generation `T`, returning the final [`ABCSMCState`](@ref). This is the
counterpart of [`run_mcmc`](@ref), which has the same `(config, summary_obs;
kwargs…)` shape — no manual [`make_state`](@ref) / [`initialize!`](@ref) /
[`run_smc!`](@ref) bookkeeping.

Keywords:
  - `N::Int`      — population size
  - `T::Int`      — total generations, *including* generation 1
  - `schedule`    — callable `(state, t) -> ε_t`; only generations `t ≥ 2` consult
    it. See [`QuantileSchedule`](@ref) and [`FixedSchedule`](@ref); any function of
    `(state, t)` works
  - `ε0`          — tolerance used to accept generation 1. The default `Inf`
    accepts the first prior draw for every particle, so generation 1 *is* a plain
    prior sample and the schedule sets the real tolerance from generation 2 on;
    the state then keeps `ε = Inf` if `T == 1`. Pass a finite value to start from
    a tighter, pre‑filtered set instead
  - `M::Int`      — simulator replicates per particle (summaries are averaged)
  - `base_seed`   — base seed; the same arguments replay an identical run
  - `Tθ`          — parameter type returned by `cfg.sample_prior`. `nothing`
    (default) infers it by drawing one prior sample with a throwaway RNG and
    discarding the value; pass the type to skip that draw
  - `max_attempts` — attempts allowed per particle before erroring
  - `checkpoint_dir` — if given, an atomic checkpoint is written after generation 1
    and after every completed generation. Resume with [`load_latest_state`](@ref)
    plus [`run_smc!`](@ref) — calling `run_smc` again would restart from scratch

# Example
```julia
state = run_smc(cfg, s_obs; N = 1000, T = 10)
state = run_smc(cfg, s_obs; N = 1000, T = 4,
                schedule = FixedSchedule([4.0, 2.0, 1.0, 0.5]))
```
"""
function run_smc(cfg::ABCSMCConfig, summary_obs;
                 N::Int, T::Int,
                 schedule = QuantileSchedule(0.5),
                 ε0::Real = Inf,
                 M::Int = 1,
                 base_seed::Int = 1,
                 Tθ = nothing,
                 max_attempts::Int = 10_000,
                 checkpoint_dir::Union{Nothing,AbstractString} = nothing)
    N >= 1 || throw(ArgumentError("N must be ≥ 1, got $N"))
    T >= 1 || throw(ArgumentError("T must be ≥ 1, got $T"))
    M >= 1 || throw(ArgumentError("M must be ≥ 1, got $M"))

    # The state preallocates Vector{Tθ}; infer the prior's raw parameter type from
    # a discarded draw (throwaway RNG, so the run's deterministic stream is
    # untouched) unless the caller supplied it.
    Tθ_eff = Tθ === nothing ? typeof(cfg.sample_prior(MersenneTwister(base_seed))) : Tθ

    state = make_state(cfg, Tθ_eff, N, Float64(ε0), base_seed)
    return run_smc!(state, summary_obs; T = T, schedule = schedule, M = M,
                    max_attempts = max_attempts, checkpoint_dir = checkpoint_dir)
end