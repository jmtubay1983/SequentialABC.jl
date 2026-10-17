# src/engines/PMC.jl
#
# Population Monte Carlo for ABC (Beaumont et al. 2009, "Adaptive approximate
# Bayesian computation"). The population loop, weights, seeding and diagnostics
# are shared with ABC-SMC; PMC is distinguished by two choices: a perturbation
# covariance re-derived from the population at every generation, and a
# population that is never resampled.

"""
    run_pmc!(state::ABCSMCState, summary_obs; T, schedule,
             kernel_rule = :beaumont, M = 1, max_attempts = 10_000,
             checkpoint_dir = nothing) -> ABCSMCState

Advance a PMC population from `state.gen + 1` to generation `T`. `schedule` is a
callable `(state, t) -> ε_t` (e.g. [`FixedSchedule`](@ref) or
[`QuantileSchedule`](@ref)) and `kernel_rule` is the covariance scaling rule —
the two engines' contract on resume is the same, so pass the **same `T`,
`schedule` and `kernel_rule`** you started the run with. If `state.gen == 0` the
prior sample of generation 1 is taken first, exactly as `run_smc!` does.

See [`run_pmc`](@ref) for the algorithm, every keyword and the trade-offs.
"""
function run_pmc!(state::ABCSMCState, summary_obs;
                  T::Int,
                  schedule,
                  kernel_rule::Union{Symbol,Real} = :beaumont,
                  M::Int = 1,
                  max_attempts::Int = 10_000,
                  checkpoint_dir::Union{Nothing,AbstractString} = nothing)

    # Generation 1 from the prior, uniform weights — Beaumont's f₀ → f₁ step.
    if state.gen == 0
        initialize!(state, summary_obs; max_attempts = max_attempts)
        checkpoint_dir === nothing || save_checkpoint(state, checkpoint_dir)
    end

    # Rebuilt from cfg.Σ on every entry (including a resumed one); the very first
    # update overwrites that seed covariance from the population, so the stale
    # value never influences a proposal.
    kernel = AdaptiveKernel(state.cfg.Σ; rule = kernel_rule)

    for t in (state.gen + 1):T
        # Σ_t ← rule · Cov(θ_{t-1}, w_{t-1}), adopted before any particle of
        # generation t exists: consistent proposal and importance weights.
        update!(kernel, state.θ, state.w)
        state.cfg = with_covariance(state.cfg, kernel)

        # resample = false is the PMC choice: weighted ancestor draws only.
        run_generation!(state, summary_obs; ε = schedule(state, t), M = M,
                        max_attempts = max_attempts, resample = false)
        checkpoint_dir === nothing || save_checkpoint(state, checkpoint_dir)
    end

    return state
end

"""
    run_pmc(cfg::ABCSMCConfig, summary_obs; N, T, εs = nothing,
            schedule = QuantileSchedule(0.5), ε0 = Inf, kernel_rule = :beaumont,
            M = 1, base_seed = 1, Tθ = nothing, max_attempts = 10_000,
            checkpoint_dir = nothing) -> ABCSMCState

Population Monte Carlo for approximate Bayesian computation (Beaumont et al.
2009): the same redraw‑until‑accepted population loop as ABC‑SMC, run with the
two choices that define PMC.

1. **The perturbation covariance is re‑derived from the population at every
   generation** — `Σ_t = scaling_factor(d; kernel_rule) · Cov(θ_{t-1}, w_{t-1})`,
   with `kernel_rule = :beaumont` (×2) as default. It is adopted at the
   *generation boundary*, before any particle of that generation exists, so a
   single `Σ` serves the whole generation and its importance weights.
   [`run_smc`](@ref) instead keeps `cfg.Σ` fixed, which is what makes hand
   tuning matter.
2. **The population is never resampled.** The ESS‑triggered
   [`systematic_resample`](@ref) step of ABC‑SMC is skipped (`resample = false`
   in [`run_generation!`](@ref)): weights stay informative across generations and
   the *weighted ancestor draws* do the importance sampling — that is the
   mechanism PMC is named for. So `state.ess` reports the real ESS of each
   generation rather than resetting to `N`, and `state.n_resamples` stays `0`,
   which is the diagnostic that confirms you are running PMC.

Everything else is deliberately identical to `run_smc`: the perturbation‑corrected
weight `w ∝ π(θ) / Σ_j w_j K(θ | θ_j, Σ)` from [`importance_weight`](@ref),
deterministic per‑particle streams from [`make_particle_seed`](@ref), the same
diagnostics, and the same `ABCSMCState` return value — so weights, plots,
checkpointing and benchmark tooling work unchanged.

## Keywords

  - `N::Int`  — population size
  - `T::Int`  — total generations, *including* generation 1
  - `εs`      — **fixed tolerance ladder**: one value per generation,
    non‑increasing, `length(εs) == T`. Generation `t` runs at `εs[t]`, so the
    first entry is the tolerance of the prior sample. This is the historical form
    — Toni et al. (2009) and Beaumont et al. (2009) both take `ε₁ > … > ε_T` as
    input. When given, `ε0` is ignored
  - `schedule` — **adaptive tolerance** instead: any callable
    `(state, t) -> ε_t`, e.g. [`QuantileSchedule`](@ref) (the default) or a
    closure. Only generations `t ≥ 2` consult it
  - `ε0` — tolerance accepting generation 1 in `schedule` mode; the default `Inf`
    makes generation 1 a plain prior sample
  - `kernel_rule` — scaling rule applied to the population covariance, see
    [`scaling_factor`](@ref): `:beaumont` (×2, PMC's usual choice),
    `:rosenthal` (`2.38²/d`), `:unit` (×1) or a literal `Real` factor
  - `M`, `base_seed`, `Tθ`, `max_attempts`, `checkpoint_dir` — as for
    [`run_smc`](@ref)

[`run_pmc!`](@ref) is the state‑based driver, mirroring `run_smc!`; it runs
generation 1 if `state.gen == 0` and then advances to generation `T`, which is
also how you resume.

# Example
```julia
state = run_pmc(cfg, s_obs; N = 1000, T = 5, εs = [4.0, 2.0, 1.0, 0.6, 0.4])
state = run_pmc(cfg, s_obs; N = 1000, T = 5)          # adaptive quantile ladder
state.ess                                             # true ESS of the final population
```

The cost of choice 2 is weight degeneracy: without resampling the ESS can drift
down across generations, which PMC compensates for by re‑estimating `Σ_t` from
the weighted population. Watch `state.ess`, or compare engines directly with
`bench/compare_engines.jl`.
"""
function run_pmc(cfg::ABCSMCConfig, summary_obs;
                 N::Int, T::Int,
                 εs = nothing,
                 schedule = QuantileSchedule(0.5),
                 ε0::Real = Inf,
                 kernel_rule::Union{Symbol,Real} = :beaumont,
                 M::Int = 1,
                 base_seed::Int = 1,
                 Tθ = nothing,
                 max_attempts::Int = 10_000,
                 checkpoint_dir::Union{Nothing,AbstractString} = nothing)
    N >= 1 || throw(ArgumentError("N must be ≥ 1, got $N"))
    T >= 1 || throw(ArgumentError("T must be ≥ 1, got $T"))
    M >= 1 || throw(ArgumentError("M must be ≥ 1, got $M"))

    if εs === nothing
        sched = schedule
        ε_init = Float64(ε0)
    else
        length(εs) == T ||
            throw(ArgumentError("εs must contain one tolerance per generation " *
                                "(T = $T), got length $(length(εs))"))
        sched = FixedSchedule(collect(Float64, εs))  # also validates non-increasing
        ε_init = Float64(εs[1])                      # generation 1 runs at ε₁
    end

    # Parameter type of cfg.sample_prior, inferred from a discarded draw taken
    # with a throwaway RNG so the run's deterministic stream is untouched.
    Tθ_eff = Tθ === nothing ? typeof(cfg.sample_prior(MersenneTwister(base_seed))) : Tθ

    state = make_state(cfg, Tθ_eff, N, ε_init, base_seed)
    return run_pmc!(state, summary_obs; T = T, schedule = sched,
                    kernel_rule = kernel_rule, M = M, max_attempts = max_attempts,
                    checkpoint_dir = checkpoint_dir)
end
