# Recipes

Every engine, schedule, kernel and summary in this package has a documented
"right starting value" — but they live on different pages. This page collects
them into three functions that hand those defaults back ready to drop into a
config or an engine call:

| You write | You get |
|---|---|
| `default_schedule(:smc)` | the adaptive median-quantile schedule both SMC and PMC default to |
| `default_schedule(T; ε0, εT)` | the geometric tolerance ladder of the engine-comparison benchmarks |
| `default_kernel_scaling(θs; engine)` | `Σ₀` from a pilot population: `2 · Ĉov(θ)` for SMC/PMC, `2.38²/d · Ĉov(θ)` for MCMC |
| `preset_summaries(:moments)` | a ready-made summary for the shape of your simulator output |

Nothing here is new machinery. Each recipe composes primitives that are
documented and tested on their own ([`QuantileSchedule`](@ref),
[`FixedSchedule`](@ref), [`estimate_covariance`](@ref),
[`adaptive_scale`](@ref), [`FunctionSummary`](@ref),
[`CompositeSummary`](@ref)), so a recommendation can always be read back to
the underlying knob — and swapped out piecewise.

## One run, all three

```julia
using SequentialABC, Distributions, Random

# What the simulator reports: the coordinates under stable names.
base = preset_summaries(:vector; names = (:a, :b))
sim  = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = raw -> namedtuple_to_vector(compute_summary(base, raw)),
)

# What the particles move with: 2 × the pilot covariance.
rng   = MersenneTwister(2026)
pilot = [sample_prior(rng) for _ in 1:200]
Σ0    = default_kernel_scaling(pilot)

cfg = ABCSMCConfig(prior, sample_prior, identity, l2_distance,
                   GaussianKernel(), Σ0, sim)
s_obs = [1.0, -1.0]                            # in summary_keys(base) order

state = run_inference(:smc, cfg, s_obs;
                      N = 1000, T = 6, schedule = default_schedule(:smc),
                      base_seed = 2026)
```

The preset names the summary keys, [`l2_distance`](@ref) is the distance those
keys flow into, the kernel recipe seeds the perturbation, and the schedule
recipe retightens the tolerance each generation. Every piece stays an ordinary
object: `base` can be `compose`d with further summaries, `Σ0` can be replaced
by any matrix, and the schedule is any callable `(state, t) -> ε`.

## Schedules

`default_schedule(engine)` returns the **adaptive** rule —
`QuantileSchedule(q)` with the house default `q = 0.5`, i.e. each generation
runs at the median of the previous generation's accepted distances. It is
exactly what [`run_smc`](@ref) and [`run_pmc`](@ref) already default to; the
recipe exists so the recommendation has a name and lives in one place.

```julia
schedule = default_schedule(:smc)               # QuantileSchedule(0.5)
schedule = default_schedule(:smc; q = 0.25)     # tighter adaptation
```

ABC-MCMC is the exception: it runs at a **single fixed ε** and takes no
schedule, so `default_schedule(:mcmc)` is an error that spells out the actual
recipe — calibrate ε from a short adaptive run (see [ABC-MCMC](@ref)):

```julia
st    = run_inference(:smc, cfg, s_obs; N = 500, T = 4, base_seed = 1)
ε_cal = epsilon_quantile(st, 0.5)               # the median accepted distance
chain = run_inference(:mcmc, cfg, s_obs; n_samples = 10_000, ε = ε_cal)
```

With a generation count, the second form builds the **prescribed geometric
ladder** between two tolerances — the schedule the engine-comparison benchmarks
run, where every run must see the same ε trajectory and the budget must be
known in advance:

```julia
ladder = default_schedule(5; ε0 = 4.0, εT = 0.4)
# εs = [4.0, 2.2493, 1.2649, 0.7113, 0.4]        one rung per generation
state  = run_inference(:smc, cfg, s_obs; N = 1000, T = 5,
                       schedule = ladder, ε0 = 4.0)
```

Generation 1 accepts at the engine's initial tolerance `ε0`: pass the first
rung there so the ladder and the run start together — or leave the default
`Inf`, and generation 1 is a plain prior sample. `run_pmc` consumes the same
ladder through its `εs` vector, which also sets its starting tolerance:

```julia
state = run_inference(:pmc, cfg, s_obs; N = 1000, T = 5, εs = ladder.ε)
```

## Kernel scalings

A fixed perturbation covariance is the classic pain point of ABC — too small
and the population cannot move, too large and almost every proposal misses the
acceptance region. The standard scaling per engine:

| `engine` | scaling | rationale |
|---|---|---|
| `:smc` / `:pmc` | `2 · Ĉov(θ)` | twice the population covariance, wide enough to counter resampling (Beaumont et al. 2009); `run_pmc` re-derives `Σ_t` every generation anyway |
| `:mcmc` | `2.38²/d · Ĉov(θ)` | the classic optimal random-walk scaling (Gelman, Roberts & Gilks 1996) |

```julia
rng   = MersenneTwister(2026)
pilot = [sample_prior(rng) for _ in 1:200]      # prior-predictive pilot
Σ_smc = default_kernel_scaling(pilot)           # :smc is the default engine
Σ_mc  = default_kernel_scaling(pilot; engine = :mcmc)
```

`samples` is either an `N × d` matrix or a length-`N` vector of parameter
vectors — the layout of `ABCSMCState.θ` — from a prior-predictive pilot or the
first generation of a short run. Underneath sits
[`estimate_covariance`](@ref), which shrinks toward isotropy, so the result
stays positive definite even for small or degenerate pilots; `weights` and
`shrink` forward to it. Without a pilot, `default_kernel_scaling(d)` falls back
to the scaled unit prior, `factor · I` — only meaningful when the prior is
roughly `𝒩(0, I)`; prefer the samples form as soon as a pilot exists.

The primitives remain available for other rules: call
[`adaptive_scale`](@ref) / [`scaling_factor`](@ref) with the rule you want
(see [Adaptive kernels](@ref) for all rules and the streaming estimator).

## Summary presets

`preset_summaries(preset)` returns a ready summary for the shape of output a
simulator actually reports:

| `preset` | input | keys |
|---|---|---|
| `:scalar` | one number | `name` (default `:value`) |
| `:vector` | a vector of numbers | `names`, or `:v1, :v2, …` via `d` |
| `:moments` | a collection of numbers | `:count`, `:mean`, `:sd` |
| `:quantiles` | a collection of numbers | `probs`, named by percentage |

```julia
base = preset_summaries(:vector; names = (:a, :b))
summary_keys(base)                               # [:a, :b]

pop = preset_summaries(:moments)                 # replicates within one θ
compute_summary(pop, [1.0, 2.0, 3.0, 4.0])       # (count = 4, mean = 2.5, sd = 1.291…)

q = preset_summaries(:quantiles; probs = (0.1, 0.9))
summary_keys(q)                                  # [:q10, :q90]
```

Points worth knowing:

- Keys are declared, not discovered: `:moments` always reports exactly
  `:count`, `:mean`, `:sd`, and `:quantiles` names keys by percentage — `0.975`
  becomes `:q97_5`, `1/20` becomes `:q5`.
- Passing a keyword the preset does not use is an error, and duplicate keys
  are rejected at construction — a typo cannot silently drop a summary from
  the distance.
- The result is an ordinary summary object: compose it with more summaries,
  rescale it with [`scale`](@ref), reduce it with [`pca_reduce`](@ref),
  average replicates with [`aggregate`](@ref) — see
  [Summary statistics](@ref) for the full framework.

## Determinism

The recipes are pure functions of their arguments: the same pilot gives the
same `Σ₀`, the same `(T, ε0, εT)` the same ladder, the same preset the same
keys and values — no RNG anywhere. The pilot itself is a data choice, exactly
like `s_obs`: fix its seed, and the whole run stays reproducible under the
engines' existing replay guarantee.

## API

```@docs
default_schedule
default_kernel_scaling
preset_summaries
```
