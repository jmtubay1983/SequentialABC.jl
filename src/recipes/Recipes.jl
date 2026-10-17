# src/recipes/Recipes.jl
#
# Recipes: the house recommendations, in one place.
#
# Every knob in this package has a documented "right starting value", but they
# are spread across pages: the tolerance schedule (`QuantileSchedule(0.5)`), the
# perturbation covariance (twice the estimated population covariance for SMC and
# PMC; `2.38²/d` for MCMC), and the summary a simulator should compute for the
# shape of output it returns. These functions collect the recommendations and
# hand back values that go straight into a config or an engine call:
#
#   * `default_schedule(engine)`      — the adaptive quantile rule;
#     `default_schedule(T; ε0, εT)`   — the geometric ladder between two
#                                       tolerances (prescribed schedules).
#   * `default_kernel_scaling(samples; engine)` — Σ₀ from a pilot population.
#   * `preset_summaries(preset)`      — scalar / per-key vector / moments /
#                                       quantiles for the shapes real simulator
#                                       outputs come in.
#
# Nothing here is new machinery: every recipe composes primitives that are
# already documented (`QuantileSchedule`, `FixedSchedule`, `estimate_covariance`,
# `adaptive_scale`, `scaling_factor`, `FunctionSummary`, `CompositeSummary`), so
# each recommendation can be read back to the underlying knob.

using LinearAlgebra
using Statistics

# ── engine → kernel scaling rule ─────────────────────────────────────────────
#
# `:beaumont` — twice the estimated covariance: the standard ABC‑SMC/PMC
# perturbation kernel, wide enough to counter the resampling step (Beaumont
# et al. 2009). `:rosenthal` — 2.38²/d: the classic optimal random‑walk scaling
# for MCMC (Gelman, Roberts & Gilks 1996).
function _recipe_kernel_rule(engine::Symbol)
    engine === :smc && return :beaumont
    engine === :pmc && return :beaumont
    engine === :mcmc && return :rosenthal
    throw(ArgumentError("unknown engine $(repr(engine)); use :smc, :pmc or :mcmc"))
end

# ── schedules ────────────────────────────────────────────────────────────────

"""
    default_schedule(engine::Symbol; q = 0.5) -> QuantileSchedule
    default_schedule(T::Integer; ε0, εT) -> FixedSchedule

Recommended tolerance schedule for a workflow, returned as a callable
`(state, t) -> ε_t` ready for the `schedule` keyword of [`run_smc`](@ref) /
[`run_pmc`](@ref):

  - `engine = :smc` or `:pmc` — the adaptive q‑quantile rule,
    `QuantileSchedule(q)`, with the house default `q = 0.5` (the median of the
    previous generation's accepted distances, the Toni et al. (2009) fixed
    ladder replaced by adaptation). This is exactly what both engines default
    to; naming it here keeps the recommendation in one place.
  - `engine = :mcmc` — **there is no schedule**: ABC‑MCMC runs at a single
    fixed `ε`. Calibrate it from a short adaptive run instead, e.g.
    `st = run_smc(cfg, s_obs; N = 500, T = 4); ε = epsilon_quantile(st, 0.5)`.
    The error message repeats this recipe.

```julia
schedule = default_schedule(:smc)             # QuantileSchedule(0.5)
state    = run_inference(:smc, cfg, s_obs; N = 1000, T = 6, schedule = schedule)
```

With a generation count, the geometric ladder between two tolerances — the
prescribed schedule used by the engine‑comparison benchmarks, where every run
must see the same ε trajectory:

```julia
schedule = default_schedule(5; ε0 = 4.0, εT = 0.4)
# εs = [4.0, 2.2493, 1.2649, 0.7113, 0.4]     one rung per generation
state    = run_inference(:smc, cfg, s_obs; N = 1000, T = 5,
                         schedule = schedule, ε0 = 4.0)
```

Generation 1 accepts at the engine's initial tolerance `ε0`; pass the first rung
there so the ladder and the run start together (left at the default `Inf`,
generation 1 is a plain prior sample). `run_pmc` consumes the same ladder
through its `εs` vector, which also sets its starting tolerance:
`run_pmc(cfg, s_obs; T = 5, εs = default_schedule(5; ε0 = 4.0, εT = 0.4).ε)`.

Requires `T ≥ 2`, `0 < εT ≤ ε0`, and finite endpoints. Prefer the adaptive form
— it needs no tuning; take the geometric ladder when the ε budget must be known
in advance (cost planning, HPC) or held fixed for comparisons.
"""
function default_schedule(engine::Symbol; q::Real = 0.5)
    if engine === :mcmc
        throw(ArgumentError(
            "ABC-MCMC runs at a single fixed tolerance and takes no schedule: " *
            "calibrate `ε` with a short adaptive run instead, e.g. " *
            "`st = run_smc(cfg, s_obs; N = 500, T = 4); " *
            "ε = epsilon_quantile(st, 0.5)`"))
    end
    (engine === :smc || engine === :pmc) || throw(ArgumentError(
        "unknown engine $(repr(engine)); schedules exist for :smc and :pmc"))
    return QuantileSchedule(q)
end

function default_schedule(T::Integer; ε0::Real, εT::Real)
    T >= 2 || throw(ArgumentError("T must be ≥ 2 to fit both ε0 and εT, got $T"))
    (isfinite(ε0) && ε0 > 0) ||
        throw(ArgumentError("ε0 must be finite and positive, got $ε0"))
    (isfinite(εT) && εT > 0) ||
        throw(ArgumentError("εT must be finite and positive, got $εT"))
    εT <= ε0 || throw(ArgumentError(
        "εT must be ≤ ε0 (tolerances tighten down), got ε0 = $ε0, εT = $εT"))

    εs = Float64[ε0 * (εT / ε0)^((t - 1) / (T - 1)) for t in 1:T]
    εs[1] = Float64(ε0)     # pin both endpoints against rounding: the interior
    εs[end] = Float64(εT)   # is geometric, the ends are exactly what was asked
    return FixedSchedule(εs)  # also validates the ladder is non-increasing
end

# ── kernel scalings ──────────────────────────────────────────────────────────

"""
    default_kernel_scaling(samples; engine = :smc, weights = nothing, shrink = 1e-6)
    default_kernel_scaling(d::Integer; engine = :smc)

Recommended initial perturbation covariance `Σ₀` for a config, estimated from a
pilot population and scaled by the standard rule for the workflow:

| `engine` | scaling | rationale |
|---|---|---|
| `:smc` | `2 · Ĉov(θ)` | twice the population covariance — the usual ABC‑SMC perturbation kernel (Beaumont et al. 2009), wide enough to counter resampling |
| `:pmc` | `2 · Ĉov(θ)` | the same rule; `run_pmc` re‑derives `Σ_t` every generation anyway, so this only seeds the adaptive kernel |
| `:mcmc` | `2.38²/d · Ĉov(θ)` | the classic optimal random‑walk scaling (Gelman, Roberts & Gilks 1996) |

`samples` may be an `N × d` matrix or a length‑`N` vector of parameter vectors
(exactly the layout of `ABCSMCState.θ`) — take it from a prior‑predictive pilot
or from a short uncalibrated run. The estimate underneath is
[`estimate_covariance`](@ref), which shrinks towards isotropy and therefore stays
positive definite even for tiny or degenerate pilots; `weights` and `shrink` are
forwarded to it.

```julia
rng   = MersenneTwister(2026)
pilot = [sample_prior(rng) for _ in 1:200]
Σ0    = default_kernel_scaling(pilot)             # :smc ⇒ 2 × pilot covariance
cfg   = ABCSMCConfig(prior, sample_prior, transform, distance,
                     GaussianKernel(), Σ0, sim)
```

Without a pilot yet, `default_kernel_scaling(d)` falls back to the scaled unit
prior, `factor · I` — only meaningful when the prior is roughly `𝒩(0, I)`;
prefer the samples form as soon as a pilot exists.

The primitives stay available for other choices: [`estimate_covariance`](@ref)
for the raw estimate and [`adaptive_scale`](@ref) / [`scaling_factor`](@ref) for
custom rules (see [Adaptive kernels](@ref)).
"""
function default_kernel_scaling(samples; engine::Symbol = :smc,
                                weights = nothing, shrink::Real = 1e-6)
    rule = _recipe_kernel_rule(engine)
    Σ = estimate_covariance(samples, weights; shrink = shrink)
    return adaptive_scale(Σ; rule = rule)
end

function default_kernel_scaling(d::Integer; engine::Symbol = :smc)
    rule = _recipe_kernel_rule(engine)
    return scaling_factor(d; rule = rule) .* Matrix{Float64}(I, d, d)
end

# ── summary presets ──────────────────────────────────────────────────────────

# The presets `preset_summaries` accepts; anything else is a typo.
const _RECIPE_PRESETS = (:scalar, :vector, :moments, :quantiles)

# Reject keyword arguments that the chosen preset does not use, so a misplaced
# `names`/`probs` cannot silently do nothing.
function _recipe_check_preset_kwargs(preset::Symbol, names, d, probs)
    bad = String[]
    if preset !== :vector
        names === nothing || push!(bad, "names")
        d === nothing || push!(bad, "d")
    end
    preset !== :quantiles && probs !== nothing && push!(bad, "probs")
    isempty(bad) || throw(ArgumentError(
        "preset `$preset` does not take " * join(("`$k`" for k in bad), ", ")))
    return nothing
end

function _recipe_vector_names(names, d)
    if names === nothing
        d === nothing && throw(ArgumentError(
            "preset `:vector` needs `names` (one Symbol per entry) or `d`"))
        d >= 1 || throw(ArgumentError("d must be ≥ 1, got $d"))
        return [Symbol(:v, i) for i in 1:d]
    end
    isempty(names) && throw(ArgumentError("names must not be empty"))
    d !== nothing && d != length(names) && throw(ArgumentError(
        "d = $d does not match the $(length(names)) names given"))
    return collect(names)
end

function _recipe_check_probs(ps::AbstractVector{<:Real})
    isempty(ps) && throw(ArgumentError("probs must not be empty"))
    for p in ps
        (isfinite(p) && 0 <= p <= 1) || throw(ArgumentError(
            "quantile probabilities must be finite and in [0, 1], got $p"))
    end
    issorted(ps) ||
        throw(ArgumentError("probs must be sorted in increasing order, got $ps"))
    allunique(ps) || throw(ArgumentError("probs must be distinct, got $ps"))
    return ps
end

# `0.1` → `:q10`, `0.975` → `:q97_5`: named by percentage, so the key carries
# the probability the summary was built at.
function _recipe_pct_key(p::Real)
    s = string(100 * Float64(p))
    endswith(s, ".0") && (s = chop(s, tail = 2))
    return Symbol(:q, replace(s, "." => "_"))
end

"""
    preset_summaries(preset::Symbol; name = nothing, names = nothing,
                     d = nothing, probs = nothing) -> AbstractSummary

Ready-made summary for the four shapes simulator outputs come in — the object a
`ContinuousSimulator`/`AgentsSimulator` should put in its `summary` field (see
[Summary statistics](@ref)):

| `preset` | input | keys | example |
|---|---|---|---|
| `:scalar` | one number | `name` (default `:value`) | `(value = 3.5,)` |
| `:vector` | a vector of numbers | `names`, or `:v1, :v2, …` via `d` | `(a = 10.0, b = 20.0)` |
| `:moments` | a collection of numbers | `:count`, `:mean`, `:sd` | `(count = 200, mean = 3.1, sd = 0.4)` |
| `:quantiles` | a collection of numbers | `probs`, named by percentage | `(q10 = …, q50 = …, q90 = …)` |

```julia
base = preset_summaries(:vector; names = (:a, :b))
summary_keys(base)                              # [:a, :b]
namedtuple_to_vector(compute_summary(base, [10.0, 20.0]))   # [10.0, 20.0]

pop = preset_summaries(:moments)                # count/mean/sd over samples
compute_summary(pop, [1.0, 2.0, 3.0, 4.0])      # (count = 4, mean = 2.5, sd = 1.291…)

q = preset_summaries(:quantiles; probs = (0.25, 0.75))
summary_keys(q)                                 # [:q25, :q75]
```

`name` renames the summary itself (its key for `:scalar`, its name for the
composite presets); `:vector` takes `names` (one `Symbol` per entry) or `d`
(defaults `:v1 … :vd`); `:quantiles` takes `probs`, defaulting to `(0.1, 0.5,
0.9)` and named by percentage (`0.975` becomes `:q97_5`). Passing a keyword the
chosen preset does not use is an error, and so is a key clash with another part
when the preset is composed — keys are not silently dropped or renamed.

The result is an ordinary summary ([`FunctionSummary`](@ref) or
[`CompositeSummary`](@ref)): compose it with `compose`, rescale it with
[`scale`](@ref), reduce it with [`pca_reduce`](@ref), or average replicates with
[`aggregate`](@ref), exactly as any hand-written summary.
"""
function preset_summaries(preset::Symbol;
                          name::Union{Nothing,Symbol} = nothing,
                          names::Union{Nothing,AbstractVector{Symbol},Tuple{Vararg{Symbol}}} = nothing,
                          d::Union{Nothing,Integer} = nothing,
                          probs::Union{Nothing,AbstractVector{<:Real},Tuple{Vararg{Real}}} = nothing)
    preset in _RECIPE_PRESETS || throw(ArgumentError(
        "unknown summary preset $(repr(preset)); use " *
        join(("`:$p`" for p in _RECIPE_PRESETS), ", ")))

    _recipe_check_preset_kwargs(preset, names, d, probs)

    if preset === :scalar
        return FunctionSummary(identity, something(name, :value))
    elseif preset === :moments
        return CompositeSummary([FunctionSummary(length, :count),
                                 FunctionSummary(mean, :mean),
                                 FunctionSummary(std, :sd)],
                                something(name, :moments))
    elseif preset === :vector
        ks = _recipe_vector_names(names, d)
        return CompositeSummary(
            [FunctionSummary(v -> v[i], ks[i]) for i in eachindex(ks)],
            something(name, :vector))
    else  # :quantiles
        ps = _recipe_check_probs(
            probs === nothing ? [0.1, 0.5, 0.9] : collect(Float64, probs))
        return CompositeSummary(
            [FunctionSummary(v -> quantile(v, p), _recipe_pct_key(p)) for p in ps],
            something(name, :quantiles))
    end
end
