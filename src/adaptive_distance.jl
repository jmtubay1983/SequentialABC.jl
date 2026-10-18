# src/adaptive_distance.jl
#
# Adaptive inverse-distance scaling for ABC-SMC.
#
# Two layers, strictly separated:
#   * Immutable configuration — AdaptiveDistanceConfig: strategy, estimator
#     source, initializer, and the single NumericalPolicy. Set once at
#     construction, never mutated, re-supplied by the caller when resuming a
#     checkpointed run.
#   * Mutable runtime state — AdaptiveDistanceState (internal): summary key
#     order, current inverse-distance weights, generation estimated,
#     epsilon-selection distances, and the next epsilon. Concrete field types
#     only; checkpointed separately from the configuration.
#
# Numerical policy lives in exactly one place — NumericalPolicy. Strategy
# types identify the estimator only and never carry floor / constant-summary /
# max-weight fields.
#
# Distance semantics:
#   * `state.dists` — acceptance distances computed with the weights that
#     governed the completed generation; never overwritten by recomputation.
#   * `state.adaptive.epsilon_distances` — the same accepted summaries
#     recomputed at the generation boundary under the newly estimated weights;
#     this is the vector from which the epsilon schedule derives the next
#     tolerance.

using Statistics

# ── strategy types (estimator identity only — no policy fields) ───────────────

"""
    AbstractScaleStrategy

Abstract type for adaptive inverse-distance scaling strategies. A strategy
identifies *how* raw summary-scale estimates are turned into per-key inverse
distance weights. Numerical policy (floor, constant-summary handling, maximum
weight) is owned separately by [`NumericalPolicy`](@ref) — strategy types
identify the estimator only and never duplicate policy fields.
"""
abstract type AbstractScaleStrategy end

"""
    NoAdaptation <: AbstractScaleStrategy

No scaling: every inverse-distance weight is 1.0, so adaptive mode is
numerically identical to a plain non-adaptive run. The compatibility default.
"""
struct NoAdaptation <: AbstractScaleStrategy
    function NoAdaptation(; kwargs...)
        isempty(kwargs) || throw(ArgumentError(
            "NoAdaptation carries no policy fields; floor, constant_policy and " *
            "max_weight belong to NumericalPolicy"))
        return new()
    end
end

"""
    FixedScale(weights::AbstractVector{<:Real}) <: AbstractScaleStrategy

User-supplied per-key inverse-distance weights. `weights` is validated once at
construction (non-empty, finite, at least one positive entry) and is applied
in the stable summary key order owned by the runtime state.
"""
struct FixedScale <: AbstractScaleStrategy
    weights::Vector{Float64}
    function FixedScale(weights::AbstractVector{<:Real})
        w = Float64.(weights)
        isempty(w) && throw(ArgumentError("FixedScale weights must not be empty"))
        all(isfinite, w) || throw(ArgumentError("FixedScale weights must be finite"))
        any(>(0), w) || throw(ArgumentError(
            "FixedScale weights must have at least one positive entry"))
        return new(w)
    end
end

"""
    StandardDeviationScale <: AbstractScaleStrategy

Inverse-distance weight `w[k] = 1 / std(summaries[:, k])` estimated from the
previous generation's accepted summaries using the *unweighted* standard
deviation. Numerical policy (`floor`, `constant_policy`, `max_weight`) is
supplied separately via [`NumericalPolicy`](@ref) and shared through
[`AdaptiveDistanceConfig`](@ref).

Importance-weighted standard deviation and MAD-based estimators are deferred
to a later release.
"""
struct StandardDeviationScale <: AbstractScaleStrategy
    function StandardDeviationScale(; kwargs...)
        isempty(kwargs) || throw(ArgumentError(
            "StandardDeviationScale carries no policy fields; floor, " *
            "constant_policy and max_weight belong to NumericalPolicy"))
        return new()
    end
end

# ── estimator source and initializer identifiers ──────────────────────────────

"""
    PreviousGenAccepted

Estimator source: weights for generation `t + 1` are estimated from the
accepted summaries of the completed generation `t`. The only estimator source
available in this release; prior-predictive and all-simulation sources are
deferred.
"""
struct PreviousGenAccepted end

"""
    UnitInit

Initializer: generation 1 runs with every inverse-distance weight set to
`1.0` (unit scaling), so the first generation is numerically identical to a
non-adaptive run. Prior-predictive initialization is deferred.
"""
struct UnitInit end

# ── numerical policy (single source of truth) ────────────────────────────────

"""
    NumericalPolicy(; floor = 1e-8, constant_policy = :error, max_weight = 1e6)

Single source of truth for numerical policy, shared across all strategies.
Fields:
  - `floor` — lower bound applied to a raw scale estimate before inverting it
    (guards against division by near-zero variance)
  - `constant_policy` — what to do when a key has zero or non-finite scale:
    `:error` (throw), `:ignore` (weight `0.0`), or `:clamp` (weight =
    `max_weight`)
  - `max_weight` — the weight used by `constant_policy === :clamp`

Belongs to [`AdaptiveDistanceConfig`](@ref), never to strategy types.
"""
struct NumericalPolicy
    floor::Float64
    constant_policy::Symbol
    max_weight::Float64
    function NumericalPolicy(; floor::Real = 1e-8,
                             constant_policy::Symbol = :error,
                             max_weight::Real = 1e6)
        (isfinite(floor) && floor > 0) || throw(ArgumentError(
            "floor must be finite and > 0, got $floor"))
        constant_policy in (:error, :ignore, :clamp) || throw(ArgumentError(
            "constant_policy must be :error, :ignore, or :clamp, got $constant_policy"))
        (isfinite(max_weight) && max_weight > 0) || throw(ArgumentError(
            "max_weight must be finite and > 0, got $max_weight"))
        return new(Float64(floor), constant_policy, Float64(max_weight))
    end
end

# ── immutable adaptive configuration ─────────────────────────────────────────

"""
    AdaptiveDistanceConfig(strategy, [estimator_source, initializer, policy])

Immutable configuration for adaptive distance scaling — the *only* place the
four policy inputs live:

  - `strategy::AbstractScaleStrategy` — the estimator ([`NoAdaptation`](@ref),
    [`FixedScale`](@ref), [`StandardDeviationScale`](@ref))
  - `estimator_source` — [`PreviousGenAccepted`](@ref) (default; the only
    source in this release)
  - `initializer` — [`UnitInit`](@ref) (default; the only initializer in this
    release)
  - `policy::NumericalPolicy` — numerical policy (default `NumericalPolicy()`)

```julia
AdaptiveDistanceConfig(StandardDeviationScale())
AdaptiveDistanceConfig(StandardDeviationScale(), NumericalPolicy(constant_policy = :clamp))
AdaptiveDistanceConfig(StandardDeviationScale(), PreviousGenAccepted(), UnitInit(),
                       NumericalPolicy(floor = 1e-6))
```

Pass it as the `adaptive` keyword of [`run_smc!`](@ref) / [`run_smc`](@ref).
The summary *key order* is not part of the configuration — it is runtime state
owned by the sampler and derived from the observed summary.
"""
struct AdaptiveDistanceConfig{S,I}
    strategy::AbstractScaleStrategy
    estimator_source::S
    initializer::I
    policy::NumericalPolicy
    # Inner constructor carries the scope validation (this release supports
    # only PreviousGenAccepted / UnitInit) so every construction path — outer,
    # convenience, or direct `AdaptiveDistanceConfig{S,I}(...)` — runs it.
    function AdaptiveDistanceConfig{S,I}(strategy::AbstractScaleStrategy,
                                         estimator_source,
                                         initializer,
                                         policy::NumericalPolicy) where {S,I}
        estimator_source isa PreviousGenAccepted || throw(ArgumentError(
            "unsupported estimator source $(typeof(estimator_source)); this release " *
            "supports only PreviousGenAccepted (estimator source)"))
        initializer isa UnitInit || throw(ArgumentError(
            "unsupported initializer $(typeof(initializer)); this release supports " *
            "only UnitInit (initializer)"))
        return new{S,I}(strategy, estimator_source, initializer, policy)
    end
end

# Unparameterized outer constructor (the struct's inner constructor suppresses
# the default one): infers {S, I} from the arguments and validates above.
function AdaptiveDistanceConfig(strategy::AbstractScaleStrategy,
                                estimator_source, initializer,
                                policy::NumericalPolicy)
    return AdaptiveDistanceConfig{typeof(estimator_source), typeof(initializer)}(
        strategy, estimator_source, initializer, policy)
end

AdaptiveDistanceConfig(strategy::AbstractScaleStrategy) =
    AdaptiveDistanceConfig(strategy, PreviousGenAccepted(), UnitInit(), NumericalPolicy())

# `(strategy, policy)` convenience — `NumericalPolicy` is the only other type
# accepted as the second positional argument.
AdaptiveDistanceConfig(strategy::AbstractScaleStrategy, policy::NumericalPolicy) =
    AdaptiveDistanceConfig(strategy, PreviousGenAccepted(), UnitInit(), policy)

AdaptiveDistanceConfig(strategy::AbstractScaleStrategy,
                       estimator_source, initializer) =
    AdaptiveDistanceConfig(strategy, estimator_source, initializer, NumericalPolicy())

# ── mutable runtime state (internal; checkpointed) ───────────────────────────

"""
    AdaptiveDistanceState

Mutable runtime state for adaptive distance scaling. **Internal** — not
exported; inspect it read-only through `state.adaptive`. All fields have
concrete types. Fields (restart-critical, persisted with the checkpoint):

  - `key_order::Vector{Symbol}` — stable summary key order
  - `weights::Vector{Float64}` — inverse-distance weights frozen for the
    current generation
  - `gen_estimated::Int` — generation the weights were estimated from
  - `epsilon_distances::Vector{Float64}` — acceptance distances of the
    completed generation recomputed under the current weights; the vector the
    epsilon schedule derives the next tolerance from (distinct from
    `state.dists`)
  - `epsilon::Float64` — next / current derived tolerance
"""
mutable struct AdaptiveDistanceState
    key_order::Vector{Symbol}
    weights::Vector{Float64}
    gen_estimated::Int
    epsilon_distances::Vector{Float64}
    epsilon::Float64
    function AdaptiveDistanceState(key_order::Vector{Symbol},
                                   weights::AbstractVector{<:Real},
                                   gen_estimated::Int,
                                   epsilon_distances::AbstractVector{<:Real},
                                   epsilon::Real)
        isempty(key_order) && throw(ArgumentError("adaptive key_order must not be empty"))
        w = Float64.(weights)
        length(w) == length(key_order) || throw(ArgumentError(
            "adaptive weights length $(length(w)) does not match key_order length " *
            "$(length(key_order))"))
        all(isfinite, w) || throw(ArgumentError("adaptive weights must be finite"))
        any(>(0), w) || throw(ArgumentError(
            "adaptive weights must have at least one positive entry — all-zero " *
            "weights make every distance zero"))
        εd = Float64.(epsilon_distances)
        all(isfinite, εd) || throw(ArgumentError(
            "adaptive epsilon-selection distances must be finite"))
        return new(collect(Symbol, key_order), w, gen_estimated, εd, Float64(epsilon))
    end
end

# Unit initialization: generation 1 runs with unit weights (exactly the
# non-adaptive distance) until the first estimation from generation 1.
_initial_weights(::UnitInit, key_order::Vector{Symbol}) =
    ones(Float64, length(key_order))

# ── weight estimation ────────────────────────────────────────────────────────

"""
    estimate_weights(strategy, summaries::Matrix{Float64},
                     importance_weights::Vector{Float64},
                     key_order::Vector{Symbol}, policy::NumericalPolicy)
        -> Vector{Float64}

Estimate per-key inverse-distance weights from a matrix of accepted summaries
(one row per particle, one column per key in `key_order` order). Returns a
`Vector{Float64}` aligned with `key_order`; entry `k` is `1 / scale[k]`.

`importance_weights` is accepted for signature stability — the initial release
estimates scales *unweighted* (importance-weighted standard deviation is
deferred). `policy` owns the floor and the constant-summary handling.
"""
function estimate_weights(strategy::NoAdaptation,
                          summaries::Matrix{Float64},
                          _importance_weights::Vector{Float64},
                          key_order::Vector{Symbol},
                          _policy::NumericalPolicy)
    return ones(Float64, length(key_order))
end

function estimate_weights(strategy::FixedScale,
                          summaries::Matrix{Float64},
                          _importance_weights::Vector{Float64},
                          key_order::Vector{Symbol},
                          _policy::NumericalPolicy)
    length(strategy.weights) == length(key_order) || throw(ArgumentError(
        "FixedScale has $(length(strategy.weights)) weights but key_order has " *
        "$(length(key_order)) keys $(Tuple(key_order))"))
    return copy(strategy.weights)
end

function estimate_weights(strategy::StandardDeviationScale,
                          summaries::Matrix{Float64},
                          _importance_weights::Vector{Float64},
                          key_order::Vector{Symbol},
                          policy::NumericalPolicy)
    n, d = size(summaries)
    length(key_order) == d || throw(ArgumentError(
        "key_order has $(length(key_order)) keys but summaries have $d columns"))
    out = Vector{Float64}(undef, d)
    for k in 1:d
        scale = std(view(summaries, :, k))
        if scale == 0.0 || !isfinite(scale)
            # Constant (or degenerate) summary key — apply the single policy.
            if policy.constant_policy === :error
                throw(ArgumentError(
                    "summary key `:$(key_order[k])` has zero or non-finite scale " *
                    "(std = $scale); cannot compute an inverse-distance weight " *
                    "with constant_policy = :error"))
            elseif policy.constant_policy === :ignore
                out[k] = 0.0
            else  # :clamp
                out[k] = policy.max_weight
            end
        else
            # Floor is a lower bound on the raw scale before inversion.
            out[k] = 1.0 / max(scale, policy.floor)
        end
    end
    return out
end

# ── weight application ───────────────────────────────────────────────────────

"""
    apply_weights(summary::NamedTuple, weights::AbstractVector{<:Real},
                  key_order::Vector{Symbol}) -> NamedTuple

Return a copy of `summary` in which every value is multiplied by the weight of
its key (`weights[i]` for `key_order[i]`). The key *set* must match
`key_order` exactly — a missing or extra key throws `ArgumentError`, as does a
non-`Real` value. The returned summary keeps the original key order, so any
configured `cfg.distance` receives the same summary shape it always does
(pre-scaling both sides yields the weighted metric).
"""
function apply_weights(summary::NamedTuple,
                       weights::AbstractVector{<:Real},
                       key_order::Vector{Symbol})
    length(weights) == length(key_order) || throw(ArgumentError(
        "weights length $(length(weights)) does not match key_order length " *
        "$(length(key_order))"))
    length(keys(summary)) == length(key_order) || throw(ArgumentError(
        "summary keys $(Tuple(keys(summary))) do not match key_order " *
        "$(Tuple(key_order))"))
    index = Dict{Symbol,Int}(k => i for (i, k) in enumerate(key_order))
    scaled = map(keys(summary)) do k
        i = get(index, k, 0)
        i == 0 && throw(ArgumentError(
            "summary has key `:$k` not present in key_order $(Tuple(key_order))"))
        v = summary[k]
        v isa Real || throw(ArgumentError(
            "summary key `:$k` has type $(typeof(v)); only Real values can be weighted"))

        Float64(v) * weights[i]
    end
    return NamedTuple{keys(summary)}(scaled)
end

# ── adaptive distance computation ────────────────────────────────────────────

"""
    adaptive_distance(state, base_distance::Function, s_sim, s_obs) -> Float64

Distance used for acceptance. When `state.adaptive === nothing` this is
exactly `base_distance(s_sim, s_obs)` (non-adaptive runs are bit-identical to
main). When adaptive state is present, both summaries are pre-scaled by the
frozen inverse-distance weights (`state.adaptive.weights`,
`state.adaptive.key_order`) before `base_distance` is applied — for an L2 base
distance that is the adaptive weighted L2, and any other configured distance
sees the same summary shape it always does.
"""
function adaptive_distance(state, base_distance::Function, s_sim, s_obs)
    ad = state.adaptive
    ad === nothing && return base_distance(s_sim, s_obs)
    return base_distance(apply_weights(s_sim, ad.weights, ad.key_order),
                         apply_weights(s_obs, ad.weights, ad.key_order))
end# ── generation-boundary lifecycle ────────────────────────────────────────────

"""
    update_adaptive!(state, config::AdaptiveDistanceConfig,
                     base_distance::Function, summary_obs) -> state

Generation-boundary adaptation, run *after* a generation completes and *before*
the next one starts:

  1. estimate new inverse-distance weights from the completed generation's
     accepted summaries and importance weights (`PreviousGenAccepted`),
  2. recompute that generation's acceptance distances under the new weights,
  3. store both in `state.adaptive` (weights + `epsilon_distances`) and freeze
     `gen_estimated = state.gen`.

`state.dists` — the acceptance distances computed with the weights that
governed the completed generation — is deliberately **not** overwritten.
Epsilon is not derived here; the schedule receives
`state.adaptive.epsilon_distances` on its next call and the driver freezes the
result in `state.adaptive.epsilon`.

If `state.adaptive === nothing`, the runtime state is attached lazily from
`summary_obs` (key order) and `config.initializer` (unit weights), so a first
call on a non-adaptive state activates adaptive mode.

Throws `ArgumentError` on key mismatch (summaries vs `key_order`) and on
non-finite summary values.
"""
function update_adaptive!(state, config::AdaptiveDistanceConfig,
                          base_distance::Function, summary_obs)
    state.gen >= 1 || throw(ArgumentError(
        "update_adaptive! requires a completed generation (state.gen = $(state.gen))"))

    if state.adaptive === nothing
        summary_obs isa NamedTuple || throw(ArgumentError(
            "adaptive distance scaling requires NamedTuple summaries; got an " *
            "observation of type $(typeof(summary_obs))"))
        key_order = collect(Symbol, keys(summary_obs))
        state.adaptive = AdaptiveDistanceState(
            key_order,
            _initial_weights(config.initializer, key_order),
            state.gen,
            Float64[],
            state.ε)
    end
    ad = state.adaptive

    # Build the summaries matrix (n particles × d keys) in stable key order,
    # validating keys and values up front.
    n = length(state.summaries)
    d = length(ad.key_order)
    X = Matrix{Float64}(undef, n, d)
    for i in 1:n
        s = state.summaries[i]
        s isa NamedTuple || throw(ArgumentError(
            "adaptive weight estimation requires NamedTuple summaries; particle $i " *
            "has $(typeof(s))"))
        length(keys(s)) == d || throw(ArgumentError(
            "summary key mismatch for particle $i: summary has $(Tuple(keys(s))) " *
            "but key_order is $(Tuple(ad.key_order))"))
        for (j, k) in enumerate(ad.key_order)
            haskey(s, k) || throw(ArgumentError(
                "summary key mismatch for particle $i: missing key `:$k` " *
                "(key_order is $(Tuple(ad.key_order)))"))
            v = s[k]
            v isa Real || throw(ArgumentError(
                "summary key `:$k` has type $(typeof(v)); only Real values can be " *
                "used for adaptive weight estimation"))
            isfinite(v) || throw(ArgumentError(
                "summary key `:$k` of particle $i is non-finite (value $v); clean " *
                "the summaries before adaptive weight estimation"))
            X[i, j] = Float64(v)
        end
    end

    # 1. Estimate new weights for the next generation (unweighted std for now).
    new_weights = estimate_weights(config.strategy, X, state.w, ad.key_order,
                                   config.policy)
    any(>(0), new_weights) || throw(ArgumentError(
        "all inverse-distance weights are zero (every key constant under " *
        "constant_policy = :ignore); the distance would degenerate"))

    # 2. Recompute the completed generation's acceptance distances under the
    #    new weights — stored separately; `state.dists` is left untouched.
    recomputed = Vector{Float64}(undef, n)
    for i in 1:n
        recomputed[i] = base_distance(
            apply_weights(state.summaries[i], new_weights, ad.key_order),
            apply_weights(summary_obs, new_weights, ad.key_order))
    end

    # 3. Freeze the new weights for the next generation.
    ad.weights = new_weights
    ad.gen_estimated = state.gen
    ad.epsilon_distances = recomputed
    return state
end

