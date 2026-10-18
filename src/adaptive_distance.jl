# src/adaptive_distance.jl
#
# Adaptive inverse-distance scaling for ABC-SMC.
#
# Architecture:
#   * Immutable configuration (AdaptiveDistanceConfig): strategy, estimator
#     source, initializer, numerical policy. Set once at construction.
#   * Mutable runtime state (AdaptiveDistanceState): key order, current
#     inverse-distance weights, generation estimated, epsilon-selection
#     distances, next epsilon. Checkpointed separately.
#
# Distance semantics:
#   * state.dists — acceptance distances computed with the weights governing
#     the completed generation (unchanged semantics).
#   * adaptive epsilon-selection distances — recomputed at the generation
#     boundary under newly estimated weights; distinct from state.dists.
#   * cfg.distance is never mutated; no anonymous closures are installed per
#     generation. The engine calls `adaptive_distance(state, base_distance,
#     s_sim, s_obs)` which applies frozen weights.

using Statistics

# ── immutable strategy configuration ─────────────────────────────────────────

"""
    AbstractScaleStrategy

Abstract type for adaptive inverse-distance scaling strategies. A strategy
identifies *how* raw summary-scale estimates are turned into per-key inverse
distance weights. Numerical policy (floor, constant-summary handling, max
weight) is owned separately by [`NumericalPolicy`](@ref) — strategy types
identify the estimator only and never duplicate policy fields.
"""
abstract type AbstractScaleStrategy end

"""
    NoAdaptation <: AbstractScaleStrategy

No scaling: every inverse-distance weight is 1.0. Identical to current main
behavior. The compatibility default.
"""
struct NoAdaptation <: AbstractScaleStrategy end

"""
    FixedScale(weights::AbstractVector{<:Real}) <: AbstractScaleStrategy

User-supplied per-key inverse-distance weights. `weights` is validated once at
construction (finite, positive, non-empty) and stored aligned to the stable
summary key order.
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
previous generation's accepted summaries using the unweighted standard
deviation. Numerical policy (`floor`, `constant_policy`, `max_weight`) is
supplied separately via [`NumericalPolicy`](@ref) and shared through
[`AdaptiveDistanceConfig`](@ref).

Importance-weighted standard deviation is deferred to a later release.
"""
struct StandardDeviationScale <: AbstractScaleStrategy end

# ── numerical policy (single source of truth) ────────────────────────────────

"""
    NumericalPolicy

Single source of truth for numerical policy shared across strategy types.
Fields:
  - `floor` — lower bound on raw scale estimates (guards against division by
    zero or near-zero variance)
  - `constant_policy` — `:error` (throw), `:ignore` (weight 0), or `:clamp`
    (weight = max_weight) when a key has zero variance
  - `max_weight` — weight assigned to a constant key when
    `constant_policy === :clamp`
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

# ── immutable adaptive distance configuration ────────────────────────────────

"""
    AdaptiveDistanceConfig

Immutable configuration for adaptive distance scaling. Set once at construction
and never mutated.

Fields:
  - `strategy::AbstractScaleStrategy` — estimator strategy
  - `key_order::Vector{Symbol}` — stable summary key order
  - `estimator_source::Symbol` — `:previous_gen_accepted` (initial release)
  - `policy::NumericalPolicy` — numerical policy (single source of truth)
"""
struct AdaptiveDistanceConfig
    strategy::AbstractScaleStrategy
    key_order::Vector{Symbol}
    estimator_source::Symbol
    policy::NumericalPolicy
    function AdaptiveDistanceConfig(strategy::AbstractScaleStrategy,
                                    key_order::Vector{Symbol},
                                    estimator_source::Symbol = :previous_gen_accepted;
                                    policy::NumericalPolicy = NumericalPolicy())
        isempty(key_order) && throw(ArgumentError(
            "key_order must not be empty"))
        estimator_source in (:previous_gen_accepted, :prior_predictive,
                             :all_simulations) || throw(ArgumentError(
            "estimator_source must be :previous_gen_accepted, :prior_predictive, " *
            "or :all_simulations, got $estimator_source"))
        return new(strategy, collect(Symbol, key_order), estimator_source, policy)
    end
end

# ── mutable runtime state (checkpointed) ─────────────────────────────────────

"""
    AdaptiveDistanceState

Mutable runtime state for adaptive distance scaling. Checkpointed separately
from the strategy configuration.

Fields:
  - `weights::Vector{Float64}` — current inverse-distance weights per key
  - `gen_estimated::Int` — generation these weights were estimated from
  - `epsilon_distances::Vector{Float64}` — recomputed distances used for
    epsilon selection (distinct from `state.dists`)
  - `epsilon::Float64` — derived epsilon, frozen for the next generation
"""
mutable struct AdaptiveDistanceState
    weights::Vector{Float64}
    gen_estimated::Int
    epsilon_distances::Vector{Float64}
    epsilon::Float64
    function AdaptiveDistanceState(weights::AbstractVector{<:Real},
                                   gen_estimated::Int,
                                   epsilon_distances::AbstractVector{<:Real},
                                   epsilon::Real)
        w = Float64.(weights)
        isempty(w) && throw(ArgumentError("adaptive weights must not be empty"))
        all(isfinite, w) || throw(ArgumentError(
            "adaptive weights must be finite"))
        any(>(0), w) || throw(ArgumentError(
            "adaptive weights must have at least one positive entry"))
        return new(w, Int(gen_estimated), Float64.(epsilon_distances),
                   Float64(epsilon))
    end
end

# ── weight estimation ────────────────────────────────────────────────────────

"""
    estimate_weights(strategy::AbstractScaleStrategy,
                     summaries::Matrix{Float64},
                     importance_weights::Vector{Float64},
                     key_order::Vector{Symbol},
                     policy::NumericalPolicy) -> Vector{Float64}

Estimate per-key inverse-distance weights from a matrix of accepted summaries
(one row per particle, one column per key in `key_order` order) and the
corresponding importance weights.

Returns a `Vector{Float64}` aligned with `key_order`. Each entry is
`1 / scale[k]` where `scale[k]` is the robust scale estimate for key `k`.
"""
function estimate_weights(strategy::NoAdaptation,
                          summaries::Matrix{Float64},
                          _importance_weights::Vector{Float64},
                          _key_order::Vector{Symbol},
                          _policy::NumericalPolicy)
    return ones(Float64, size(summaries, 2))
end

function estimate_weights(strategy::FixedScale,
                          summaries::Matrix{Float64},
                          _importance_weights::Vector{Float64},
                          _key_order::Vector{Symbol},
                          _policy::NumericalPolicy)
    length(strategy.weights) == size(summaries, 2) || throw(ArgumentError(
        "FixedScale has $(length(strategy.weights)) weights but summaries have " *
        "$(size(summaries, 2)) keys"))
    return copy(strategy.weights)
end

function estimate_weights(strategy::StandardDeviationScale,
                          summaries::Matrix{Float64},
                          _importance_weights::Vector{Float64},
                          _key_order::Vector{Symbol},
                          policy::NumericalPolicy)
    n, d = size(summaries)
    out = Vector{Float64}(undef, d)
    for k in 1:d
        x = view(summaries, :, k)
        scale = std(x)
        # Floor guards against zero or near-zero variance.
        if scale <= 0 || !isfinite(scale) || scale < policy.floor
            if policy.constant_policy === :error
                throw(ArgumentError(
                    "summary key has zero or non-finite variance; "
                    * "cannot compute inverse-distance weight"))
            elseif policy.constant_policy === :ignore
                out[k] = 0.0
                continue
            elseif policy.constant_policy === :clamp
                out[k] = policy.max_weight
                continue
            end
        end
        out[k] = 1.0 / scale
    end
    return out
end

# ── weight application ───────────────────────────────────────────────────────

"""
    apply_weights(summary::NamedTuple, weights::AbstractVector{<:Real},
                  key_order::Vector{Symbol}) -> Vector{Float64}

Convert a named summary to a `Vector{Float64}` aligned with `key_order` and
multiply elementwise by `weights`. Keys absent from `summary` throw.
"""
function apply_weights(summary::NamedTuple,
                       weights::AbstractVector{<:Real},
                       key_order::Vector{Symbol})
    length(weights) == length(key_order) || throw(ArgumentError(
        "weights length $(length(weights)) does not match key_order length " *
        "$(length(key_order))"))
    vec = Vector{Float64}(undef, length(key_order))
    for (i, k) in enumerate(key_order)
        haskey(summary, k) || throw(ArgumentError(
            "summary is missing key `:$(k)` required by key_order"))
        v = summary[k]
        v isa Real || throw(ArgumentError(
            "summary key `:$(k)` has type $(typeof(v)); only Real values can be " *
            "weighted"))
        vec[i] = Float64(v) * weights[i]
    end
    return vec
end

# ── adaptive distance computation ────────────────────────────────────────────

"""
    adaptive_distance(state, base_distance::Function,
                      s_sim::NamedTuple, s_obs::NamedTuple) -> Float64

Compute the base distance between `s_sim` and `s_obs` after applying the
frozen inverse-distance weights from `state.adaptive`. When `state.adaptive`
is `nothing`, falls back to `base_distance(s_sim, s_obs)` unchanged.
"""
function adaptive_distance(state, base_distance::Function,
                           s_sim::NamedTuple, s_obs::NamedTuple)
    ad = state.adaptive
    if ad === nothing
        return base_distance(s_sim, s_obs)
    end
    return base_distance(apply_weights(s_sim, ad.weights, ad.key_order),
                         apply_weights(s_obs, ad.weights, ad.key_order))
end

# ── lifecycle ────────────────────────────────────────────────────────────────

"""
    update_adaptive!(state::ABCSMCState, config::AdaptiveDistanceConfig,
                     base_distance::Function,
                     summary_obs::NamedTuple) -> Nothing

Generation-boundary adaptation. Estimates new inverse-distance weights from the
current generation's accepted summaries and importance weights, recomputes the
current generation's accepted-particle distances under the new weights, and
freezes both for the next generation.

Does NOT modify `state.dists` (acceptance distances remain as computed with the
previous generation's weights). Does NOT derive epsilon — that is the schedule's
job; it receives `state.adaptive.epsilon_distances` on the next call.

The recomputed distances are stored in `state.adaptive.epsilon_distances` and
the new weights in `state.adaptive.weights`. `state.adaptive.epsilon` is left
unchanged until the schedule sets it.
"""
function update_adaptive!(state, config::AdaptiveDistanceConfig,
                          base_distance::Function,
                          summary_obs::NamedTuple)
    ad = state.adaptive
    ad === nothing && return nothing

    n = length(state.summaries)
    # Build the summaries matrix (n particles × d keys) in key_order.
    X = Matrix{Float64}(undef, n, length(config.key_order))
    for i in 1:n
        for (j, k) in enumerate(config.key_order)
            v = state.summaries[i][k]
            v isa Real || throw(ArgumentError(
                "summary key `:$(k)` has type $(typeof(v)); only Real values " *
                "can be used for adaptive weight estimation"))
            X[i, j] = Float64(v)
        end
    end

    # 1. Estimate new weights from current accepted summaries.
    new_weights = estimate_weights(config.strategy, X, state.w,
                                   config.key_order, config.policy)

    # 2. Recompute current generation's distances under new weights.
    recomputed = Vector{Float64}(undef, n)
    for i in 1:n
        recomputed[i] = base_distance(
            apply_weights(state.summaries[i], new_weights, config.key_order),
            apply_weights(summary_obs, new_weights, config.key_order))
    end


    # 4. Freeze.
    ad.weights = new_weights
    ad.gen_estimated = state.gen
    ad.epsilon_distances = recomputed
    return nothing
end

