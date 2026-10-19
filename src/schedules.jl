"""
    epsilon_quantile(distances::AbstractVector{<:Real}, q::Real,
                     ε_cap::Real=Inf) -> Float64

Adaptive tolerance: the `q`-quantile of the supplied accepted distances.

Typical choice q = 0.5 (median), used to define ε_{t+1} from population t.

# Monotonicity and the `ε_cap` guard

The monotonicity story depends on the distance source:

- **Non-adaptive mode** (`state.dists`): monotone non-increasing by
  construction — every stored distance is ≤ the tolerance that accepted it,
  so any quantile of them is ≤ the previous ε.
- **Adaptive mode** (`state.adaptive.epsilon_selection_distances`): the
  vector is *recomputed* at the generation boundary under the newly
  estimated weights. Recomputed distances are **not** bounded by the
  previous tolerance, so monotonicity does not follow from construction
  alone — the explicit `min(ε_new, ε_cap)` guard is load-bearing there
  (it is what keeps ε from growing past the caller's cap), not merely
  belt-and-braces. It remains belt-and-braces in non-adaptive mode.

# Distance source

- Non-adaptive mode: pass `state.dists` (acceptance distances from the
  completed generation).
- Adaptive mode: pass `state.adaptive.epsilon_selection_distances` (recomputed at the
  generation boundary under newly estimated weights).
"""
function epsilon_quantile(distances::AbstractVector{<:Real}, q::Real,
                          ε_cap::Real = Inf)
    (0 < q <= 1) || throw(ArgumentError("q must be in (0, 1], got $q"))
    ε_new = quantile(distances, q)
    return min(ε_new, ε_cap)
end

"""
    epsilon_quantile(state::ABCSMCState, q::Real) -> Float64

Backward-compatible wrapper: uses `state.dists` (acceptance distances) and
caps at `state.ε`. Prefer [`epsilon_quantile`](@ref)
when you have a specific distance vector.
"""
function epsilon_quantile(state::ABCSMCState, q::Real)
    return epsilon_quantile(state.dists, q, state.ε)
end

"""
    QuantileSchedule(q)

Callable tolerance schedule for `run_smc!`: each generation uses the `q`-quantile
of the previous population's distances. Toni et al. (2009) used a fixed schedule;
quantile adaptation is the standard post‑Toni optimization and converges to
small ε without manual tuning.

# Call signature

`(sched::QuantileSchedule)(state, t, distances) -> ε_t`

- `distances` is the vector from which ε is derived:
  - Non-adaptive mode: `state.dists`
  - Adaptive mode: `state.adaptive.epsilon_selection_distances`
- The 2-arg form `(state, t)` is preserved for backward compatibility and uses
  `state.dists`.

`FixedSchedule` ignores the supplied distances.
"""
struct QuantileSchedule
    q::Float64
    function QuantileSchedule(q::Real)
        (0 < q <= 1) || throw(ArgumentError("q must be in (0, 1], got $q"))
        return new(Float64(q))
    end
end

# 3-arg: derive ε from the supplied distances (adaptive or non-adaptive).
function (sched::QuantileSchedule)(state::ABCSMCState,
                                   t::Int,
                                   distances::AbstractVector{<:Real})
    return epsilon_quantile(distances, sched.q, state.ε)
end

# 2-arg: backward compatibility — uses state.dists.
(sched::QuantileSchedule)(state::ABCSMCState, t::Int) =
    epsilon_quantile(state, sched.q)

"""
    FixedSchedule(εs::AbstractVector{<:Real})

Callable tolerance schedule `(state, t) -> εs[t]` for `run_smc!`. Toni et al. (2009) style hand-picked schedule; must be non-increasing.
Generation 1 uses `state.ε` (the initial tolerance), so index `t` ≥ 2 entries
are consumed for a T-generation run started from gen 0 — pass the full
intended schedule `ε_1, ε_2, …, ε_T` and it is indexed directly by `t`.

# Call signature

`(sched::FixedSchedule)(state, t, distances) -> ε_t`

The `distances` argument is accepted for interface uniformity but ignored —
`FixedSchedule` is fully prescribed. The 2-arg form `(state, t)` is preserved
for backward compatibility.
"""
struct FixedSchedule
    ε::Vector{Float64}
    function FixedSchedule(εs::AbstractVector{<:Real})
        issorted(εs; rev = true) ||
            throw(ArgumentError("ε schedule must be non-increasing, got $εs"))
        return new(collect(Float64, εs))
    end
end

# 3-arg: distances accepted but ignored.
function (sched::FixedSchedule)(state::ABCSMCState,
                                t::Int,
                                _distances::AbstractVector{<:Real})
    checkbounds(sched.ε, t)
    return sched.ε[t]
end

# 2-arg: backward compatibility.
function (sched::FixedSchedule)(state::ABCSMCState, t::Int)
    checkbounds(sched.ε, t)
    return sched.ε[t]
end
