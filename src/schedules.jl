"""
    epsilon_quantile(state::ABCSMCState, q::Real) -> Float64

Adaptive tolerance: the `q`-quantile of the current population's accepted
distances `d(s_i, s_obs)` (cached in `state.dists`).

Monotone non-increasing **by construction**: every stored distance is ≤ the
tolerance that accepted it, so any quantile of them is ≤ `state.ε`. An
explicit `min` guard is kept as belt-and-braces.

Typical choice q = 0.5 (median), used to define ε_{t+1} from population t.
"""
function epsilon_quantile(state::ABCSMCState, q::Real)
    (0 < q <= 1) || throw(ArgumentError("q must be in (0, 1], got $q"))
    ε_new = quantile(state.dists, q)
    return min(ε_new, state.ε)
end

"""
    QuantileSchedule(q)

Callable tolerance schedule `(state, t) -> ε_t` for `run_smc!`: each generation
uses the `q`-quantile of the previous population's distances. Toni et al. (2009)
used a fixed schedule; quantile adaptation is the standard post‑Toni
optimization and converges to small ε without manual tuning.
"""
struct QuantileSchedule
    q::Float64
    function QuantileSchedule(q::Real)
        (0 < q <= 1) || throw(ArgumentError("q must be in (0, 1], got $q"))
        return new(Float64(q))
    end
end

(sched::QuantileSchedule)(state::ABCSMCState, t::Int) = epsilon_quantile(state, sched.q)

"""
    FixedSchedule(εs::AbstractVector{<:Real})

Callable tolerance schedule `(state, t) -> εs[t]` for `run_smc!`.
Toni et al. (2009) style hand-picked schedule; must be non-increasing.
Generation 1 uses `state.ε` (the initial tolerance), so index `t` ≥ 2 entries
are consumed for a T-generation run started from gen 0 — pass the full
intended schedule `ε_1, ε_2, …, ε_T` and it is indexed directly by `t`.
"""
struct FixedSchedule
    ε::Vector{Float64}
    function FixedSchedule(εs::AbstractVector{<:Real})
        issorted(εs; rev = true) ||
            throw(ArgumentError("ε schedule must be non-increasing, got $εs"))
        return new(collect(Float64, εs))
    end
end

function (sched::FixedSchedule)(state::ABCSMCState, t::Int)
    checkbounds(sched.ε, t)
    return sched.ε[t]
end