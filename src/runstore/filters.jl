# ============================================================================
# Backend-independent query helpers
# ============================================================================
#
# The query semantics of a run store are part of the public contract, so every
# backend shares one implementation. The helpers operate on a set of committed
# generations and on plain record fields; they never touch a backend's storage
# medium, which keeps the memory and native backends from drifting apart.

"""
    _next_writable_generation(committed::AbstractSet{Int64}) -> Int64

The generation that may be written next: generations are contiguous and start
at 1, so this is one past the latest committed generation.
"""
function _next_writable_generation(committed::AbstractSet{Int64})
    isempty(committed) && return Int64(1)
    return maximum(committed) + 1
end

"""
    _selected_generations(committed::AbstractSet{Int64}, generation) -> Vector{Int64}

Resolve the `generation` filter of a query into a sorted list of generations.

`nothing` selects every committed generation; an integer selects one; a
collection selects several. A requested generation that is not committed is an
error: a query never silently returns an empty result for a generation the
caller asked for.
"""
function _selected_generations(committed::AbstractSet{Int64}, generation)
    generation === nothing && return sort!(collect(committed))
    if generation isa Integer
        gen = _checked_int64(generation, "generation")
        gen in committed || throw(MissingGeneration(gen))
        return [gen]
    end
    gens = Int64[]
    for g in generation
        gen = _checked_int64(g, "generation")
        gen in committed || throw(MissingGeneration(gen))
        push!(gens, gen)
    end
    return sort!(unique(gens))
end

_matches_index(value::Int64, filter) =
    filter === nothing ? true : (filter isa Integer ? value == filter : value in filter)

_matches_symbol(value::Symbol, filter) =
    filter === nothing ? true : (filter isa Symbol ? value === filter : value in filter)

function _matches_failure_category(record::EvaluationRecord, filter)
    filter === nothing && return true
    category = record.failure === nothing ? nothing : record.failure.category
    category === nothing && return false
    filter isa Symbol && return category === filter
    return category in filter
end

function _checked_bound(value, name::AbstractString)
    value === nothing && return nothing
    v = Float64(value)
    isfinite(v) && v >= 0.0 || throw(ArgumentError(
        "$(name) must be finite and >= 0, got $(v)"))
    return v
end
