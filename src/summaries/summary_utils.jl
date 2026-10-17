# src/summaries/summary_utils.jl
#
# Small, dependency-free helpers shared by the summary types: flattening nested
# NamedTuples, converting a named summary to the numeric vector a distance
# function consumes (and back), and checking that a summary produced the keys it
# declared. Everything here is deterministic.

"""
    flatten_namedtuple(nt::NamedTuple) -> NamedTuple

Flatten a NamedTuple whose values may themselves be NamedTuples into a single
flat NamedTuple, joining nested keys with `_`:

```julia
flatten_namedtuple((a = 1, b = (x = 2, y = 3)))   # (a = 1, b_x = 2, b_y = 3)
```

Values that are not NamedTuples (scalars, arrays, …) pass through untouched.
Duplicate keys after flattening throw an `ArgumentError` rather than silently
dropping a value — this is what makes [`CompositeSummary`](@ref) safe when one
of its parts returns a structured value.
"""
function flatten_namedtuple(nt::NamedTuple)
    keys_out = Symbol[]
    vals_out = Any[]
    for (k, v) in pairs(nt)
        if v isa NamedTuple
            for (k2, v2) in pairs(flatten_namedtuple(v))
                push!(keys_out, Symbol(k, :_, k2))
                push!(vals_out, v2)
            end
        else
            push!(keys_out, k)
            push!(vals_out, v)
        end
    end

    seen = Set{Symbol}()
    for k in keys_out
        k in seen && throw(ArgumentError(
            "duplicate summary key `$k` after flattening; " *
            "rename one of the summaries (keys: $(Tuple(keys_out)))"))
        push!(seen, k)
    end

    return NamedTuple{Tuple(keys_out)}(Tuple(vals_out))
end

"""
    namedtuple_to_vector(nt::NamedTuple) -> Vector{Float64}

The values of `nt` as a `Vector{Float64}`, in key order — the bridge from a
named summary to the numeric vector that a distance function consumes. Every
value must be `Real`; anything else (an array, a string, a nested NamedTuple)
raises an `ArgumentError` naming the offending key, because it has no single
numeric value to compare against the observed summary.
"""
function namedtuple_to_vector(nt::NamedTuple)
    out = Vector{Float64}(undef, length(nt))
    for (i, (k, v)) in enumerate(pairs(nt))
        v isa Real || throw(ArgumentError(
            "summary key `$k` has type $(typeof(v)); only Real values can be " *
            "converted to a numeric distance vector — reduce arrays to scalars " *
            "(a mean, a variance, …) first"))
        out[i] = Float64(v)
    end
    return out
end

"""
    vector_to_namedtuple(v::AbstractVector, expected) -> NamedTuple

Rebuild a NamedTuple from a vector of values and a key order — the inverse of
[`namedtuple_to_vector`](@ref). Throws if the lengths disagree.
"""
function vector_to_namedtuple(v::AbstractVector, expected)
    length(v) == length(expected) || throw(ArgumentError(
        "got $(length(v)) values for $(length(expected)) keys: $(Tuple(expected))"))
    ks = Tuple(Symbol.(expected))
    return NamedTuple{ks}(Tuple(Float64.(v)))
end

"""
    validate_summary_keys(nt::NamedTuple, expected::AbstractVector{Symbol}) -> expected
    validate_summary_keys(nt::NamedTuple, summary::AbstractSummary) -> expected

Throw an `ArgumentError` unless the keys of `nt` are exactly `expected`, in
order; otherwise return `expected`. Used to check that a summary function
returned what its declaration promises before the values are combined or
projected.
"""
function validate_summary_keys(nt::NamedTuple, expected::AbstractVector{Symbol})
    got = collect(keys(nt))
    exp = collect(expected)
    got == exp || throw(ArgumentError(
        "summary returned keys $(Tuple(got)), expected $(Tuple(exp))"))
    return expected
end

validate_summary_keys(nt::NamedTuple, s::AbstractSummary) =
    validate_summary_keys(nt, summary_keys(s))
