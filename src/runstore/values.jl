# ============================================================================
# Canonical recorded values
# ============================================================================
#
# A run-history store must be able to hold scientific values without holding
# references to arbitrary user objects. The `RecordedValue` tree is the closed
# vocabulary used for summaries and provenance:
#
#   null | Bool | Int64 | UInt64 | Float64 | String | Symbol
#        | vector | tuple | named tuple | explicitly omitted value
#
# Rules:
#   * conversion is deterministic and recursively validated;
#   * nonfinite numbers fail unless represented by a tagged sentinel;
#   * unsupported values fail by default;
#   * an explicit omission is visible, typed, and carries a reason;
#   * arbitrary user objects are never stored implicitly.

"""
    RecordedValue

Abstract supertype of the canonical, closed value tree used by the run-history
store.

Concrete variants:

| variant | payload |
|---|---|
| [`RecordedNull`](@ref) | `nothing` |
| [`RecordedBool`](@ref) | `Bool` |
| [`RecordedInt`](@ref) | `Int64` |
| [`RecordedUInt`](@ref) | `UInt64` |
| [`RecordedFloat`](@ref) | finite `Float64` |
| [`RecordedString`](@ref) | `String` |
| [`RecordedSymbol`](@ref) | `Symbol` |
| [`RecordedVector`](@ref) | ordered sequence of `RecordedValue` |
| [`RecordedTuple`](@ref) | ordered sequence of `RecordedValue` |
| [`RecordedNamedTuple`](@ref) | named sequence of `RecordedValue` |
| [`RecordedOmitted`](@ref) | a value that was deliberately not recorded |
| [`RecordedSentinel`](@ref) | tagged nonfinite scientific sentinel |

Convert ordinary Julia values with [`recorded_value`](@ref).
"""
abstract type RecordedValue end

"""
    RecordedNull()

The canonical representation of `nothing`.
"""
struct RecordedNull <: RecordedValue end

"""
    RecordedBool(value::Bool)

The canonical representation of a boolean.
"""
struct RecordedBool <: RecordedValue
    value::Bool
end

"""
    RecordedInt(value::Int64)

The canonical representation of a signed integer.
"""
struct RecordedInt <: RecordedValue
    value::Int64
end

"""
    RecordedUInt(value::UInt64)

The canonical representation of an unsigned integer.
"""
struct RecordedUInt <: RecordedValue
    value::UInt64
end

"""
    RecordedFloat(value::Float64)

The canonical representation of a floating-point number.

Only finite values are allowed; use [`RecordedSentinel`](@ref) for a
deliberately recorded nonfinite number.
"""
struct RecordedFloat <: RecordedValue
    value::Float64
end

"""
    RecordedString(value::String)

The canonical representation of text.
"""
struct RecordedString <: RecordedValue
    value::String
end

"""
    RecordedSymbol(value::Symbol)

The canonical representation of a symbol.
"""
struct RecordedSymbol <: RecordedValue
    value::Symbol
end

"""
    RecordedVector(values::Vector{RecordedValue})

The canonical representation of an ordered sequence.
"""
struct RecordedVector <: RecordedValue
    values::Vector{RecordedValue}
end

"""
    RecordedTuple(values::Vector{RecordedValue})

The canonical representation of an ordered, unnamed tuple.
"""
struct RecordedTuple <: RecordedValue
    values::Vector{RecordedValue}
end

"""
    RecordedNamedTuple(names::Vector{Symbol}, values::Vector{RecordedValue})

The canonical representation of a named tuple. `names` and `values` have the
same length.
"""
struct RecordedNamedTuple <: RecordedValue
    names::Vector{Symbol}
    values::Vector{RecordedValue}
end

"""
    RecordedOmitted(type_name::String, reason::String)

A value that was deliberately **not** recorded.

An omission is visible: it carries the type name of the omitted value and a
human-readable reason. Omissions are reported by
[`validate_store`](@ref) and are never silently dropped.
"""
struct RecordedOmitted <: RecordedValue
    type_name::String
    reason::String
end

"""
    RECORDED_SENTINEL_KINDS

The permitted kinds of the tagged scientific sentinel
[`RecordedSentinel`](@ref): `:nan`, `:positive_infinity`, and
`:negative_infinity`.
"""
const RECORDED_SENTINEL_KINDS = (:nan, :positive_infinity, :negative_infinity)

"""
    RecordedSentinel(kind::Symbol)

A tagged, explicitly recorded nonfinite number.

`kind` is one of [`RECORDED_SENTINEL_KINDS`](@ref). The sentinel is the only
way a nonfinite number enters a recorded value tree: ordinary nonfinite
numbers are rejected by [`recorded_value`](@ref).
"""
struct RecordedSentinel <: RecordedValue
    kind::Symbol

    function RecordedSentinel(kind::Symbol)
        kind in RECORDED_SENTINEL_KINDS || throw(ArgumentError(
            "RecordedSentinel kind must be one of $(RECORDED_SENTINEL_KINDS), " *
            "got $(repr(kind))"))
        return new(kind)
    end
end

# --- equality ---------------------------------------------------------------
# Structural equality on the payload, so that a recorded tree round-trips
# through `==` regardless of how it was built.

Base.:(==)(a::RecordedNull, b::RecordedNull) = true
Base.:(==)(a::RecordedBool, b::RecordedBool) = a.value == b.value
Base.:(==)(a::RecordedInt, b::RecordedInt) = a.value == b.value
Base.:(==)(a::RecordedUInt, b::RecordedUInt) = a.value == b.value
Base.:(==)(a::RecordedFloat, b::RecordedFloat) = a.value == b.value
Base.:(==)(a::RecordedString, b::RecordedString) = a.value == b.value
Base.:(==)(a::RecordedSymbol, b::RecordedSymbol) = a.value == b.value
Base.:(==)(a::RecordedVector, b::RecordedVector) = a.values == b.values
Base.:(==)(a::RecordedTuple, b::RecordedTuple) = a.values == b.values
Base.:(==)(a::RecordedNamedTuple, b::RecordedNamedTuple) =
    a.names == b.names && a.values == b.values
Base.:(==)(a::RecordedOmitted, b::RecordedOmitted) =
    a.type_name == b.type_name && a.reason == b.reason
Base.:(==)(a::RecordedSentinel, b::RecordedSentinel) = a.kind == b.kind

# --- copying ----------------------------------------------------------------
# Records are immutable, but the payloads of the container variants are
# mutable vectors. Copies are therefore made whenever a value leaves the
# store, so that a caller can never mutate stored state.

Base.copy(v::RecordedNull) = v
Base.copy(v::RecordedBool) = v
Base.copy(v::RecordedInt) = v
Base.copy(v::RecordedUInt) = v
Base.copy(v::RecordedFloat) = v
Base.copy(v::RecordedString) = v
Base.copy(v::RecordedSymbol) = v
Base.copy(v::RecordedVector) = RecordedVector(copy(v.values))
Base.copy(v::RecordedTuple) = RecordedTuple(copy(v.values))
Base.copy(v::RecordedNamedTuple) = RecordedNamedTuple(copy(v.names), copy(v.values))
Base.copy(v::RecordedOmitted) = v
Base.copy(v::RecordedSentinel) = v

# --- conversion -------------------------------------------------------------

"""
    recorded_value(x; omit::Bool=false, reason=nothing) -> RecordedValue

Convert `x` into the canonical [`RecordedValue`](@ref) tree.

Supported inputs are `nothing`, `Bool`, integers (checked against the
`Int64`/`UInt64` ranges), floating-point numbers, `String`, `Symbol`,
one-dimensional arrays, tuples, and named tuples. Values that are already
`RecordedValue` are returned unchanged, which is how an explicit
[`RecordedSentinel`](@ref) or [`RecordedOmitted`](@ref) is placed in a tree.

Nonfinite numbers are rejected with [`NonfiniteRecordedValue`](@ref). Values of
any other type are rejected with [`UnsupportedRecordedValue`](@ref).

With `omit=true`, an unsupported value is recorded as
[`RecordedOmitted`](@ref) carrying its type name and `reason` instead of
failing. Omission is always explicit and always visible. `omit=true` does not
rescue a nonfinite number: a nonfinite value is represented explicitly with a
[`RecordedSentinel`](@ref), never by omission.

# Examples
```julia
recorded_value([1.0, 2.0])            # RecordedVector
recorded_value((a=1, b="x"))          # RecordedNamedTuple
recorded_value(1.0, omit=true)        # RecordedFloat (supported: not omitted)
recorded_value(Dict(:a => 1))         # UnsupportedRecordedValue
```
"""
function recorded_value(x; omit::Bool=false, reason::Union{Nothing,AbstractString}=nothing)
    return _recorded_value(x, omit, reason === nothing ? "" : String(reason))
end

_recorded_value(x::RecordedValue, omit::Bool, reason::String) = x
_recorded_value(x::Nothing, omit::Bool, reason::String) = RecordedNull()
_recorded_value(x::Bool, omit::Bool, reason::String) = RecordedBool(x)

function _recorded_value(x::Integer, omit::Bool, reason::String)
    if x isa Unsigned
        x <= typemax(UInt64) || throw(UnsupportedRecordedValue(
            string(typeof(x)),
            "unsigned integer does not fit in UInt64"))
        return RecordedUInt(UInt64(x))
    end
    if typemin(Int64) <= x <= typemax(Int64)
        return RecordedInt(Int64(x))
    elseif x >= 0 && x <= typemax(UInt64)
        return RecordedUInt(UInt64(x))
    end
    throw(UnsupportedRecordedValue(
        string(typeof(x)),
        "integer does not fit in Int64 or UInt64"))
end

function _recorded_value(x::AbstractFloat, omit::Bool, reason::String)
    f = try
        Float64(x)
    catch
        throw(UnsupportedRecordedValue(
            string(typeof(x)), "value cannot be represented as Float64"))
    end
    isfinite(f) || throw(NonfiniteRecordedValue(string(typeof(x)), string(f)))
    return RecordedFloat(f)
end

_recorded_value(x::AbstractString, omit::Bool, reason::String) = RecordedString(String(x))
_recorded_value(x::Symbol, omit::Bool, reason::String) = RecordedSymbol(x)

function _recorded_value(x::AbstractVector, omit::Bool, reason::String)
    return RecordedVector(RecordedValue[_recorded_value(v, omit, reason) for v in x])
end

function _recorded_value(x::Tuple, omit::Bool, reason::String)
    return RecordedTuple(RecordedValue[_recorded_value(v, omit, reason) for v in x])
end

function _recorded_value(x::NamedTuple, omit::Bool, reason::String)
    names = collect(Symbol, keys(x))
    vals = RecordedValue[_recorded_value(v, omit, reason) for v in Base.values(x)]
    return RecordedNamedTuple(names, vals)
end

function _recorded_value(x::Missing, omit::Bool, reason::String)
    return _unsupported(
        x, omit, reason,
        "missing values are not part of the canonical value tree; use `nothing` for null")
end

function _recorded_value(x::Any, omit::Bool, reason::String)
    return _unsupported(x, omit, reason, "value type is not part of the canonical value tree")
end

function _unsupported(x, omit::Bool, reason::String, default_reason::String)
    if omit
        r = isempty(reason) ? default_reason : reason
        return RecordedOmitted(string(typeof(x)), r)
    end
    throw(UnsupportedRecordedValue(
        string(typeof(x)),
        isempty(reason) ? default_reason : reason))
end

# --- provenance -------------------------------------------------------------

"""
    RecordedProvenance

Canonical, ordered provenance entries of a run-history record.

Each entry is a `String` key bound to a [`RecordedValue`](@ref). Keys must be
nonempty and unique. Provenance is a closed tree: arbitrary user objects are
never stored implicitly.

Build it from a `NamedTuple` or from explicit key/value pairs:

```julia
RecordedProvenance((model="toy", revision=3))
RecordedProvenance(["model" => "toy", "revision" => 3])
```
"""
struct RecordedProvenance
    entries::Vector{Pair{String,RecordedValue}}

    function RecordedProvenance(entries::Vector{Pair{String,RecordedValue}})
        seen = Set{String}()
        for (k, _) in entries
            isempty(k) && throw(ArgumentError(
                "RecordedProvenance keys must be nonempty strings"))
            k in seen && throw(ArgumentError(
                "RecordedProvenance keys must be unique; duplicate key $(repr(k))"))
            push!(seen, k)
        end
        return new(entries)
    end
end

RecordedProvenance() = RecordedProvenance(Pair{String,RecordedValue}[])

function RecordedProvenance(entries::AbstractVector{<:Pair};
                             omit::Bool=false,
                             reason::Union{Nothing,AbstractString}=nothing)
    canonical = Pair{String,RecordedValue}[]
    for (k, v) in entries
        canonicalize = recorded_value(v; omit=omit, reason=reason)
        push!(canonical, String(k) => canonicalize)
    end
    return RecordedProvenance(canonical)
end

function RecordedProvenance(nt::NamedTuple;
                            omit::Bool=false,
                            reason::Union{Nothing,AbstractString}=nothing)
    entries_list = Pair{String,Any}[string(k) => v for (k, v) in pairs(nt)]
    return RecordedProvenance(entries_list; omit=omit, reason=reason)
end

Base.:(==)(a::RecordedProvenance, b::RecordedProvenance) = a.entries == b.entries

Base.copy(p::RecordedProvenance) =
    RecordedProvenance(Pair{String,RecordedValue}[k => copy(v) for (k, v) in p.entries])

"""
    recorded_provenance(x; omit::Bool=false, reason=nothing) -> RecordedProvenance

Convert provenance into the canonical [`RecordedProvenance`](@ref) form.

A `NamedTuple` becomes one entry per field (field order is preserved), and an
existing `RecordedProvenance` is returned unchanged. Any other provenance
object is rejected: the run-history store never stores an arbitrary
provenance object implicitly.
"""
function recorded_provenance(x::RecordedProvenance; kwargs...)
    return x
end

function recorded_provenance(x::NamedTuple; kwargs...)
    return RecordedProvenance(x; kwargs...)
end

function recorded_provenance(x; kwargs...)
    throw(ArgumentError(
        "provenance must be a NamedTuple or a RecordedProvenance to be " *
        "recorded canonically; got $(typeof(x))"))
end
