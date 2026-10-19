
# Permitted evaluation statuses.
const EVALUATION_STATUS_SUCCESS = :success
const EVALUATION_STATUS_FAILED = :failed
const EVALUATION_STATUS_NONFINITE = :nonfinite
const EVALUATION_STATUS_TIMEOUT = :timeout
const EVALUATION_STATUSES = (
    EVALUATION_STATUS_SUCCESS,
    EVALUATION_STATUS_FAILED,
    EVALUATION_STATUS_NONFINITE,
    EVALUATION_STATUS_TIMEOUT,
)

"""
    EvaluationResult{S,P}

Structured outcome of one simulator evaluation.

Fields:
  - `context::EvaluationContext` — evaluation identity
  - `status::Symbol`             — `:success`, `:failed`, `:nonfinite`,
    or `:timeout`
  - `summary::Union{Nothing,S}`  — the evaluated summary (`nothing` unless
    `:success`)
  - `provenance::P`              — caller-supplied provenance metadata
  - `failure::Union{Nothing,EvaluationFailure}` — failure detail (`nothing`
    unless status is non-success)
  - `wall_time_s::Float64`       — elapsed time in seconds (monotonic clock)

Cross-field invariants (enforced by the constructor):
  - `:success`   → `summary !== nothing`, `failure === nothing`,
    `wall_time_s` finite and ≥ 0, `summary` contains no nonfinite values
  - non-success  → `summary === nothing`, `failure !== nothing`,
    `wall_time_s` finite and ≥ 0

Summaries are not restricted to `NamedTuple`. Scalar, vector, tuple,
`NamedTuple`, and custom immutable types are supported. Recursive finiteness
validation applies to `Real`, arrays, tuples, and named tuples; opaque custom
objects are not automatically validated.
"""
struct EvaluationResult{S,P}
    context::EvaluationContext
    status::Symbol
    summary::Union{Nothing,S}
    provenance::P
    failure::Union{Nothing,EvaluationFailure}
    wall_time_s::Float64

    function EvaluationResult{S,P}(
        context::EvaluationContext,
        status::Symbol,
        summary::Union{Nothing,S},
        provenance::P,
        failure::Union{Nothing,EvaluationFailure},
        wall_time_s::Float64,
    ) where {S,P}
        status in EVALUATION_STATUSES || throw(ArgumentError(
            "EvaluationResult status must be one of $(EVALUATION_STATUSES), " *
            "got $(repr(status))"))
        isfinite(wall_time_s) && wall_time_s >= 0.0 || throw(ArgumentError(
            "EvaluationResult wall_time_s must be finite and ≥ 0, " *
            "got $(wall_time_s)"))

        if status === EVALUATION_STATUS_SUCCESS
            summary === nothing && throw(ArgumentError(
                "successful EvaluationResult must have a non-nothing summary"))
            failure !== nothing && throw(ArgumentError(
                "successful EvaluationResult must have failure === nothing"))
            _is_finite_summary(summary) || throw(ArgumentError(
                "successful EvaluationResult summary contains nonfinite values"))
        else
            summary === nothing || throw(ArgumentError(
                "non-success EvaluationResult must have summary === nothing"))
            failure === nothing && throw(ArgumentError(
                "non-success EvaluationResult must have a non-nothing failure"))
        end

        return new(context, status, summary, provenance, failure, wall_time_s)
    end
end

"""
    EvaluationResult(
        context::EvaluationContext,
        status::Symbol,
        summary,
        provenance;
        failure::Union{Nothing,EvaluationFailure}=nothing,
        wall_time_s::Float64=0.0,
    ) -> EvaluationResult

Convenience constructor that infers `S` and `P` from the arguments and
validates all cross-field invariants.
"""
function EvaluationResult(
    context::EvaluationContext,
    status::Symbol,
    summary,
    provenance;
    failure::Union{Nothing,EvaluationFailure}=nothing,
    wall_time_s::Float64=0.0,
)
    return EvaluationResult{typeof(summary),typeof(provenance)}(
        context, status, summary, provenance, failure, wall_time_s,
    )
end

Base.:(==)(a::EvaluationResult, b::EvaluationResult) =
    (a.context == b.context) &&
    (a.status == b.status) &&
    isequal(a.summary, b.summary) &&
    isequal(a.provenance, b.provenance) &&
    (a.failure == b.failure) &&
    (a.wall_time_s == b.wall_time_s)

function Base.show(io::IO, r::EvaluationResult)
    return print(io,
        "EvaluationResult($(r.context), status=$(r.status), " *
        "summary=$(r.summary !== nothing), wall_time_s=$(r.wall_time_s))")
end

# ============================================================================
# Recursive numerical-finiteness validation
# ============================================================================

"""
    _is_finite_summary(x) -> Bool

Recursive finiteness validation for evaluation summaries.

Returns `true` only if `x` and every nested numerical leaf is finite.

Handled types:
  - `Real`            → `isfinite(x)`
  - `AbstractArray`   → all elements finite (recursively)
  - `Tuple`           → all elements finite (recursively)
  - `NamedTuple`      → all values finite (recursively)
  - `Missing`         → `false` (missing is not a finite numerical value)
  - `nothing`         → `true` (absence of a value is not nonfinite)
  - everything else   → `true` (opaque custom objects are not validated)

Do not claim automatic validation of arbitrary opaque custom objects; callers
that wrap such objects are responsible for their own validity checks.
"""
function _is_finite_summary(x::Real)::Bool
    return isfinite(x)
end

function _is_finite_summary(x::AbstractArray)::Bool
    return all(_is_finite_summary, x)
end

function _is_finite_summary(x::Tuple)::Bool
    return all(_is_finite_summary, x)
end

function _is_finite_summary(x::NamedTuple)::Bool
    return all(_is_finite_summary, values(x))
end

_is_finite_summary(::Missing)::Bool = false
_is_finite_summary(::Nothing)::Bool = true
_is_finite_summary(::Any)::Bool = true
