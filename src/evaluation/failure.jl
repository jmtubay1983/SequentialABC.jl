
# ============================================================================
# Failure types
# ============================================================================

"""
    EVALUATION_FAILURE_CATEGORIES

Permitted [`EvaluationFailure`](@ref) categories.
"""
const EVALUATION_FAILURE_CATEGORIES = (
    :simulator_error,
    :nonfinite_summary,
    :timeout,
    :user_defined,
)

"""
    EvaluationFailure

Serializable description of a software failure during simulator evaluation.

Fields:
  - `category::Symbol`         — one of `:simulator_error`,
    `:nonfinite_summary`, `:timeout`, or `:user_defined`
  - `exception_type::String`   — type name of the originating exception
  - `message::String`          — exception message
  - `stacktrace::Union{Nothing,String}` — optional textual stack trace

No arbitrary `Exception` objects are persisted; only their type name,
message, and an optional textual stack trace. This keeps the type serializable
and free of runtime object references. Constructors reject unknown categories
with an `ArgumentError`.
"""
struct EvaluationFailure
    category::Symbol
    exception_type::String
    message::String
    stacktrace::Union{Nothing,String}

    function EvaluationFailure(
        category::Symbol,
        exception_type::AbstractString,
        message::AbstractString,
        stacktrace::Union{Nothing,AbstractString},
    )
        category in EVALUATION_FAILURE_CATEGORIES || throw(ArgumentError(
            "EvaluationFailure category must be one of " *
            "$(EVALUATION_FAILURE_CATEGORIES), got $(repr(category))"))
        return new(
            category, String(exception_type), String(message),
            stacktrace === nothing ? nothing : String(stacktrace),
        )
    end
end

function EvaluationFailure(
    category::Symbol,
    exception_type::AbstractString,
    message::AbstractString;
    stacktrace::Union{Nothing,AbstractString}=nothing,
)
    return EvaluationFailure(category, exception_type, message, stacktrace)
end

function EvaluationFailure(
    category::Symbol,
    exception_type::Symbol,
    message::AbstractString;
    stacktrace::Union{Nothing,AbstractString}=nothing,
)
    return EvaluationFailure(
        category, string(exception_type), message, stacktrace)
end

Base.show(io::IO, r::EvaluationFailure) =
    print(io, "EvaluationFailure($(r.category), $(r.exception_type): $(r.message))")

Base.:(==)(a::EvaluationFailure, b::EvaluationFailure) =
    (a.category == b.category) &&
    (a.exception_type == b.exception_type) &&
    (a.message == b.message) &&
    (a.stacktrace == b.stacktrace)
