# ============================================================================
# Run-history records
# ============================================================================
#
# Three immutable record types describe a run history:
#
#   * `PopulationRecord` — one accepted population of a generation
#   * `EvaluationRecord` — one simulator evaluation
#   * `GenerationCompletion` — the in-memory completion record of a generation
#
# All validation is model- and domain-agnostic. The production
# `EvaluationContext` and `EvaluationResult` types are the authoritative
# source for evaluation identity and outcome.

# --- evaluation identity ----------------------------------------------------

"""
    EvaluationKey

Identity of one evaluation inside a run.

Fields:
  - `method::Symbol`
  - `generation::Int64`
  - `particle::Int64`
  - `proposal::Int64`
  - `attempt::Int64`
  - `replicate::Int64`

The identity is derived from the production
[`EvaluationContext`](@ref), which is the authoritative source. Two
evaluations of the same run are the same evaluation exactly when their keys
are equal; the seed is deliberately not part of the key, so a repeated
identity is a duplicate even if the seed differs.

Keys define a deterministic total order (method name first, then the
identity indices), which is the order used by every query.
"""
struct EvaluationKey
    method::Symbol
    generation::Int64
    particle::Int64
    proposal::Int64
    attempt::Int64
    replicate::Int64

    function EvaluationKey(
        method::Symbol,
        generation::Integer,
        particle::Integer,
        proposal::Integer,
        attempt::Integer,
        replicate::Integer,
    )
        isempty(string(method)) && throw(ArgumentError(
            "EvaluationKey method must be a nonempty Symbol, got $(repr(method))"))
        return new(
            method,
            _positive_index(generation, "EvaluationKey generation"),
            _positive_index(particle, "EvaluationKey particle"),
            _positive_index(proposal, "EvaluationKey proposal"),
            _positive_index(attempt, "EvaluationKey attempt"),
            _positive_index(replicate, "EvaluationKey replicate"),
        )
    end
end

function _positive_index(x::Integer, name::AbstractString)
    v = _checked_int64(x, name)
    v >= 1 || throw(ArgumentError(
        "$(name) must be >= 1, got $(v)"))
    return v
end

EvaluationKey(context::EvaluationContext) = EvaluationKey(
    context.method, context.generation, context.particle,
    context.proposal, context.attempt, context.replicate,
)

Base.:(==)(a::EvaluationKey, b::EvaluationKey) =
    a.method == b.method &&
    a.generation == b.generation &&
    a.particle == b.particle &&
    a.proposal == b.proposal &&
    a.attempt == b.attempt &&
    a.replicate == b.replicate

function Base.isless(a::EvaluationKey, b::EvaluationKey)
    am, bm = string(a.method), string(b.method)
    am != bm && return am < bm
    a.generation != b.generation && return a.generation < b.generation
    a.particle != b.particle && return a.particle < b.particle
    a.proposal != b.proposal && return a.proposal < b.proposal
    a.attempt != b.attempt && return a.attempt < b.attempt
    return a.replicate < b.replicate
end

function Base.hash(k::EvaluationKey, h::UInt)
    h = hash(k.method, h)
    h = hash(k.generation, h)
    h = hash(k.particle, h)
    h = hash(k.proposal, h)
    h = hash(k.attempt, h)
    return hash(k.replicate, h)
end

Base.show(io::IO, k::EvaluationKey) = print(
    io,
    "EvaluationKey($(k.method), gen=$(k.generation), particle=$(k.particle), " *
    "proposal=$(k.proposal), attempt=$(k.attempt), replicate=$(k.replicate))",
)

# --- summary capture --------------------------------------------------------

"""
    SummaryCapture

Canonical capture of the summary of one evaluation.

Fields:
  - `state::Symbol` — `:recorded`, `:omitted`, or `:not_applicable`
  - `value::Union{Nothing,RecordedValue}` — the recorded summary
  - `type_name::String` — type name of the original summary
  - `reason::Union{Nothing,String}` — why the value was omitted

`:recorded` means the summary was converted into the canonical value tree.
`:omitted` means the summary existed but could not be represented canonically
and was deliberately not recorded; the omission is visible and carries a
reason. `:not_applicable` means the evaluation did not produce a summary
(failures and other non-success outcomes).

Use [`recorded_summary`](@ref), [`omitted_summary`](@ref), and
[`not_applicable_summary`](@ref) to build a capture.
"""
struct SummaryCapture
    state::Symbol
    value::Union{Nothing,RecordedValue}
    type_name::String
    reason::Union{Nothing,String}

    function SummaryCapture(
        state::Symbol,
        value::Union{Nothing,RecordedValue},
        type_name::AbstractString,
        reason::Union{Nothing,AbstractString},
    )
        st = Symbol(state)
        st in (:recorded, :omitted, :not_applicable) || throw(ArgumentError(
            "SummaryCapture state must be :recorded, :omitted, or " *
            ":not_applicable, got $(repr(st))"))
        tn = String(type_name)
        rs = reason === nothing ? nothing : String(reason)
        if st === :recorded
            value === nothing && throw(ArgumentError(
                "SummaryCapture state :recorded requires a value"))
            rs === nothing || throw(ArgumentError(
                "SummaryCapture state :recorded must not carry a reason"))
        elseif st === :omitted
            value === nothing || throw(ArgumentError(
                "SummaryCapture state :omitted must not carry a value"))
            rs === nothing && throw(ArgumentError(
                "SummaryCapture state :omitted requires a reason"))
        else
            value === nothing || throw(ArgumentError(
                "SummaryCapture state :not_applicable must not carry a value"))
            rs === nothing || throw(ArgumentError(
                "SummaryCapture state :not_applicable must not carry a reason"))
        end
        return new(st, value, tn, rs)
    end
end

"""
    recorded_summary(value; omit::Bool=false, reason=nothing) -> SummaryCapture

Capture a summary that was recorded canonically.

With `omit=true`, an unsupported summary becomes a visible
[`RecordedOmitted`](@ref) instead of an error. Use
[`omitted_summary`](@ref) to omit deliberately.
"""
function recorded_summary(value; omit::Bool=false,
                          reason::Union{Nothing,AbstractString}=nothing)
    type_name = string(typeof(value))
    if omit
        rec = recorded_value(value; omit=true, reason=reason)
        if rec isa RecordedOmitted
            return SummaryCapture(:omitted, nothing, type_name, rec.reason)
        end
        return SummaryCapture(:recorded, rec, type_name, nothing)
    end
    rec = recorded_value(value)
    return SummaryCapture(:recorded, rec, type_name, nothing)
end

"""
    omitted_summary(value; reason, type_name=string(typeof(value))) -> SummaryCapture

Capture a summary that existed but was deliberately not recorded.

The omission stays visible: it carries the type name of the original value
and the reason it was not recorded.
"""
function omitted_summary(value; reason::AbstractString,
                         type_name::AbstractString=string(typeof(value)))
    isempty(reason) && throw(ArgumentError(
        "omitted_summary requires a nonempty reason"))
    return SummaryCapture(:omitted, nothing, type_name, String(reason))
end

"""
    not_applicable_summary(type_name::AbstractString="none") -> SummaryCapture

Capture the absence of a summary, as used by failed, nonfinite, and timed-out
evaluations.
"""
function not_applicable_summary(type_name::AbstractString="none")
    return SummaryCapture(:not_applicable, nothing, String(type_name), nothing)
end

Base.:(==)(a::SummaryCapture, b::SummaryCapture) =
    a.state == b.state && a.value == b.value &&
    a.type_name == b.type_name && a.reason == b.reason

Base.copy(s::SummaryCapture) = SummaryCapture(
    s.state, s.value === nothing ? nothing : copy(s.value),
    s.type_name, s.reason,
)

# --- evaluation record ------------------------------------------------------

"""
    EvaluationRecord

Immutable record of one simulator evaluation in a run history.

Fields:
  - `context::EvaluationContext` — the production evaluation identity
  - `status::Symbol` — `:success`, `:failed`, `:nonfinite`, or `:timeout`
  - `summary::SummaryCapture` — canonical summary capture
  - `failure::Union{Nothing,EvaluationFailure}` — structured failure
  - `wall_time_s::Float64` — elapsed wall time in seconds
  - `provenance::RecordedProvenance` — canonical provenance
  - `discrepancy::Union{Nothing,Float64}` — optional scientific discrepancy

The record mirrors the production [`EvaluationResult`](@ref) invariants:
a successful evaluation has no failure, a non-successful evaluation has a
failure and no summary, and the wall time is finite and nonnegative.

A discrepancy is only ever a value that the caller calculated scientifically
from a successful summary. The store never computes a discrepancy, and a
failure never receives one: a software failure is never turned into a number.
"""
struct EvaluationRecord
    context::EvaluationContext
    status::Symbol
    summary::SummaryCapture
    failure::Union{Nothing,EvaluationFailure}
    wall_time_s::Float64
    provenance::RecordedProvenance
    discrepancy::Union{Nothing,Float64}

    function EvaluationRecord(
        context::EvaluationContext,
        status::Symbol,
        summary::SummaryCapture,
        failure::Union{Nothing,EvaluationFailure},
        wall_time_s::Real,
        provenance::RecordedProvenance,
        discrepancy::Union{Nothing,Real},
    )
        st = Symbol(status)
        st in EVALUATION_STATUSES || throw(ArgumentError(
            "EvaluationRecord status must be one of $(EVALUATION_STATUSES), " *
            "got $(repr(st))"))
        wt = Float64(wall_time_s)
        isfinite(wt) && wt >= 0.0 || throw(ArgumentError(
            "EvaluationRecord wall_time_s must be finite and >= 0, got $(wt)"))
        disc = discrepancy === nothing ? nothing : Float64(discrepancy)
        if st === EVALUATION_STATUS_SUCCESS
            failure === nothing || throw(ArgumentError(
                "successful EvaluationRecord must have failure === nothing"))
            summary.state in (:recorded, :omitted) || throw(ArgumentError(
                "successful EvaluationRecord must have a recorded or omitted " *
                "summary, got state $(repr(summary.state))"))
        else
            failure === nothing && throw(ArgumentError(
                "non-success EvaluationRecord must have a non-nothing failure"))
            summary.state === :not_applicable || throw(ArgumentError(
                "non-success EvaluationRecord must have a not-applicable " *
                "summary, got state $(repr(summary.state))"))
            disc === nothing || throw(ArgumentError(
                "a failed evaluation must never carry a discrepancy"))
        end
        if disc !== nothing
            isfinite(disc) && disc >= 0.0 || throw(ArgumentError(
                "EvaluationRecord discrepancy must be finite and >= 0, got $(disc)"))
        end
        return new(context, st, summary, failure, wt, provenance, disc)
    end
end

"""
    EvaluationRecord(result::EvaluationResult; discrepancy=nothing,
                     provenance=nothing, omit_unsupported_summary::Bool=false,
                     omission_reason=nothing)

Build a run-history record from a production [`EvaluationResult`](@ref).

The result's status, summary, failure, and wall time are carried over
unchanged, and the result's provenance is converted to the canonical
[`RecordedProvenance`](@ref) form. A `NamedTuple` provenance becomes one entry
per field; any other provenance object is rejected, because the store never
stores an arbitrary provenance object implicitly.

`discrepancy` is optional and is only valid for a successful evaluation: it
is a value the caller calculated scientifically, and the store never derives
one. With `omit_unsupported_summary=true`, a summary that cannot be converted
canonically becomes a visible omission instead of an error.
"""
function EvaluationRecord(
    result::EvaluationResult;
    discrepancy::Union{Nothing,Real}=nothing,
    provenance::Union{Nothing,RecordedProvenance}=nothing,
    omit_unsupported_summary::Bool=false,
    omission_reason::Union{Nothing,AbstractString}=nothing,
)
    prov = provenance === nothing ?
        recorded_provenance(result.provenance) : provenance
    cap = if result.status === EVALUATION_STATUS_SUCCESS
        try
            recorded_summary(result.summary)
        catch err
            if err isa UnsupportedRecordedValue && omit_unsupported_summary
                omitted_summary(result.summary;
                                reason=omission_reason === nothing ?
                                    "summary could not be recorded canonically" :
                                    String(omission_reason))
            else
                rethrow(err)
            end
        end
    else
        not_applicable_summary()
    end
    return EvaluationRecord(
        result.context, result.status, cap, result.failure,
        result.wall_time_s, prov, discrepancy,
    )
end

Base.:(==)(a::EvaluationRecord, b::EvaluationRecord) =
    a.context == b.context &&
    a.status == b.status &&
    a.summary == b.summary &&
    a.failure == b.failure &&
    a.wall_time_s == b.wall_time_s &&
    a.provenance == b.provenance &&
    a.discrepancy == b.discrepancy

Base.copy(r::EvaluationRecord) = EvaluationRecord(
    r.context, r.status, copy(r.summary), r.failure, r.wall_time_s,
    copy(r.provenance), r.discrepancy,
)

EvaluationKey(record::EvaluationRecord) = EvaluationKey(record.context)

# --- adaptive diagnostics ---------------------------------------------------

"""
    AdaptiveDiagnostics

Optional adaptive-distance diagnostics of one population.

Fields:
  - `key_order::Vector{Symbol}` — summary keys in the adaptive order
  - `weights::Vector{Float64}` — per-key inverse-distance scale weights
  - `gen_applied::Int64` — generation the weights and epsilon were used by
  - `epsilon_selection_distances::Vector{Float64}` — acceptance distances of
    the preceding population, recomputed under `weights`
  - `epsilon::Float64` — tolerance used by generation `gen_applied`

The diagnostics are optional: a run without adaptive distances records
`nothing`.
"""
struct AdaptiveDiagnostics
    key_order::Vector{Symbol}
    weights::Vector{Float64}
    gen_applied::Int64
    epsilon_selection_distances::Vector{Float64}
    epsilon::Float64

    function AdaptiveDiagnostics(
        key_order::AbstractVector{<:Symbol},
        weights::AbstractVector{<:Real},
        gen_applied::Integer,
        epsilon_selection_distances::AbstractVector{<:Real},
        epsilon::Real,
    )
        ko = collect(Symbol, key_order)
        isempty(ko) && throw(ArgumentError(
            "AdaptiveDiagnostics key_order must be nonempty"))
        length(unique(ko)) == length(ko) || throw(ArgumentError(
            "AdaptiveDiagnostics key_order must not repeat a key"))
        w = Float64.(weights)
        isempty(w) && throw(ArgumentError(
            "AdaptiveDiagnostics weights must be nonempty"))
        length(w) == length(ko) || throw(ArgumentError(
            "AdaptiveDiagnostics weights length $(length(w)) does not match " *
            "the key order length $(length(ko))"))
        all(isfinite, w) || throw(ArgumentError(
            "AdaptiveDiagnostics weights must be finite"))
        all(>=(0.0), w) || throw(ArgumentError(
            "AdaptiveDiagnostics weights must be nonnegative"))
        any(>(0.0), w) || throw(ArgumentError(
            "AdaptiveDiagnostics weights must not be all zero"))
        ga = _checked_int64(gen_applied, "AdaptiveDiagnostics gen_applied")
        ga >= 1 || throw(ArgumentError(
            "AdaptiveDiagnostics gen_applied must be >= 1, got $(ga)"))
        esd = Float64.(epsilon_selection_distances)
        all(isfinite, esd) || throw(ArgumentError(
            "AdaptiveDiagnostics epsilon_selection_distances must be finite"))
        all(>=(0.0), esd) || throw(ArgumentError(
            "AdaptiveDiagnostics epsilon_selection_distances must be nonnegative"))
        eps = Float64(epsilon)
        isfinite(eps) && eps >= 0.0 || throw(ArgumentError(
            "AdaptiveDiagnostics epsilon must be finite and >= 0, got $(eps)"))
        return new(ko, w, ga, esd, eps)
    end
end

Base.:(==)(a::AdaptiveDiagnostics, b::AdaptiveDiagnostics) =
    a.key_order == b.key_order &&
    a.weights == b.weights &&
    a.gen_applied == b.gen_applied &&
    a.epsilon_selection_distances == b.epsilon_selection_distances &&
    a.epsilon == b.epsilon

Base.copy(a::AdaptiveDiagnostics) = AdaptiveDiagnostics(
    copy(a.key_order), copy(a.weights), a.gen_applied,
    copy(a.epsilon_selection_distances), a.epsilon,
)

# --- population record ------------------------------------------------------

"""
    RUNSTORE_WEIGHT_SUM_TOLERANCE

Absolute tolerance used when checking that population weights are normalized
(the sum must be 1 within this tolerance).
"""
const RUNSTORE_WEIGHT_SUM_TOLERANCE = 1e-8

"""
    RUNSTORE_ESS_TOLERANCE

Absolute tolerance used when checking that the effective sample size lies in
`[1, N]`.
"""
const RUNSTORE_ESS_TOLERANCE = 1e-6

"""
    PopulationRecord{P}

Immutable record of one accepted population.

Fields:
  - `generation::Int64` — generation index (>= 1)
  - `particles::Vector{P}` — accepted particles
  - `weights::Vector{Float64}` — normalized importance weights
  - `distances::Vector{Float64}` — acceptance distances
  - `epsilon::Float64` — tolerance of the generation
  - `ess::Float64` — effective sample size
  - `attempts::Union{Nothing,Vector{Int64}}` — optional per-particle attempt counts
  - `ancestry::Union{Nothing,Vector{Int64}}` — optional ancestor indices
  - `resampling_count::Int64` — resampling events of this generation
  - `adaptive::Union{Nothing,AdaptiveDiagnostics}` — optional adaptive diagnostics
  - `started_at::DateTime` — generation start timestamp (UTC semantics)
  - `finished_at::DateTime` — generation end timestamp (UTC semantics)
  - `provenance::RecordedProvenance` — canonical provenance

Required invariants:

  * the generation and the population size are positive, and all vectors are
    aligned with the population size;
  * weights are finite, nonnegative, and normalized;
  * distances are finite and nonnegative;
  * `epsilon` is finite and nonnegative, except for the documented
    generation-one positive-infinity sentinel, which represents an
    unconstrained initial tolerance;
  * the effective sample size is finite and within `[1, N]` up to
    [`RUNSTORE_ESS_TOLERANCE`](@ref);
  * optional attempt counts are positive, and optional ancestry indices are
    valid positions in the population;
  * the timestamps are ordered.

The constructor is deliberately method-agnostic: it does not require
`distance <= epsilon`, because acceptance consistency is a property of a
particular inference method. Use
[`check_acceptance_consistency`](@ref) for that check.
"""
struct PopulationRecord{P}
    generation::Int64
    particles::Vector{P}
    weights::Vector{Float64}
    distances::Vector{Float64}
    epsilon::Float64
    ess::Float64
    attempts::Union{Nothing,Vector{Int64}}
    ancestry::Union{Nothing,Vector{Int64}}
    resampling_count::Int64
    adaptive::Union{Nothing,AdaptiveDiagnostics}
    started_at::DateTime
    finished_at::DateTime
    provenance::RecordedProvenance

    function PopulationRecord(
        generation::Integer,
        particles::AbstractVector,
        weights::AbstractVector{<:Real},
        distances::AbstractVector{<:Real},
        epsilon::Real,
        ess::Real,
        attempts::Union{Nothing,AbstractVector{<:Integer}},
        ancestry::Union{Nothing,AbstractVector{<:Integer}},
        resampling_count::Integer,
        adaptive::Union{Nothing,AdaptiveDiagnostics},
        started_at::DateTime,
        finished_at::DateTime,
        provenance::RecordedProvenance,
    )
        gen = _checked_int64(generation, "PopulationRecord generation")
        gen >= 1 || throw(ArgumentError(
            "PopulationRecord generation must be >= 1, got $(gen)"))
        parts = collect(particles)
        n = length(parts)
        n >= 1 || throw(ArgumentError(
            "PopulationRecord population size must be >= 1, got $(n)"))
        w = Float64.(weights)
        length(w) == n || throw(ArgumentError(
            "PopulationRecord weights length $(length(w)) does not match the " *
            "population size $(n)"))
        d = Float64.(distances)
        length(d) == n || throw(ArgumentError(
            "PopulationRecord distances length $(length(d)) does not match the " *
            "population size $(n)"))
        all(isfinite, w) || throw(ArgumentError(
            "PopulationRecord weights must be finite"))
        all(>=(0.0), w) || throw(ArgumentError(
            "PopulationRecord weights must be nonnegative"))
        total = sum(w)
        abs(total - 1.0) <= RUNSTORE_WEIGHT_SUM_TOLERANCE || throw(ArgumentError(
            "PopulationRecord weights must be normalized; the sum is $(total) " *
            "(tolerance $(RUNSTORE_WEIGHT_SUM_TOLERANCE))"))
        all(isfinite, d) || throw(ArgumentError(
            "PopulationRecord distances must be finite"))
        all(>=(0.0), d) || throw(ArgumentError(
            "PopulationRecord distances must be nonnegative"))
        eps = Float64(epsilon)
        if isfinite(eps)
            eps >= 0.0 || throw(ArgumentError(
                "PopulationRecord epsilon must be >= 0, got $(eps)"))
        else
            (eps == Inf && gen == 1) || throw(ArgumentError(
                "PopulationRecord epsilon must be finite, except for the " *
                "documented positive-infinity sentinel in generation 1; got " *
                "$(eps) in generation $(gen)"))
        end
        e = Float64(ess)
        isfinite(e) || throw(ArgumentError(
            "PopulationRecord ess must be finite, got $(e)"))
        (e >= 1.0 - RUNSTORE_ESS_TOLERANCE && e <= n + RUNSTORE_ESS_TOLERANCE) ||
            throw(ArgumentError(
                "PopulationRecord ess must lie in [1, $(n)] up to a tolerance of " *
                "$(RUNSTORE_ESS_TOLERANCE), got $(e)"))
        att = attempts === nothing ? nothing :
            Int64[_checked_int64(a, "PopulationRecord attempts") for a in attempts]
        if att !== nothing
            length(att) == n || throw(ArgumentError(
                "PopulationRecord attempts length must match the population size " *
                "$(n)"))
            all(>=(1), att) || throw(ArgumentError(
                "PopulationRecord attempts must be >= 1"))
        end
        anc = ancestry === nothing ? nothing :
            Int64[_checked_int64(a, "PopulationRecord ancestry") for a in ancestry]
        if anc !== nothing
            length(anc) == n || throw(ArgumentError(
                "PopulationRecord ancestry length must match the population size $(n)"))
            all(i -> 1 <= i <= n, anc) || throw(ArgumentError(
                "PopulationRecord ancestry indices must lie in 1:$(n)"))
        end
        rc = _checked_int64(resampling_count, "PopulationRecord resampling_count")
        rc >= 0 || throw(ArgumentError(
            "PopulationRecord resampling_count must be >= 0, got $(rc)"))
        if adaptive !== nothing
            nsel = length(adaptive.epsilon_selection_distances)
            (nsel == 0 || nsel == n) || throw(ArgumentError(
                "PopulationRecord adaptive epsilon_selection_distances length " *
                "$(nsel) does not match the population size $(n)"))
        end
        started_at <= finished_at || throw(ArgumentError(
            "PopulationRecord started_at must not be after finished_at"))
        return new{eltype(parts)}(
            gen, parts, w, d, eps, e, att, anc, rc, adaptive,
            started_at, finished_at, provenance,
        )
    end
end

"""
    check_acceptance_consistency(record::PopulationRecord; tolerance=0.0) -> Bool

Check that every acceptance distance of `record` is within `epsilon`.

This is a *method-specific* consistency property, not a generic structural
invariant, so it is not enforced by the [`PopulationRecord`](@ref)
constructor. The positive-infinity sentinel of generation 1 trivially satisfies
the check.

Returns `true` when the record is consistent and throws an
[`InvalidRunStoreRecord`](@ref) otherwise.
"""
function check_acceptance_consistency(record::PopulationRecord; tolerance::Real=0.0)
    tol = Float64(tolerance)
    tol >= 0.0 || throw(ArgumentError("tolerance must be >= 0, got $(tol)"))
    if isfinite(record.epsilon)
        for (i, d) in enumerate(record.distances)
            d <= record.epsilon + tol || throw(InvalidRunStoreRecord(
                "population",
                "acceptance distance $(d) of particle $(i) exceeds epsilon " *
                "$(record.epsilon) (tolerance $(tol))"))
        end
    end
    return true
end

Base.:(==)(a::PopulationRecord, b::PopulationRecord) =
    a.generation == b.generation &&
    a.particles == b.particles &&
    a.weights == b.weights &&
    a.distances == b.distances &&
    a.epsilon == b.epsilon &&
    a.ess == b.ess &&
    a.attempts == b.attempts &&
    a.ancestry == b.ancestry &&
    a.resampling_count == b.resampling_count &&
    a.adaptive == b.adaptive &&
    a.started_at == b.started_at &&
    a.finished_at == b.finished_at &&
    a.provenance == b.provenance

Base.copy(p::PopulationRecord) = PopulationRecord(
    p.generation, copy(p.particles), copy(p.weights), copy(p.distances),
    p.epsilon, p.ess,
    p.attempts === nothing ? nothing : copy(p.attempts),
    p.ancestry === nothing ? nothing : copy(p.ancestry),
    p.resampling_count, p.adaptive === nothing ? nothing : copy(p.adaptive),
    p.started_at, p.finished_at, copy(p.provenance),
)

# --- generation completion and validation report ----------------------------

"""
    GenerationCompletion

In-memory completion record of one complete generation.

Fields:
  - `generation::Int64` — generation index
  - `population_size::Int` — number of accepted particles
  - `evaluation_count::Int` — number of recorded evaluations
  - `finalized_at::DateTime` — completion timestamp (UTC semantics)

A complete generation is immutable.
"""
struct GenerationCompletion
    generation::Int64
    population_size::Int
    evaluation_count::Int
    finalized_at::DateTime
end

"""
    StoreValidationReport

Result of [`validate_store`](@ref).

Fields:
  - `valid::Bool` — whether the store satisfies every invariant
  - `errors::Vector{String}` — violations that make the store invalid
  - `warnings::Vector{String}` — non-fatal observations
  - `generations::Vector{Int}` — complete generations
  - `pending_generations::Vector{Int}` — generations that were started but
    never completed
  - `omitted_values::Vector{String}` — visible omissions, so that a value that
    was deliberately not recorded is never invisible
"""
struct StoreValidationReport
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
    generations::Vector{Int}
    pending_generations::Vector{Int}
    omitted_values::Vector{String}
end
