# ============================================================================
# Run-store error types
# ============================================================================
#
# Every error raised by the run-history store is a `RunStoreError`. The store
# fails closed: an invalid record, a duplicate identity, a conflicting
# metadata object, or an unsupported schema never yields a partially written
# or silently altered result.
#
# These errors are model- and domain-agnostic. They never carry domain
# terminology, and they never turn a software failure into a numeric value.

"""
    RunStoreError

Abstract supertype of every error raised by the run-history store.

A `RunStoreError` signals that a run-history operation could not be completed
while preserving the store's invariants. The store is never left in a state
where a failed operation is visible as a successful one.
"""
abstract type RunStoreError <: Exception end

"""
    UnsupportedRecordedValue(type_name, reason)

A value could not be represented in the closed [`RecordedValue`](@ref) tree.

The offending value was **not** stored. Use an explicit omission
(`RecordedOmitted`) when the absence of a value must itself be recorded.
"""
struct UnsupportedRecordedValue <: RunStoreError
    type_name::String
    reason::String
end

Base.showerror(io::IO, e::UnsupportedRecordedValue) = print(
    io,
    "UnsupportedRecordedValue: cannot record a value of type $(e.type_name) " *
    "in the canonical run-history value tree ($(e.reason))",
)

"""
    NonfiniteRecordedValue(type_name, value)

A nonfinite numeric value was encountered while recording a value.

Nonfinite numbers are rejected by default. They may be represented
explicitly with the tagged scientific sentinel
[`RecordedSentinel`](@ref) (`:nan`, `:positive_infinity`,
`:negative_infinity`), which is the only way a nonfinite number enters a
recorded value tree.
"""
struct NonfiniteRecordedValue <: RunStoreError
    type_name::String
    value::String
end

Base.showerror(io::IO, e::NonfiniteRecordedValue) = print(
    io,
    "NonfiniteRecordedValue: $(e.type_name) value $(e.value) is nonfinite; " *
    "record a tagged RecordedSentinel explicitly if this is intended",
)

"""
    UnsupportedSchemaVersion(found_name, found_version, supported_name, supported_versions)

The storage schema of a run-history store is not supported by this build.

Storage schema, seed schema, checkpoint schema, and portable-export schema
are independent concepts; this error refers to the storage schema only.
"""
struct UnsupportedSchemaVersion <: RunStoreError
    found_name::String
    found_version::UInt64
    supported_name::String
    supported_versions::Vector{UInt64}
end

Base.showerror(io::IO, e::UnsupportedSchemaVersion) = print(
    io,
    "UnsupportedSchemaVersion: storage schema $(e.found_name) version " *
    "$(e.found_version) is not supported; this build reads " *
    "$(e.supported_name) version(s) $(join(e.supported_versions, ", ")) " *
    "and performs no automatic migration",
)

"""
    ConflictingRunMetadata(run_id, expected_hash_free_summary, found_summary)

A run-history store was opened or reused with metadata that differs from the
metadata it was initialized with.

Metadata conflicts are always errors: a store has exactly one identity.
"""
struct ConflictingRunMetadata <: RunStoreError
    run_id::String
    existing::String
    requested::String
end

Base.showerror(io::IO, e::ConflictingRunMetadata) = print(
    io,
    "ConflictingRunMetadata: run $(repr(e.run_id)) was initialized with " *
    "metadata ($(e.existing)) but was opened with different metadata " *
    "($(e.requested))",
)

"""
    InvalidRunStoreRecord(kind, reason)

A run-history record violates a structural invariant.

The record is rejected at construction time whenever the violation is
detectable there, and at validation time for cross-record invariants.
"""
struct InvalidRunStoreRecord <: RunStoreError
    kind::String
    reason::String
end

Base.showerror(io::IO, e::InvalidRunStoreRecord) = print(
    io,
    "InvalidRunStoreRecord: invalid $(e.kind): $(e.reason)",
)

"""
    DuplicatePopulation(generation)

A population was written for a generation that already has one.

Populations are append-only: a generation holds exactly one population.
"""
struct DuplicatePopulation <: RunStoreError
    generation::Int64
end

Base.showerror(io::IO, e::DuplicatePopulation) = print(
    io,
    "DuplicatePopulation: generation $(e.generation) already has a population; " *
    "populations are append-only and are never replaced",
)

"""
    DuplicateEvaluation(key)

An evaluation with an identity that already exists in the same generation was
written.

The identity of an evaluation is `(method, generation, particle, proposal,
attempt, replicate)`.
"""
struct DuplicateEvaluation <: RunStoreError
    key::Any
end

Base.showerror(io::IO, e::DuplicateEvaluation) = print(
    io,
    "DuplicateEvaluation: an evaluation with identity $(e.key) already exists " *
    "in this generation",
)

"""
    GenerationAlreadyComplete(generation)

An operation targeted a generation that is already complete.

Completed generations are immutable.
"""
struct GenerationAlreadyComplete <: RunStoreError
    generation::Int64
end

Base.showerror(io::IO, e::GenerationAlreadyComplete) = print(
    io,
    "GenerationAlreadyComplete: generation $(e.generation) is already complete " *
    "and is immutable",
)

"""
    MissingGeneration(generation)

A generation was requested that the store does not contain.

Only complete generations are visible through the read API.
"""
struct MissingGeneration <: RunStoreError
    generation::Int64
end

Base.showerror(io::IO, e::MissingGeneration) = print(
    io,
    "MissingGeneration: generation $(e.generation) is not present in this store",
)

"""
    GenerationGap(generation, expected)

A generation was written out of order.

Generations are contiguous and start at 1; a generation may only be opened
when it directly follows the latest complete generation.
"""
struct GenerationGap <: RunStoreError
    generation::Int64
    expected::Int64
end

Base.showerror(io::IO, e::GenerationGap) = print(
    io,
    "GenerationGap: generation $(e.generation) cannot be written; the next " *
    "writable generation is $(e.expected)",
)

"""
    MissingPopulation(generation)

A generation was finalized without a population.

Every complete generation holds exactly one population.
"""
struct MissingPopulation <: RunStoreError
    generation::Int64
end

Base.showerror(io::IO, e::MissingPopulation) = print(
    io,
    "MissingPopulation: generation $(e.generation) has no population and " *
    "cannot be finalized",
)

"""
    MissingExpectedEvaluation(keys)

A finalization expectation listed evaluations that the store does not hold.

Missing evaluations are detected when an expected plan is supplied.
"""
struct MissingExpectedEvaluation <: RunStoreError
    keys::Vector{Any}
end

Base.showerror(io::IO, e::MissingExpectedEvaluation) = print(
    io,
    "MissingExpectedEvaluation: $(length(e.keys)) expected evaluation(s) are " *
    "missing: $(join(map(string, e.keys), ", "))",
)

"""
    UnexpectedEvaluation(keys)

A finalization expectation did not list evaluations that the store holds.

Extra evaluations are detected when an expected plan is supplied.
"""
struct UnexpectedEvaluation <: RunStoreError
    keys::Vector{Any}
end

Base.showerror(io::IO, e::UnexpectedEvaluation) = print(
    io,
    "UnexpectedEvaluation: $(length(e.keys)) evaluation(s) are not part of the " *
    "expected plan: $(join(map(string, e.keys), ", "))",
)

"""
    MethodMismatch(recorded, expected)

An evaluation was written whose method differs from the run's method.

One run-history store represents one method.
"""
struct MethodMismatch <: RunStoreError
    recorded::Symbol
    expected::Symbol
end

Base.showerror(io::IO, e::MethodMismatch) = print(
    io,
    "MethodMismatch: evaluation method $(repr(e.recorded)) does not match the " *
    "run method $(repr(e.expected))",
)
