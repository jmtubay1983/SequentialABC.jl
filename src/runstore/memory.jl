# ============================================================================
# In-memory reference run store
# ============================================================================
#
# `MemoryRunStore` is the normative reference backend for run-history
# semantics. It defines what a run history *means* — what a generation is,
# when it is complete, which records may coexist, and how queries are
# ordered — without committing to a storage technology.
#
# Deliberately out of scope in this PR: files, directories, serialization,
# checksums, locks, atomic publication, and portable export. Those belong to
# a later backend that implements the same semantics.
#
# The store records history. It never resumes inference: there is no API that
# reconstructs sampler state from a store, and none is planned.

"""
    AbstractRunStore

Abstract supertype of run-history stores.

A run store holds the queryable scientific and operational history of a run:
populations, evaluations, statuses, failures, timings, and provenance. A
store is **not** a checkpoint: it does not hold executable sampler state and
it cannot resume inference.

The concrete backend in this build is [`MemoryRunStore`](@ref), the normative
reference implementation of the store semantics. Backends implement the same
function names:

  * [`initialize_run!`](@ref), [`open_run`](@ref)
  * [`write_population!`](@ref), [`write_evaluation!`](@ref),
    [`finalize_generation!`](@ref)
  * [`read_population`](@ref), [`read_evaluations`](@ref)
  * [`latest_generation`](@ref), [`generations`](@ref)
  * [`run_metadata`](@ref), [`schema_version`](@ref)
  * [`validate_store`](@ref)
"""
abstract type AbstractRunStore end

"""
    MemoryRunStore <: AbstractRunStore

In-memory reference run store.

The store holds:

  * one immutable [`RunMetadata`](@ref);
  * pending generations (a population and/or evaluations that are not
    finalized yet);
  * complete generations, whose populations and evaluations are immutable.

Completed generations are immutable: writing to a complete generation is an
error, and no record is ever silently replaced. Generations are contiguous
and start at 1.

This backend performs no I/O. It exists to fix the semantics that a
persistent backend must reproduce.
"""
mutable struct MemoryRunStore <: AbstractRunStore
    metadata::RunMetadata
    populations::Dict{Int64,PopulationRecord}
    evaluations::Dict{Int64,Vector{EvaluationRecord}}
    complete::Set{Int64}
    pending_populations::Dict{Int64,PopulationRecord}
    pending_evaluations::Dict{Int64,Vector{EvaluationRecord}}
end

function MemoryRunStore(metadata::RunMetadata)
    return MemoryRunStore(
        metadata,
        Dict{Int64,PopulationRecord}(),
        Dict{Int64,Vector{EvaluationRecord}}(),
        Set{Int64}(),
        Dict{Int64,PopulationRecord}(),
        Dict{Int64,Vector{EvaluationRecord}}(),
    )
end

Base.show(io::IO, store::MemoryRunStore) = print(
    io,
    "MemoryRunStore(run_id=$(repr(store.metadata.run_id)), " *
    "method=$(repr(store.metadata.method)), " *
    "generations=$(length(store.complete)))",
)

# Backends expose their metadata through a `metadata` field; this accessor
# keeps the generic accessors backend-agnostic.
_runstore_metadata(store::AbstractRunStore) = getfield(store, :metadata)

_metadata_summary(m::RunMetadata) = string(
    "run_id=", repr(m.run_id),
    ", method=", repr(m.method),
    ", base_seed=", m.base_seed,
    ", seed_schema=", m.seed_schema_name, "/", m.seed_schema_version,
)

# --- initialization ---------------------------------------------------------

"""
    initialize_run!(::Type{MemoryRunStore}; metadata::RunMetadata) -> MemoryRunStore

Create a new in-memory run store with the given [`RunMetadata`](@ref).

The metadata becomes the store's identity and is never changed afterwards.
"""
function initialize_run!(::Type{MemoryRunStore}; metadata::RunMetadata)
    return MemoryRunStore(metadata)
end

"""
    initialize_run!(store::MemoryRunStore; metadata::RunMetadata) -> MemoryRunStore

Reuse an already initialized in-memory store.

When `metadata` is identical to the store's metadata, the store is returned
unchanged. Conflicting metadata is an error
([`ConflictingRunMetadata`](@ref)): a run has exactly one identity.
"""
function initialize_run!(store::MemoryRunStore; metadata::RunMetadata)
    metadata == store.metadata || throw(ConflictingRunMetadata(
        store.metadata.run_id,
        _metadata_summary(store.metadata),
        _metadata_summary(metadata),
    ))
    return store
end

"""
    open_run(store::MemoryRunStore; metadata::RunMetadata) -> MemoryRunStore

Open an already initialized in-memory store under identical metadata.

An in-memory store is its own handle, so opening it means reusing it. The
metadata must match exactly; otherwise
[`ConflictingRunMetadata`](@ref) is thrown.
"""
function open_run(store::MemoryRunStore; metadata::RunMetadata)
    metadata == store.metadata || throw(ConflictingRunMetadata(
        store.metadata.run_id,
        _metadata_summary(store.metadata),
        _metadata_summary(metadata),
    ))
    return store
end

"""
    close_run!(store::MemoryRunStore) -> Nothing

Release an in-memory run store.

The in-memory backend holds no external resource: there is no writer lease
and no pending state that outlives the process, so this is a no-op that
exists so that both backends share one lifecycle.
"""
close_run!(store::MemoryRunStore) = nothing

# --- writing ----------------------------------------------------------------

function _next_writable_generation(store::MemoryRunStore)
    return _next_writable_generation(store.complete)
end

"""
    write_population!(store::MemoryRunStore, record::PopulationRecord) -> PopulationRecord

Record the population of a generation.

The generation must be the next writable generation: generations are
contiguous and start at 1. A generation that is already complete is never
modified ([`GenerationAlreadyComplete`](@ref)), and a generation that already
has a population is never given a second one
([`DuplicatePopulation`](@ref)).
"""
function write_population!(store::MemoryRunStore, record::PopulationRecord)
    gen = record.generation
    gen in store.complete && throw(GenerationAlreadyComplete(gen))
    expected = _next_writable_generation(store)
    gen == expected || throw(GenerationGap(gen, expected))
    haskey(store.pending_populations, gen) && throw(DuplicatePopulation(gen))
    store.pending_populations[gen] = copy(record)
    return record
end

"""
    write_evaluation!(store::MemoryRunStore, record::EvaluationRecord) -> EvaluationRecord

Record one evaluation of a generation.

The generation is the one named by the record's
[`EvaluationContext`](@ref). The evaluation's method must match the run's
method ([`MethodMismatch`](@ref)), and its identity must be unique inside the
generation ([`DuplicateEvaluation`](@ref)).
"""
function write_evaluation!(store::MemoryRunStore, record::EvaluationRecord)
    gen = record.context.generation
    gen in store.complete && throw(GenerationAlreadyComplete(gen))
    expected = _next_writable_generation(store)
    gen == expected || throw(GenerationGap(gen, expected))
    record.context.method == store.metadata.method || throw(MethodMismatch(
        record.context.method, store.metadata.method))
    key = EvaluationKey(record)
    records = get!(store.pending_evaluations, gen, EvaluationRecord[])
    for existing in records
        EvaluationKey(existing) == key && throw(DuplicateEvaluation(key))
    end
    push!(records, copy(record))
    return record
end

"""
    finalize_generation!(store::MemoryRunStore, generation::Integer;
                         expected_evaluations=nothing) -> GenerationCompletion

Complete a generation.

The generation must have a population, must be the next writable generation,
and must not be complete already. When `expected_evaluations` is supplied, it
is a collection of [`EvaluationKey`](@ref) values (or of
[`EvaluationRecord`](@ref)s) describing the evaluations the generation must
contain: missing identities raise [`MissingExpectedEvaluation`](@ref) and
extra identities raise [`UnexpectedEvaluation`](@ref).

On success the population and the evaluations of the generation become
immutable, and the evaluations are stored in the canonical
[`EvaluationKey`](@ref) order.
"""
function finalize_generation!(store::MemoryRunStore, generation::Integer;
                              expected_evaluations=nothing)
    gen = _checked_int64(generation, "generation")
    gen in store.complete && throw(GenerationAlreadyComplete(gen))
    expected = _next_writable_generation(store)
    gen == expected || throw(GenerationGap(gen, expected))
    haskey(store.pending_populations, gen) || throw(MissingPopulation(gen))
    records = get(store.pending_evaluations, gen, EvaluationRecord[])

    if expected_evaluations !== nothing
        planned = Set{EvaluationKey}()
        for item in expected_evaluations
            push!(planned, item isa EvaluationRecord ? EvaluationKey(item) : item)
        end
        actual = Set{EvaluationKey}(EvaluationKey(r) for r in records)
        missing_keys = sort!(collect(setdiff(planned, actual)))
        isempty(missing_keys) || throw(MissingExpectedEvaluation(missing_keys))
        extra_keys = sort!(collect(setdiff(actual, planned)))
        isempty(extra_keys) || throw(UnexpectedEvaluation(extra_keys))
    end

    sort!(records; by = r -> EvaluationKey(r))
    population = pop!(store.pending_populations, gen)
    store.populations[gen] = population
    store.evaluations[gen] = records
    delete!(store.pending_evaluations, gen)
    push!(store.complete, gen)
    return GenerationCompletion(
        gen, length(population.particles), length(records), now(UTC))
end

# --- reading ----------------------------------------------------------------

"""
    read_population(store::MemoryRunStore, generation::Integer) -> PopulationRecord

Read the population of a complete generation.

Only complete generations are visible. A generation that is missing or not
complete raises [`MissingGeneration`](@ref). The returned record is a copy:
mutating it cannot change the store.
"""
function read_population(store::MemoryRunStore, generation::Integer)
    gen = _checked_int64(generation, "generation")
    haskey(store.populations, gen) || throw(MissingGeneration(gen))
    return copy(store.populations[gen])
end

"""
    read_evaluations(store::MemoryRunStore; generation=nothing, particle=nothing,
                     proposal=nothing, attempt=nothing, replicate=nothing,
                     status=nothing, failure_category=nothing,
                     min_wall_time_s=nothing, max_wall_time_s=nothing)
        -> Vector{EvaluationRecord}

Read recorded evaluations in a deterministic order.

Filters:

  * `generation` — `nothing`, one generation, or a collection of generations.
    Every requested generation must be complete, otherwise
    [`MissingGeneration`](@ref) is thrown.
  * `particle`, `proposal`, `attempt`, `replicate` — `nothing`, one index, or
    a collection of indices.
  * `status` — `nothing`, one status symbol, or a collection of them.
  * `failure_category` — `nothing`, one failure category, or a collection of
    them. Successful evaluations have no failure category and are excluded
    when this filter is used.
  * `min_wall_time_s`, `max_wall_time_s` — inclusive wall-time bounds.

The result is ordered by generation and then by
[`EvaluationKey`](@ref), which is a deterministic total order. The returned
records are copies.
"""
function read_evaluations(
    store::MemoryRunStore;
    generation=nothing,
    particle=nothing,
    proposal=nothing,
    attempt=nothing,
    replicate=nothing,
    status=nothing,
    failure_category=nothing,
    min_wall_time_s=nothing,
    max_wall_time_s=nothing,
)
    generations_to_read = _selected_generations(store.complete, generation)
    min_t = _checked_bound(min_wall_time_s, "min_wall_time_s")
    max_t = _checked_bound(max_wall_time_s, "max_wall_time_s")
    min_t === nothing || max_t === nothing || min_t <= max_t || throw(ArgumentError(
        "min_wall_time_s ($(min_t)) must not exceed max_wall_time_s ($(max_t))"))

    out = EvaluationRecord[]
    for gen in generations_to_read
        for record in get(store.evaluations, gen, EvaluationRecord[])
            _matches_index(record.context.particle, particle) || continue
            _matches_index(record.context.proposal, proposal) || continue
            _matches_index(record.context.attempt, attempt) || continue
            _matches_index(record.context.replicate, replicate) || continue
            _matches_symbol(record.status, status) || continue
            _matches_failure_category(record, failure_category) || continue
            if min_t !== nothing || max_t !== nothing
                wt = record.wall_time_s
                min_t !== nothing && wt < min_t && continue
                max_t !== nothing && wt > max_t && continue
            end
            push!(out, copy(record))
        end
    end
    sort!(out; by = r -> (r.context.generation, EvaluationKey(r)))
    return out
end

# --- store-level accessors --------------------------------------------------

"""
    latest_generation(store::MemoryRunStore) -> Union{Nothing,Int}

The latest complete generation, or `nothing` when the store is empty.
"""
function latest_generation(store::MemoryRunStore)
    isempty(store.complete) && return nothing
    return maximum(store.complete)
end

"""
    generations(store::MemoryRunStore) -> Vector{Int}

The complete generations of the store, in ascending order.
"""
function generations(store::MemoryRunStore)
    return sort!(collect(store.complete))
end

"""
    run_metadata(store::AbstractRunStore) -> RunMetadata

The immutable identity of the store.
"""
function run_metadata(store::AbstractRunStore)
    return copy(_runstore_metadata(store))
end

"""
    schema_version(store::AbstractRunStore) -> UInt64

The storage schema version of the store.

The storage schema is independent of the seed schema, the checkpoint schema,
and any portable-export schema.
"""
function schema_version(store::AbstractRunStore)
    return _runstore_metadata(store).storage_schema_version
end

"""
    schema_version(::Type{MemoryRunStore}) -> UInt64

The storage schema version written by this build.
"""
schema_version(::Type{MemoryRunStore}) = RUNSTORE_SCHEMA_VERSION

# --- validation -------------------------------------------------------------

"""
    validate_store(store::MemoryRunStore) -> StoreValidationReport

Validate every invariant of the store.

The report lists errors (which make the store invalid), warnings (such as
generations that were started but never completed), the complete and pending
generations, and every visible omission, so that a value that was
deliberately not recorded is never invisible.

Validation never mutates the store.
"""
function validate_store(store::MemoryRunStore)
    errors = String[]
    warnings = String[]
    omissions = String[]

    metadata = store.metadata
    if metadata.storage_schema_name != RUNSTORE_SCHEMA_NAME ||
            metadata.storage_schema_version != RUNSTORE_SCHEMA_VERSION
        push!(errors, string(
            "unsupported storage schema ", metadata.storage_schema_name, "/",
            metadata.storage_schema_version, "; this build reads ",
            RUNSTORE_SCHEMA_NAME, "/", RUNSTORE_SCHEMA_VERSION))
    end

    complete = sort!(collect(store.complete))
    expected = Int64(1)
    for gen in complete
        gen == expected || push!(errors, string(
            "generations must be contiguous and start at 1; expected generation ",
            expected, " but found ", gen))
        expected = gen + 1
    end

    for (gen, population) in sort!(collect(store.pending_populations); by = first)
        push!(warnings, string(
            "generation ", gen, " has a population that was never finalized"))
    end
    for gen in sort!(collect(keys(store.pending_evaluations)))
        push!(warnings, string(
            "generation ", gen, " has ", length(store.pending_evaluations[gen]),
            " evaluation(s) that were never finalized"))
    end

    seen = Set{EvaluationKey}()
    for gen in complete
        haskey(store.populations, gen) || push!(errors, string(
            "generation ", gen, " is complete but has no population"))
        haskey(store.evaluations, gen) || push!(errors, string(
            "generation ", gen, " is complete but has no evaluation record"))
        population = get(store.populations, gen, nothing)
        if population !== nothing && population.generation != gen
            push!(errors, string(
                "generation ", gen, " stores a population of generation ",
                population.generation))
        end
        if population !== nothing
            for (k, v) in population.provenance.entries
                if v isa RecordedOmitted
                    push!(omissions, string(
                        "generation ", gen, " population: provenance entry ",
                        repr(k), " of type ", v.type_name, " was omitted (",
                        v.reason, ")"))
                end
            end
        end
        for record in get(store.evaluations, gen, EvaluationRecord[])
            key = EvaluationKey(record)
            key in seen && push!(errors, string(
                "duplicate evaluation identity ", key, " in generation ", gen))
            push!(seen, key)
            record.context.generation == gen || push!(errors, string(
                "evaluation ", key, " is stored in generation ", gen,
                " but its context names generation ", record.context.generation))
            record.context.method == metadata.method || push!(errors, string(
                "evaluation ", key, " has method ", record.context.method,
                " but the run method is ", metadata.method))
            _collect_omissions!(omissions, record, gen)
        end
    end

    return StoreValidationReport(
        isempty(errors), errors, warnings, complete,
        sort!(collect(keys(store.pending_evaluations))), omissions,
    )
end

function _collect_omissions!(omissions::Vector{String},
                             record::EvaluationRecord, gen::Int64)
    key = EvaluationKey(record)
    if record.summary.state === :omitted
        push!(omissions, string(
            "generation ", gen, " evaluation ", key, ": summary of type ",
            record.summary.type_name, " was omitted (", record.summary.reason, ")"))
    end
    for (k, v) in record.provenance.entries
        if v isa RecordedOmitted
            push!(omissions, string(
                "generation ", gen, " evaluation ", key, ": provenance entry ",
                repr(k), " of type ", v.type_name, " was omitted (", v.reason, ")"))
        end
    end
    return omissions
end
