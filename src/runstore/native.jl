# ============================================================================
# Native, file-backed run store
# ============================================================================
#
# `NativeRunStore` stores a run history in a directory tree:
#
#   run_dir/
#     metadata/
#       schema                 # storage schema name and version
#       run.jls                # the run metadata (written once)
#       run.sha256             # SHA-256 of the exact bytes of run.jls
#       writer.lease           # exclusive writer lease (only while held)
#     generations/
#       generation_000001/
#         population.jls       # the accepted population
#         evaluations.jls      # the evaluations, in canonical order
#         manifest             # sizes, digests, counts, and identities
#         COMPLETE             # written last; binds the manifest
#     tmp/
#       generation_000001.<nonce>/   # staging; never visible
#
# A generation becomes visible only when its directory exists and its
# COMPLETE marker, manifest, and partitions verify. The publication of a
# generation is a single directory rename, so an interruption leaves at most
# a staging directory behind: a partial generation is never canonical.
#
# The `.jls` partitions use Julia `Serialization`. They are native to Julia
# and to the types of this package, they are trusted-input only, and they are
# not a portable export format.
#
# Pending records live in the store session and are **not durable** until
# `finalize_generation!` publishes the generation directory.

const NATIVE_STORE_MODES = (:read, :write)

const NATIVE_POPULATION_FILE = "population.jls"
const NATIVE_EVALUATION_FILE = "evaluations.jls"
const NATIVE_MANIFEST_FILE = "manifest"
const NATIVE_COMPLETE_FILE = "COMPLETE"

"""
    NativeRunStore <: AbstractRunStore

File-backed run store.

A native store keeps the committed history of a run in a directory tree and
reconstructs its visible state from the committed files when it is opened.
The store never keeps the authoritative history in process memory: only the
*pending* generation of the current session is held in memory, and it is
discarded when the session ends without publishing it.

Fields:
  - `root::String`             — absolute path of the run directory
  - `metadata::RunMetadata`    — validated identity of the run
  - `mode::Symbol`             — `:read` or `:write`
  - `pending_generation::Union{Nothing,Int64}` — the generation being written
  - `pending_population::Union{Nothing,PopulationRecord}` — its population
  - `pending_evaluations::Vector{EvaluationRecord}` — its evaluations
  - `pending_keys::Set{EvaluationKey}` — their identities
  - `lease_path::Union{Nothing,String}` — the writer lease this store owns
  - `closed::Bool`             — whether the store was closed

Open a store with [`open_run`](@ref) and close it with [`close_run!`](@ref).
A store in write mode owns the writer lease for the whole session and
releases it only when it is closed; a store in read mode never takes the
lease and can read committed generations while another writer is active.

The native format is Julia-native, trusted-input only, and **not** portable.
A run store records history; it never resumes inference.
"""
mutable struct NativeRunStore <: AbstractRunStore
    root::String
    metadata::RunMetadata
    mode::Symbol
    pending_generation::Union{Nothing,Int64}
    pending_population::Union{Nothing,PopulationRecord}
    pending_evaluations::Vector{EvaluationRecord}
    pending_keys::Set{EvaluationKey}
    lease_path::Union{Nothing,String}
    closed::Bool
end

function NativeRunStore(root::AbstractString, metadata::RunMetadata, mode::Symbol)
    return NativeRunStore(
        abspath(root), metadata, mode, nothing, nothing,
        EvaluationRecord[], Set{EvaluationKey}(), nothing, false,
    )
end

Base.show(io::IO, store::NativeRunStore) = print(
    io,
    "NativeRunStore(run_id=$(repr(store.metadata.run_id)), " *
    "mode=$(repr(store.mode)), root=$(repr(store.root)))",
)

# --- paths ------------------------------------------------------------------

_native_metadata_dir(root::AbstractString) = joinpath(root, "metadata")
_native_generations_dir(root::AbstractString) = joinpath(root, "generations")
_native_tmp_dir(root::AbstractString) = joinpath(root, "tmp")
_native_generation_name(generation::Integer) =
    "generation_" * lpad(string(generation), 6, '0')
_native_generation_dir(root::AbstractString, generation::Integer) =
    joinpath(_native_generations_dir(root), _native_generation_name(generation))

# --- lifecycle --------------------------------------------------------------

"""
    initialize_run!(::Type{NativeRunStore}, root::AbstractString;
                    metadata::RunMetadata) -> NativeRunStore

Create a native run store, or open the existing one under identical metadata.

The run directory is created with a `metadata/` directory, a `generations/`
directory, and a `tmp/` staging directory. The storage schema file, the run
metadata, and the metadata digest are written once; an existing store is
never overwritten.

The returned store is in write mode and therefore owns the writer lease for
the session; release it with [`close_run!`](@ref).
"""
function initialize_run!(::Type{NativeRunStore}, root::AbstractString;
                         metadata::RunMetadata)
    rootp = abspath(root)
    metadata_path = joinpath(_native_metadata_dir(rootp), "run.jls")
    if isfile(metadata_path)
        store = open_run(NativeRunStore, rootp; mode = :write)
        store.metadata == metadata || throw(ConflictingRunMetadata(
            store.metadata.run_id,
            _metadata_summary(store.metadata),
            _metadata_summary(metadata),
        ))
        return store
    end

    mkpath(_native_generations_dir(rootp))
    mkpath(_native_tmp_dir(rootp))
    mkpath(_native_metadata_dir(rootp))

    _write_canonical_kv(
        joinpath(_native_metadata_dir(rootp), "schema"),
        Pair{String,String}[
            "schema" => RUNSTORE_SCHEMA_NAME,
            "version" => string(RUNSTORE_SCHEMA_VERSION),
        ],
    )
    open(metadata_path, "w") do io
        serialize(io, metadata)
    end
    open(joinpath(_native_metadata_dir(rootp), "run.sha256"), "w") do io
        print(io, _file_digest(metadata_path), "\n")
    end

    store = NativeRunStore(rootp, metadata, :write)
    _acquire_lease!(store)
    return store
end

"""
    open_run(::Type{NativeRunStore}, root::AbstractString;
             mode::Symbol=:read) -> NativeRunStore

Open an existing native run store.

The storage schema, the run metadata, and the metadata digest are validated
before the store is usable:

  * a missing schema file raises [`MissingRunStoreSchema`](@ref);
  * an unknown schema name or version raises
    [`UnsupportedSchemaVersion`](@ref);
  * a metadata digest mismatch raises [`ChecksumMismatch`](@ref);
  * unreadable metadata raises [`CorruptStore`](@ref).

With `mode = :read` the store takes no writer lease and can read the
committed generations while another writer is active. With `mode = :write`
the store acquires the writer lease and rejects a second writer.

Opening is cheap: the committed generations are discovered from the
generation directories, and the partitions are verified when they are read.
"""
function open_run(::Type{NativeRunStore}, root::AbstractString;
                  mode::Symbol = :read)
    mode in NATIVE_STORE_MODES || throw(ArgumentError(
        "open mode must be :read or :write, got $(repr(mode))"))
    rootp = abspath(root)
    schema_path = joinpath(_native_metadata_dir(rootp), "schema")
    isfile(schema_path) || throw(MissingRunStoreSchema(schema_path))
    entries = try
        _read_canonical_kv(schema_path)
    catch err
        throw(MalformedSchemaFile(schema_path, sprint(showerror, err)))
    end
    entries === nothing && throw(MissingRunStoreSchema(schema_path))
    name = get(entries, "schema", "")
    version = tryparse(UInt64, get(entries, "version", ""))
    version === nothing && throw(MalformedSchemaFile(
        schema_path, "the version entry is not an unsigned integer"))
    if name != RUNSTORE_SCHEMA_NAME || version != RUNSTORE_SCHEMA_VERSION
        throw(UnsupportedSchemaVersion(
            name, version, RUNSTORE_SCHEMA_NAME, [RUNSTORE_SCHEMA_VERSION]))
    end

    metadata_path = joinpath(_native_metadata_dir(rootp), "run.jls")
    isfile(metadata_path) || throw(CorruptStore(
        "", 0, metadata_path, "the run metadata file is missing"))
    digest_path = joinpath(_native_metadata_dir(rootp), "run.sha256")
    isfile(digest_path) || throw(CorruptStore(
        "", 0, digest_path, "the run metadata digest file is missing"))
    expected = strip(read(digest_path, String))
    actual = _file_digest(metadata_path)
    expected == actual || throw(ChecksumMismatch(
        "", 0, metadata_path, expected, actual))
    metadata = try
        open(deserialize, metadata_path)
    catch err
        throw(CorruptStore(
            "", 0, metadata_path,
            "the run metadata could not be deserialized ($(typeof(err)))"))
    end
    metadata isa RunMetadata || throw(CorruptStore(
        "", 0, metadata_path,
        "the run metadata has the unexpected type $(typeof(metadata))"))

    store = NativeRunStore(rootp, metadata, mode)
    mode === :write && _acquire_lease!(store)
    return store
end

"""
    open_run(store::NativeRunStore; metadata::RunMetadata) -> NativeRunStore

Reuse an already opened native store under identical metadata.

Conflicting metadata is an error ([`ConflictingRunMetadata`](@ref)): a run
has exactly one identity.
"""
function open_run(store::NativeRunStore; metadata::RunMetadata)
    metadata == store.metadata || throw(ConflictingRunMetadata(
        store.metadata.run_id,
        _metadata_summary(store.metadata),
        _metadata_summary(metadata),
    ))
    return store
end

"""
    initialize_run!(store::NativeRunStore; metadata::RunMetadata) -> NativeRunStore

Reuse an already opened native store under identical metadata.

Conflicting metadata is an error ([`ConflictingRunMetadata`](@ref)).
"""
function initialize_run!(store::NativeRunStore; metadata::RunMetadata)
    metadata == store.metadata || throw(ConflictingRunMetadata(
        store.metadata.run_id,
        _metadata_summary(store.metadata),
        _metadata_summary(metadata),
    ))
    return store
end

"""
    close_run!(store::NativeRunStore) -> Nothing

Close a native run store.

The writer lease owned by this store is released, and the store is marked
closed: later writes raise [`StoreClosed`](@ref).

A pending generation that was never published is **discarded** with a
warning. Pending records were never durable, so discarding them loses
nothing that was ever persisted; the generation can be recomputed from its
deterministic seeds.
"""
function close_run!(store::NativeRunStore)
    store.closed && return nothing
    if store.lease_path !== nothing
        isdir(store.lease_path) &&
            rm(store.lease_path; recursive = true, force = true)
        store.lease_path = nothing
    end
    if store.pending_generation !== nothing
        @warn "closing a native run store discards a pending generation that " *
              "was never published; pending records are not durable" generation =
              store.pending_generation run_id = store.metadata.run_id
    end
    store.pending_generation = nothing
    store.pending_population = nothing
    empty!(store.pending_evaluations)
    empty!(store.pending_keys)
    store.closed = true
    return nothing
end

"""
    schema_version(::Type{NativeRunStore}) -> UInt64

The storage schema version written by this build.
"""
schema_version(::Type{NativeRunStore}) = RUNSTORE_SCHEMA_VERSION

function _check_usable(store::NativeRunStore)
    store.closed && throw(StoreClosed())
    return store
end

function _check_writable(store::NativeRunStore)
    _check_usable(store)
    store.mode === :write || throw(ReadOnlyRunStore())
    return store
end

# --- writer lease -----------------------------------------------------------

function _native_hostname()
    try
        return string(Sockets.gethostname())
    catch
        return "unknown"
    end
end

function _native_nonce()
    return string(time_ns(), base = 16) * "-" * string(getpid(), base = 16)
end

function _lease_path(root::AbstractString)
    return joinpath(_native_metadata_dir(root), "writer.lease")
end

# The writer lease is a directory: creating it with `mkdir` is an atomic
# exclusive create on POSIX and on Windows, and Julia exposes no
# `O_EXCL` mode for `open`. The holder record lives inside it.
function _lease_holder_path(root::AbstractString)
    return joinpath(_lease_path(root), "holder")
end

function _acquire_lease!(store::NativeRunStore)
    path = _lease_path(store.root)
    entries = Pair{String,String}[
        "schema" => RUNSTORE_SCHEMA_NAME,
        "version" => string(RUNSTORE_SCHEMA_VERSION),
        "run_id" => _canonical_encode(store.metadata.run_id),
        "hostname" => _canonical_encode(_native_hostname()),
        "pid" => string(getpid()),
        "acquired_at" => _native_timestamp(now(UTC)),
        "nonce" => _canonical_encode(_native_nonce()),
    ]
    try
        mkdir(path)
    catch
        holder = _read_lease_holder(path)
        if holder === nothing
            throw(StaleWriterLease(
                path, store.metadata.run_id, "", 0, "",
                "the lease exists but could not be interpreted"))
        end
        throw(WriterAlreadyActive(
            path, holder.run_id, holder.hostname, holder.pid, holder.acquired_at))
    end
    try
        _write_canonical_kv(joinpath(path, "holder"), entries)
    catch
        rm(path; recursive = true, force = true)
        rethrow()
    end
    store.lease_path = path
    return store
end

function _read_lease_holder(path::AbstractString)
    holder_path = joinpath(path, "holder")
    isfile(holder_path) || return nothing
    try
        entries = _read_canonical_kv(holder_path)
        entries === nothing && return nothing
        pid = tryparse(Int, get(entries, "pid", ""))
        pid === nothing && return nothing
        return (
            run_id = _canonical_decode(get(entries, "run_id", "")),
            hostname = _canonical_decode(get(entries, "hostname", "")),
            pid = pid,
            acquired_at = get(entries, "acquired_at", ""),
        )
    catch
        return nothing
    end
end

# --- committed generations ---------------------------------------------------

function _native_committed_generations(store::NativeRunStore)
    directory = _native_generations_dir(store.root)
    isdir(directory) || return Int64[]
    generations = Int64[]
    for entry in readdir(directory)
        matched = match(r"^generation_(\d{6})$", entry)
        matched === nothing && continue
        generation = parse(Int64, matched.captures[1])
        generation_dir = joinpath(directory, entry)
        isdir(generation_dir) || continue
        isfile(joinpath(generation_dir, NATIVE_COMPLETE_FILE)) || continue
        push!(generations, generation)
    end
    sort!(generations)
    return generations
end

function _native_generation_directory(store::NativeRunStore, generation::Integer)
    return _native_generation_dir(store.root, generation)
end

# A generation whose directory exists but that was never published is
# incomplete, not missing. An explicit request for it fails loudly instead of
# quietly returning nothing.
function _native_check_requested_generations(store::NativeRunStore,
                                             generation,
                                             committed::Set{Int64})
    generation === nothing && return nothing
    requested = generation isa Integer ? (Int64(generation),) :
               (Int64(g) for g in generation)
    for gen in requested
        gen in committed && continue
        isdir(_native_generation_directory(store, gen)) && throw(IncompleteGeneration(
            store.metadata.run_id, gen, [NATIVE_COMPLETE_FILE]))
    end
    return nothing
end

# --- writing ----------------------------------------------------------------

"""
    write_population!(store::NativeRunStore, record::PopulationRecord) -> PopulationRecord

Record the population of the next writable generation.

The store must be in write mode, and the generation must directly follow the
latest committed generation. A generation that is already committed is never
modified ([`GenerationAlreadyComplete`](@ref)), and a generation that already
has a population never gets a second one
([`DuplicatePopulation`](@ref)).

The population is held in the store session until it is published by
[`finalize_generation!`](@ref); it is not durable before that.
"""
function write_population!(store::NativeRunStore, record::PopulationRecord)
    _check_writable(store)
    generation = record.generation
    committed = Set(_native_committed_generations(store))
    generation in committed && throw(GenerationAlreadyComplete(generation))
    expected = _next_writable_generation(committed)
    generation == expected || throw(GenerationGap(generation, expected))
    if store.pending_generation !== nothing && store.pending_generation != generation
        throw(PendingGenerationConflict(store.metadata.run_id, generation))
    end
    store.pending_population === nothing || throw(DuplicatePopulation(generation))
    store.pending_generation = generation
    store.pending_population = copy(record)
    return record
end

"""
    write_evaluation!(store::NativeRunStore, record::EvaluationRecord) -> EvaluationRecord

Record one evaluation of the next writable generation.

The evaluation's method must match the method of the run
([`MethodMismatch`](@ref)), and its identity must be unique inside the
generation ([`DuplicateEvaluation`](@ref)).

The evaluation is held in the store session until it is published by
[`finalize_generation!`](@ref); it is not durable before that.
"""
function write_evaluation!(store::NativeRunStore, record::EvaluationRecord)
    _check_writable(store)
    generation = record.context.generation
    committed = Set(_native_committed_generations(store))
    generation in committed && throw(GenerationAlreadyComplete(generation))
    expected = _next_writable_generation(committed)
    generation == expected || throw(GenerationGap(generation, expected))
    record.context.method == store.metadata.method || throw(MethodMismatch(
        record.context.method, store.metadata.method))
    key = EvaluationKey(record)
    key in store.pending_keys && throw(DuplicateEvaluation(key))
    push!(store.pending_evaluations, copy(record))
    push!(store.pending_keys, key)
    return record
end

"""
    finalize_generation!(store::NativeRunStore, generation::Integer;
                         expected_evaluations=nothing) -> GenerationCompletion

Publish a generation and make it durable.

The generation must have a population
([`MissingPopulation`](@ref)), must be the next writable generation, and
must not be committed already. When `expected_evaluations` is supplied, the
recorded identities must match the expected ones exactly
([`MissingExpectedEvaluation`](@ref) and [`UnexpectedEvaluation`](@ref)).

Publication writes the population and the evaluation partitions, the
manifest, and the `COMPLETE` marker into a staging directory, verifies
them, and then moves the staging directory to its canonical name with a
single rename. An interruption before that rename leaves only a staging
directory, which is never visible.

This is the durability boundary of the run store: before it, the records of
the generation exist only in the store session.
"""
function finalize_generation!(store::NativeRunStore, generation::Integer;
                              expected_evaluations=nothing)
    _check_writable(store)
    gen = _checked_int64(generation, "generation")
    committed = Set(_native_committed_generations(store))
    gen in committed && throw(GenerationAlreadyComplete(gen))
    expected = _next_writable_generation(committed)
    gen == expected || throw(GenerationGap(gen, expected))
    if store.pending_generation !== nothing && store.pending_generation != gen
        throw(PendingGenerationConflict(store.metadata.run_id, gen))
    end
    population = store.pending_population
    population === nothing && throw(MissingPopulation(gen))
    records = store.pending_evaluations

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

    completion = _publish_generation!(store, gen, population, records,
                                      expected_evaluations)

    store.pending_generation = nothing
    store.pending_population = nothing
    empty!(store.pending_evaluations)
    empty!(store.pending_keys)
    return completion
end

function _publish_generation!(store::NativeRunStore,
                              generation::Int64,
                              population::PopulationRecord,
                              records::Vector{EvaluationRecord},
                              expected_evaluations)
    root = store.root
    staging = joinpath(_native_tmp_dir(root),
                        _native_generation_name(generation) * "." * _native_nonce())
    final_dir = _native_generation_dir(root, generation)
    isdir(final_dir) && throw(TargetExists(final_dir))

    mkpath(staging)
    published = false
    try
        population_path = joinpath(staging, NATIVE_POPULATION_FILE)
        evaluation_path = joinpath(staging, NATIVE_EVALUATION_FILE)

        ordered = sort!(collect(records); by = r -> EvaluationKey(r))
        _write_partition(
            population_path,
            NativePartitionHeader(
                RUNSTORE_SCHEMA_NAME, RUNSTORE_SCHEMA_VERSION, :population,
                store.metadata.run_id, generation, 1, now(UTC)),
            [population],
        )
        _write_partition(
            evaluation_path,
            NativePartitionHeader(
                RUNSTORE_SCHEMA_NAME, RUNSTORE_SCHEMA_VERSION, :evaluations,
                store.metadata.run_id, generation, length(ordered), now(UTC)),
            ordered,
        )

        # Reopen the staged partitions and revalidate them.
        _verify_partition(population_path, store.metadata.run_id, generation,
                          :population, 1)
        _verify_partition(evaluation_path, store.metadata.run_id, generation,
                          :evaluations, length(ordered))

        population_digest = _file_digest(population_path)
        population_bytes = filesize(population_path)
        evaluation_digest = _file_digest(evaluation_path)
        evaluation_bytes = filesize(evaluation_path)
        key_set_digest = _evaluation_key_set_hash(
            EvaluationKey(r) for r in ordered)
        plan_digest = expected_evaluations === nothing ? nothing :
            _evaluation_key_set_hash(
                item isa EvaluationRecord ? EvaluationKey(item) : item
                for item in expected_evaluations)
        omission_count = _count_omissions(ordered)

        manifest_entries = Pair{String,String}[
            "schema" => RUNSTORE_SCHEMA_NAME,
            "version" => string(RUNSTORE_SCHEMA_VERSION),
            "run_id" => _canonical_encode(store.metadata.run_id),
            "generation" => string(generation),
            "metadata_sha256" => _file_digest(
                joinpath(_native_metadata_dir(root), "run.jls")),
            "population_file" => NATIVE_POPULATION_FILE,
            "population_sha256" => population_digest,
            "population_bytes" => string(population_bytes),
            "population_count" => "1",
            "evaluation_file" => NATIVE_EVALUATION_FILE,
            "evaluation_sha256" => evaluation_digest,
            "evaluation_bytes" => string(evaluation_bytes),
            "evaluation_count" => string(length(ordered)),
            "evaluation_key_set_sha256" => key_set_digest,
            "omission_count" => string(omission_count),
            "package_version" => _canonical_encode(store.metadata.package_version),
            "julia_version" => _canonical_encode(store.metadata.julia_version),
            "completed_at" => _native_timestamp(now(UTC)),
        ]
        plan_digest === nothing ||
            push!(manifest_entries, "expected_plan_sha256" => plan_digest)

        manifest_path = joinpath(staging, NATIVE_MANIFEST_FILE)
        _write_canonical_kv(manifest_path, manifest_entries)
        manifest_digest = _file_digest(manifest_path)
        _read_manifest(manifest_path)

        complete_path = joinpath(staging, NATIVE_COMPLETE_FILE)
        _write_canonical_kv(
            complete_path,
            Pair{String,String}[
                "schema" => RUNSTORE_SCHEMA_NAME,
                "version" => string(RUNSTORE_SCHEMA_VERSION),
                "run_id" => _canonical_encode(store.metadata.run_id),
                "generation" => string(generation),
                "manifest_file" => NATIVE_MANIFEST_FILE,
                "manifest_sha256" => manifest_digest,
                "completed_at" => _native_timestamp(now(UTC)),
            ],
        )

        # Validate the whole staging directory before it becomes canonical.
        _verify_generation_directory(store, staging, generation)

        isdir(final_dir) && throw(TargetExists(final_dir))
        _move_directory(staging, final_dir)
        published = true

        _verify_generation_directory(store, final_dir, generation)
    finally
        if !published && isdir(staging)
            try
                rm(staging; recursive = true, force = true)
            catch
                # A leftover staging directory is reported by `validate_store`.
            end
        end
    end

    return GenerationCompletion(
        generation, length(population.particles), length(records), now(UTC))
end

function _count_omissions(records)
    count = 0
    for record in records
        record.summary.state === :omitted && (count += 1)
        for (_, value) in record.provenance.entries
            value isa RecordedOmitted && (count += 1)
        end
    end
    return count
end

function _verify_partition(path::AbstractString,
                           run_id::AbstractString,
                           generation::Int64,
                           kind::Symbol,
                           expected_count::Int)
    header, records = _read_partition(path, run_id, generation, kind)
    length(records) == expected_count || throw(CorruptStore(
        run_id, generation, path,
        "the partition holds $(length(records)) record(s) but $(expected_count) " *
        "were expected"))
    if kind === :population
        record = records[1]
        record isa PopulationRecord || throw(CorruptStore(
            run_id, generation, path, "the partition does not hold a population"))
        record.generation == generation || throw(CorruptStore(
            run_id, generation, path,
            "the population belongs to generation $(record.generation)"))
    else
        previous = nothing
        for record in records
            record isa EvaluationRecord || throw(CorruptStore(
                run_id, generation, path, "the partition does not hold evaluations"))
            key = EvaluationKey(record)
            previous !== nothing && key == previous && throw(CorruptStore(
                run_id, generation, path, "duplicate evaluation identity $(key)"))
            previous = key
        end
        sort!(records; by = r -> EvaluationKey(r))
        for index in 2:length(records)
            EvaluationKey(records[index]) < EvaluationKey(records[index - 1]) &&
                throw(CorruptStore(
                    run_id, generation, path,
                    "the evaluations are not stored in the canonical order"))
        end
    end
    return nothing
end

function _move_directory(source::AbstractString, target::AbstractString)
    try
        Base.Filesystem.rename(source, target)
    catch err
        if err isa Base.IOError || err isa SystemError
            message = sprint(showerror, err)
            if occursin("EXDEV", message) || occursin("cross-device", lowercase(message))
                throw(CrossDeviceStaging(source, target))
            end
            throw(CorruptStore("", 0, target,
                "the staging directory could not be moved into place ($(message))"))
        end
        rethrow(err)
    end
    return target
end

# --- reading ----------------------------------------------------------------

function _read_manifest(path::AbstractString)
    entries = try
        _read_canonical_kv(path)
    catch err
        throw(MalformedManifest(path, sprint(showerror, err)))
    end
    entries === nothing && throw(MissingManifest("", 0, path))
    return entries
end

function _read_marker(path::AbstractString)
    entries = try
        _read_canonical_kv(path)
    catch err
        throw(MalformedMarker(path, sprint(showerror, err)))
    end
    entries === nothing && throw(MalformedMarker(path, "the marker is empty"))
    return entries
end

function _verify_generation_directory(store::NativeRunStore,
                                      directory::AbstractString,
                                      generation::Int64)
    run_id = store.metadata.run_id
    complete_path = joinpath(directory, NATIVE_COMPLETE_FILE)
    isfile(complete_path) || throw(IncompleteGeneration(
        run_id, generation, [NATIVE_COMPLETE_FILE]))
    marker = _read_marker(complete_path)
    get(marker, "run_id", "") == _canonical_encode(run_id) || throw(CorruptStore(
        run_id, generation, complete_path, "the marker belongs to another run"))
    get(marker, "generation", "") == string(generation) || throw(CorruptStore(
        run_id, generation, complete_path,
        "the marker names generation $(get(marker, "generation", ""))"))
    manifest_file = get(marker, "manifest_file", NATIVE_MANIFEST_FILE)
    manifest_path = joinpath(directory, manifest_file)
    isfile(manifest_path) || throw(MissingManifest(run_id, generation, manifest_path))
    manifest_digest = _file_digest(manifest_path)
    manifest_digest == get(marker, "manifest_sha256", "") || throw(ChecksumMismatch(
        run_id, generation, manifest_path,
        get(marker, "manifest_sha256", ""), manifest_digest))
    manifest = _read_manifest(manifest_path)
    get(manifest, "run_id", "") == _canonical_encode(run_id) || throw(CorruptStore(
        run_id, generation, manifest_path, "the manifest belongs to another run"))
    get(manifest, "generation", "") == string(generation) || throw(CorruptStore(
        run_id, generation, manifest_path,
        "the manifest names generation $(get(manifest, "generation", ""))"))
    return manifest
end

function _verify_generation_data(store::NativeRunStore,
                                 directory::AbstractString,
                                 generation::Int64,
                                 manifest)
    run_id = store.metadata.run_id
    population_path = joinpath(directory, get(manifest, "population_file", ""))
    isfile(population_path) || throw(CorruptStore(
        run_id, generation, population_path, "the population partition is missing"))
    population_digest = _file_digest(population_path)
    population_digest == get(manifest, "population_sha256", "") ||
        throw(ChecksumMismatch(
            run_id, generation, population_path,
            get(manifest, "population_sha256", ""), population_digest))
    string(filesize(population_path)) == get(manifest, "population_bytes", "") ||
        throw(CorruptStore(
            run_id, generation, population_path,
            "the population partition has an unexpected size"))

    evaluation_path = joinpath(directory, get(manifest, "evaluation_file", ""))
    isfile(evaluation_path) || throw(CorruptStore(
        run_id, generation, evaluation_path, "the evaluation partition is missing"))
    evaluation_digest = _file_digest(evaluation_path)
    evaluation_digest == get(manifest, "evaluation_sha256", "") ||
        throw(ChecksumMismatch(
            run_id, generation, evaluation_path,
            get(manifest, "evaluation_sha256", ""), evaluation_digest))
    string(filesize(evaluation_path)) == get(manifest, "evaluation_bytes", "") ||
        throw(CorruptStore(
            run_id, generation, evaluation_path,
            "the evaluation partition has an unexpected size"))
    return nothing
end

"""
    read_population(store::NativeRunStore, generation::Integer) -> PopulationRecord

Read the population of a committed generation.

The COMPLETE marker, the manifest, and the partition digests are verified
before anything is deserialized, and the record invariants are rechecked
afterwards. A generation that fails verification raises
[`CorruptStore`](@ref) or [`ChecksumMismatch`](@ref): an explicitly
requested generation never falls back to an older one.

The returned record is a copy; the caller cannot mutate the stored history.
"""
function read_population(store::NativeRunStore, generation::Integer)
    _check_usable(store)
    gen = _checked_int64(generation, "generation")
    directory = _native_generation_directory(store, gen)
    isdir(directory) || throw(MissingGeneration(gen))
    manifest = _verify_generation_directory(store, directory, gen)
    _verify_generation_data(store, directory, gen, manifest)
    population_path = joinpath(directory, get(manifest, "population_file", ""))
    _, records = _read_partition(population_path, store.metadata.run_id, gen,
                                 :population)
    length(records) == 1 || throw(CorruptStore(
        store.metadata.run_id, gen, population_path,
        "the population partition holds $(length(records)) record(s)"))
    record = records[1]
    record isa PopulationRecord || throw(CorruptStore(
        store.metadata.run_id, gen, population_path,
        "the partition does not hold a population"))
    record.generation == gen || throw(CorruptStore(
        store.metadata.run_id, gen, population_path,
        "the population belongs to generation $(record.generation)"))
    return copy(record)
end

"""
    read_evaluations(store::NativeRunStore; generation=nothing, particle=nothing,
                     proposal=nothing, attempt=nothing, replicate=nothing,
                     status=nothing, failure_category=nothing,
                     min_wall_time_s=nothing, max_wall_time_s=nothing)
        -> Vector{EvaluationRecord}

Read recorded evaluations in the deterministic order of the run-history
contract.

The filters and the ordering are exactly those of the in-memory backend: the
result is ordered by generation and then by
[`EvaluationKey`](@ref), and the filters are `generation`, `particle`,
`proposal`, `attempt`, `replicate`, `status`, `failure_category`, and the
inclusive wall-time bounds.

Only committed generations are read, and every read verifies the COMPLETE
marker, the manifest, and the partition digests. A generation that fails
verification raises [`CorruptStore`](@ref) or
[`ChecksumMismatch`](@ref) instead of being skipped.
"""
function read_evaluations(
    store::NativeRunStore;
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
    _check_usable(store)
    committed = Set(_native_committed_generations(store))
    _native_check_requested_generations(store, generation, committed)
    selected = _selected_generations(committed, generation)
    min_t = _checked_bound(min_wall_time_s, "min_wall_time_s")
    max_t = _checked_bound(max_wall_time_s, "max_wall_time_s")
    min_t === nothing || max_t === nothing || min_t <= max_t || throw(ArgumentError(
        "min_wall_time_s ($(min_t)) must not exceed max_wall_time_s ($(max_t))"))

    out = EvaluationRecord[]
    for gen in selected
        directory = _native_generation_directory(store, gen)
        isdir(directory) || throw(MissingGeneration(gen))
        manifest = _verify_generation_directory(store, directory, gen)
        _verify_generation_data(store, directory, gen, manifest)
        evaluation_path = joinpath(directory, get(manifest, "evaluation_file", ""))
        _each_partition_record(evaluation_path, store.metadata.run_id, gen,
                               :evaluations) do record
            record isa EvaluationRecord || throw(CorruptStore(
                store.metadata.run_id, gen, evaluation_path,
                "the partition does not hold evaluations"))
            _matches_index(record.context.particle, particle) || return nothing
            _matches_index(record.context.proposal, proposal) || return nothing
            _matches_index(record.context.attempt, attempt) || return nothing
            _matches_index(record.context.replicate, replicate) || return nothing
            _matches_symbol(record.status, status) || return nothing
            _matches_failure_category(record, failure_category) || return nothing
            if min_t !== nothing || max_t !== nothing
                wall_time = record.wall_time_s
                min_t !== nothing && wall_time < min_t && return nothing
                max_t !== nothing && wall_time > max_t && return nothing
            end
            push!(out, copy(record))
            return nothing
        end
    end
    sort!(out; by = r -> (r.context.generation, EvaluationKey(r)))
    return out
end

"""
    generations(store::NativeRunStore) -> Vector{Int}

The committed generations of the store, in ascending order.

A generation is committed when its directory exists and holds a `COMPLETE`
marker. The marker, the manifest, and the partition digests are verified
when a generation is read, and every violation is reported by
[`validate_store`](@ref); a corrupt generation is never silently skipped.
"""
function generations(store::NativeRunStore)
    _check_usable(store)
    return _native_committed_generations(store)
end

"""
    latest_generation(store::NativeRunStore) -> Union{Nothing,Int}

The latest committed generation, or `nothing` when the store is empty.
"""
function latest_generation(store::NativeRunStore)
    _check_usable(store)
    committed = _native_committed_generations(store)
    isempty(committed) && return nothing
    return maximum(committed)
end

# --- validation -------------------------------------------------------------

"""
    validate_store(store::NativeRunStore) -> StoreValidationReport

Validate every invariant of a native run store.

The report lists the errors (schema, metadata, digest, marker, manifest,
partition digests and sizes, record counts, record invariants, duplicate
identities, ordering, expected plans, omission counts, generation continuity,
orphan artifacts, leftover staging directories), the warnings (a writer lease
that is held, a pending generation), the committed generations, and every
visible omission.

Validation never repairs, deletes, normalizes, or rewrites the store.
"""
function validate_store(store::NativeRunStore)
    errors = String[]
    warnings = String[]
    omissions = String[]
    run_id = store.metadata.run_id

    if store.metadata.storage_schema_name != RUNSTORE_SCHEMA_NAME ||
            store.metadata.storage_schema_version != RUNSTORE_SCHEMA_VERSION
        push!(errors, string(
            "unsupported storage schema ", store.metadata.storage_schema_name, "/",
            store.metadata.storage_schema_version, "; this build reads ",
            RUNSTORE_SCHEMA_NAME, "/", RUNSTORE_SCHEMA_VERSION))
    end

    schema_path = joinpath(_native_metadata_dir(store.root), "schema")
    if !isfile(schema_path)
        push!(errors, "the storage schema file $(schema_path) is missing")
    else
        entries = try
            _read_canonical_kv(schema_path)
        catch err
            push!(errors, "the storage schema file is malformed ($(sprint(showerror, err)))")
            entries = nothing
        end
        if entries !== nothing
            version = tryparse(UInt64, get(entries, "version", ""))
            if get(entries, "schema", "") != RUNSTORE_SCHEMA_NAME ||
                    version != RUNSTORE_SCHEMA_VERSION
                push!(errors, string(
                    "unsupported storage schema ", get(entries, "schema", ""), "/",
                    get(entries, "version", ""), "; this build reads ",
                    RUNSTORE_SCHEMA_NAME, "/", RUNSTORE_SCHEMA_VERSION))
            end
        end
    end

    metadata_path = joinpath(_native_metadata_dir(store.root), "run.jls")
    digest_path = joinpath(_native_metadata_dir(store.root), "run.sha256")
    if !isfile(metadata_path)
        push!(errors, "the run metadata file $(metadata_path) is missing")
    elseif !isfile(digest_path)
        push!(errors, "the run metadata digest file $(digest_path) is missing")
    else
        expected = strip(read(digest_path, String))
        actual = _file_digest(metadata_path)
        expected == actual || push!(errors, string(
            "the run metadata digest does not match: expected ", expected,
            " but the file hashes to ", actual))
    end

    lease_path = _lease_path(store.root)
    if isdir(lease_path) && store.lease_path === nothing
        # The lease of this very session is not an anomaly; a lease that is
        # held by somebody else is reported.
        holder = _read_lease_holder(lease_path)
        if holder === nothing
            push!(warnings, string(
                "the writer lease ", lease_path, " exists but cannot be interpreted"))
        else
            push!(warnings, string(
                "a writer lease is held by run ", repr(holder.run_id), " on host ",
                repr(holder.hostname), " (pid ", holder.pid, ", acquired ",
                holder.acquired_at, "); it is never removed automatically"))
        end
    end

    if store.pending_generation !== nothing
        push!(warnings, string(
            "generation ", store.pending_generation,
            " is pending and was never published; pending records are not durable"))
    end

    committed = _native_committed_generations(store)
    expected_generation = 1
    for generation in committed
        generation == expected_generation || push!(errors, string(
            "generations must be contiguous and start at 1; expected generation ",
            expected_generation, " but found ", generation))
        expected_generation = generation + 1

        directory = _native_generation_directory(store, generation)
        manifest = try
            _verify_generation_directory(store, directory, generation)
        catch err
            push!(errors, sprint(showerror, err))
            continue
        end

        # A generation directory holds exactly the four canonical artifacts.
        unexpected = setdiff(readdir(directory),
                             [NATIVE_POPULATION_FILE, NATIVE_EVALUATION_FILE,
                              NATIVE_MANIFEST_FILE, NATIVE_COMPLETE_FILE])
        for entry in unexpected
            push!(errors, sprint(showerror, OrphanFile(
                joinpath(directory, entry),
                "the generation directory holds an unexpected artifact")))
        end
        try
            _verify_generation_data(store, directory, generation, manifest)
        catch err
            push!(errors, sprint(showerror, err))
            continue
        end

        population_path = joinpath(directory, get(manifest, "population_file", ""))
        evaluation_path = joinpath(directory, get(manifest, "evaluation_file", ""))
        try
            _verify_partition(population_path, run_id, generation, :population, 1)
        catch err
            push!(errors, sprint(showerror, err))
        end
        evaluation_count = tryparse(Int, get(manifest, "evaluation_count", ""))
        if evaluation_count === nothing
            push!(errors, "the manifest of generation $(generation) has no valid evaluation count")
        else
            try
                _verify_partition(evaluation_path, run_id, generation,
                                  :evaluations, evaluation_count)
            catch err
                push!(errors, sprint(showerror, err))
            end
        end

        try
            _, records = _read_partition(evaluation_path, run_id, generation,
                                         :evaluations)
            seen = Set{EvaluationKey}()
            for record in records
                key = EvaluationKey(record)
                key in seen && push!(errors, string(
                    "duplicate evaluation identity ", key, " in generation ",
                    generation))
                push!(seen, key)
                _collect_omissions!(omissions, record, generation)
            end
            digest = _evaluation_key_set_hash(seen)
            expected_digest = get(manifest, "evaluation_key_set_sha256", "")
            digest == expected_digest || push!(errors, string(
                "the evaluation identity digest of generation ", generation,
                " does not match the manifest"))
            if haskey(manifest, "expected_plan_sha256")
                manifest["expected_plan_sha256"] == digest || push!(errors, string(
                    "the recorded evaluations of generation ", generation,
                    " do not match the expected plan"))
            end
            omission_count = _count_omissions(records)
            recorded_omissions = tryparse(Int, get(manifest, "omission_count", ""))
            if recorded_omissions !== nothing && recorded_omissions != omission_count
                push!(errors, string(
                    "the omission count of generation ", generation, " is ",
                    omission_count, " but the manifest records ",
                    recorded_omissions))
            end
        catch err
            push!(errors, sprint(showerror, err))
        end
    end

    for entry in readdir(_native_generations_dir(store.root))
        matched = match(r"^generation_(\d{6})$", entry)
        if matched === nothing
            push!(warnings, "unexpected entry $(entry) in the generations directory")
            continue
        end
        generation = parse(Int64, matched.captures[1])
        generation in committed && continue
        push!(errors, string(
            "generation ", generation, " has artifacts but no COMPLETE marker; ",
            "it was never published"))
    end

    tmp_dir = _native_tmp_dir(store.root)
    if isdir(tmp_dir)
        for entry in readdir(tmp_dir)
            push!(warnings, string(
                "the staging directory ", entry, " was left behind by an ",
                "interrupted publication and is ignored by readers"))
        end
    end

    return StoreValidationReport(
        isempty(errors), errors, warnings, committed, Int64[], omissions)
end
