# ============================================================================
# Native-store error types
# ============================================================================
#
# Errors of the file-backed backend. They all derive from `RunStoreError`, so
# a caller can catch every run-history failure with one type, and they all
# carry the context that makes a failure actionable: the run, the generation,
# and the file that was involved.
#
# Raw `Serialization` failures are never surfaced: they are wrapped in
# `CorruptStore` or `TruncatedPartition` with the run, generation, and path.

"""
    MissingRunStoreSchema(path)

The storage schema file of a run store is missing.

A run store without a schema cannot be read safely: the reader does not know
which layout it is looking at.
"""
struct MissingRunStoreSchema <: RunStoreError
    path::String
end

Base.showerror(io::IO, e::MissingRunStoreSchema) = print(
    io,
    "MissingRunStoreSchema: no storage schema file at $(e.path)",
)

"""
    MalformedSchemaFile(path, reason)

The storage schema file exists but does not follow the canonical encoding.
"""
struct MalformedSchemaFile <: RunStoreError
    path::String
    reason::String
end

Base.showerror(io::IO, e::MalformedSchemaFile) = print(
    io,
    "MalformedSchemaFile: the storage schema file $(e.path) is malformed ($(e.reason))",
)

"""
    WriterAlreadyActive(path, run_id, hostname, pid, acquired_at)

Another writer holds the run store.

The writer lease is never broken automatically. If the recorded holder is
gone, an operator has to remove the lease file by hand; see the run-history
documentation for the procedure.
"""
struct WriterAlreadyActive <: RunStoreError
    path::String
    run_id::String
    hostname::String
    pid::Int
    acquired_at::String
end

Base.showerror(io::IO, e::WriterAlreadyActive) = print(
    io,
    "WriterAlreadyActive: the writer lease $(e.path) is held by run " *
    "$(repr(e.run_id)) on host $(repr(e.hostname)) (pid $(e.pid), acquired " *
    "$(e.acquired_at)); a stale lease is never removed automatically — " *
    "verify that the holder is gone and remove the lease file manually",
)

"""
    StaleWriterLease(path, run_id, hostname, pid, acquired_at, reason)

A writer lease exists but could not be interpreted, so the holder is unknown.

The lease is never removed automatically; an operator has to inspect the file
and remove it by hand.
"""
struct StaleWriterLease <: RunStoreError
    path::String
    run_id::String
    hostname::String
    pid::Int
    acquired_at::String
    reason::String
end

Base.showerror(io::IO, e::StaleWriterLease) = print(
    io,
    "StaleWriterLease: the writer lease $(e.path) exists but cannot be " *
    "interpreted ($(e.reason)); it is never removed automatically — inspect " *
    "the file and remove it manually",
)

"""
    CrossDeviceStaging(source, target)

A file was about to be moved across filesystems.

The native store only moves files inside the run directory, so a cross-device
move means the layout assumption is violated.
"""
struct CrossDeviceStaging <: RunStoreError
    source::String
    target::String
end

Base.showerror(io::IO, e::CrossDeviceStaging) = print(
    io,
    "CrossDeviceStaging: cannot move $(e.source) to $(e.target); the native " *
    "store only publishes inside one filesystem",
)

"""
    TargetExists(path)

A publication target already exists.

The native store never overwrites a canonical artifact: a target that exists
means the generation was already published (or the directory is inconsistent)
and the write is refused.
"""
struct TargetExists <: RunStoreError
    path::String
end

Base.showerror(io::IO, e::TargetExists) = print(
    io,
    "TargetExists: the publication target $(e.path) already exists; the " *
    "native store never overwrites a canonical artifact",
)

"""
    IncompleteGeneration(run_id, generation, missing)

A generation directory exists but is not complete: the `COMPLETE` marker is
missing, so the generation was never published.
"""
struct IncompleteGeneration <: RunStoreError
    run_id::String
    generation::Int64
    missing::Vector{String}
end

Base.showerror(io::IO, e::IncompleteGeneration) = print(
    io,
    "IncompleteGeneration: generation $(e.generation) of run $(repr(e.run_id)) " *
    "is missing $(join(e.missing, ", ")); it was never published",
)

"""
    CorruptStore(run_id, generation, path, reason)

A native file could not be read as the run store's schema requires.

The underlying cause (a deserialization error, a wrong type, a foreign run
identity) is reported in the reason; the raw exception is never surfaced.
"""
struct CorruptStore <: RunStoreError
    run_id::String
    generation::Int64
    path::String
    reason::String
end

Base.showerror(io::IO, e::CorruptStore) = print(
    io,
    "CorruptStore: $(e.path) is corrupt (run $(repr(e.run_id)), generation " *
    "$(e.generation)): $(e.reason)",
)

"""
    ChecksumMismatch(run_id, generation, path, expected, actual)

The SHA-256 digest of a file does not match the digest that was recorded.

This detects accidental corruption and truncation; it does not authenticate
against a writer that rewrites data, manifest, and marker consistently.
"""
struct ChecksumMismatch <: RunStoreError
    run_id::String
    generation::Int64
    path::String
    expected::String
    actual::String
end

Base.showerror(io::IO, e::ChecksumMismatch) = print(
    io,
    "ChecksumMismatch: $(e.path) has SHA-256 $(e.actual) but $(e.expected) was " *
    "recorded (run $(repr(e.run_id)), generation $(e.generation))",
)

"""
    TruncatedPartition(run_id, generation, path, expected_count, actual_count)

A native partition ends before all of its records were read.
"""
struct TruncatedPartition <: RunStoreError
    run_id::String
    generation::Int64
    path::String
    expected_count::Int
    actual_count::Int
end

Base.showerror(io::IO, e::TruncatedPartition) = print(
    io,
    "TruncatedPartition: $(e.path) holds $(e.actual_count) of " *
    "$(e.expected_count) record(s) (run $(repr(e.run_id)), generation " *
    "$(e.generation))",
)

"""
    MissingManifest(run_id, generation, path)

The manifest of a generation is missing.
"""
struct MissingManifest <: RunStoreError
    run_id::String
    generation::Int64
    path::String
end

Base.showerror(io::IO, e::MissingManifest) = print(
    io,
    "MissingManifest: the manifest $(e.path) of generation $(e.generation) " *
    "is missing (run $(repr(e.run_id)))",
)

"""
    MalformedManifest(path, reason)

A manifest does not follow the canonical encoding.
"""
struct MalformedManifest <: RunStoreError
    path::String
    reason::String
end

Base.showerror(io::IO, e::MalformedManifest) = print(
    io,
    "MalformedManifest: the manifest $(e.path) is malformed ($(e.reason))",
)

"""
    MalformedMarker(path, reason)

A `COMPLETE` marker does not follow the canonical encoding.
"""
struct MalformedMarker <: RunStoreError
    path::String
    reason::String
end

Base.showerror(io::IO, e::MalformedMarker) = print(
    io,
    "MalformedMarker: the COMPLETE marker $(e.path) is malformed ($(e.reason))",
)

"""
    OrphanFile(path, reason)

A canonical artifact exists without the generation it belongs to, or a
generation directory holds an unexpected file.
"""
struct OrphanFile <: RunStoreError
    path::String
    reason::String
end

Base.showerror(io::IO, e::OrphanFile) = print(
    io,
    "OrphanFile: $(e.path) is an orphan ($(e.reason))",
)

"""
    PendingGenerationConflict(run_id, generation)

A write targeted a generation that conflicts with the pending generation of
the store session.
"""
struct PendingGenerationConflict <: RunStoreError
    run_id::String
    generation::Int64
end

Base.showerror(io::IO, e::PendingGenerationConflict) = print(
    io,
    "PendingGenerationConflict: run $(repr(e.run_id)) already has a pending " *
    "generation; generation $(e.generation) cannot be started",
)

"""
    StoreClosed

The store was closed and cannot be used any more.
"""
struct StoreClosed <: RunStoreError end

Base.showerror(io::IO, e::StoreClosed) = print(
    io, "StoreClosed: the run store was closed and cannot be used any more",
)

"""
    ReadOnlyRunStore

A write was attempted on a store that was opened in read mode.
"""
struct ReadOnlyRunStore <: RunStoreError end

Base.showerror(io::IO, e::ReadOnlyRunStore) = print(
    io,
    "ReadOnlyRunStore: this run store was opened in read mode; open it with " *
    "mode = :write to record history",
)
