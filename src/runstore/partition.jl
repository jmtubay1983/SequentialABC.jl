# ============================================================================
# Native record partitions
# ============================================================================
#
# A generation partition is a Julia `Serialization` stream:
#
#   serialize(header)
#   serialize(record)
#   serialize(record)
#   ...
#
# The header states how many records follow, so a reader can detect a
# truncated partition instead of silently returning fewer records. The
# records are written in a canonical order (the evaluation key order), so the
# bytes of a partition are deterministic for a given set of records.
#
# The `.jls` format is native to Julia and to the types of this package. It
# is trusted-input only: deserialization reconstructs arbitrary types and can
# execute code. It is not a portable export format.

"""
    NativePartitionHeader

Header of a native record partition.

Fields:
  - `schema_name::String`     — storage schema name
  - `schema_version::UInt64`  — storage schema version
  - `kind::Symbol`            — `:population` or `:evaluations`
  - `run_id::String`          — identity of the run the partition belongs to
  - `generation::Int64`       — generation the partition belongs to
  - `record_count::Int64`     — number of records that follow the header
  - `created_at::DateTime`    — creation timestamp (UTC semantics)

The definition of this type is part of storage schema v1: a reader of schema
v1 expects exactly these fields.
"""
struct NativePartitionHeader
    schema_name::String
    schema_version::UInt64
    kind::Symbol
    run_id::String
    generation::Int64
    record_count::Int64
    created_at::DateTime
end

"""
    _write_partition(path, header, records)

Stream a partition header and its records to `path`.

The records are written one at a time, so the memory cost does not depend on
the size of the partition.
"""
function _write_partition(path::AbstractString,
                          header::NativePartitionHeader,
                          records)
    open(path, "w") do io
        serialize(io, header)
        for record in records
            serialize(io, record)
        end
    end
    return path
end

"""
    _read_partition(path, run_id, generation, kind) -> (header, records)

Read a partition and return its header and its records.

Every failure is reported with the run, the generation, and the path: a
truncated partition raises [`TruncatedPartition`](@ref) and any other
deserialization failure raises [`CorruptStore`](@ref). The raw
`Serialization` exception is never surfaced.
"""
function _read_partition(path::AbstractString,
                         run_id::AbstractString,
                         generation::Int64,
                         kind::Symbol)
    open(path, "r") do io
        header = try
            deserialize(io)
        catch err
            throw(CorruptStore(
                run_id, generation, path,
                "the partition header could not be deserialized ($(typeof(err)))"))
        end
        header isa NativePartitionHeader || throw(CorruptStore(
            run_id, generation, path,
            "the partition header has the unexpected type $(typeof(header))"))
        header.schema_name == RUNSTORE_SCHEMA_NAME || throw(UnsupportedSchemaVersion(
            header.schema_name, header.schema_version, RUNSTORE_SCHEMA_NAME,
            [RUNSTORE_SCHEMA_VERSION]))
        header.schema_version == RUNSTORE_SCHEMA_VERSION || throw(UnsupportedSchemaVersion(
            header.schema_name, header.schema_version, RUNSTORE_SCHEMA_NAME,
            [RUNSTORE_SCHEMA_VERSION]))
        header.kind === kind || throw(CorruptStore(
            run_id, generation, path,
            "the partition holds $(header.kind) records but $(kind) was expected"))
        header.run_id == run_id || throw(CorruptStore(
            run_id, generation, path,
            "the partition belongs to run $(repr(header.run_id))"))
        header.generation == generation || throw(CorruptStore(
            run_id, generation, path,
            "the partition belongs to generation $(header.generation)"))

        records = Any[]
        for index in 1:header.record_count
            record = try
                deserialize(io)
            catch err
                if err isa EOFError
                    throw(TruncatedPartition(
                        run_id, generation, path, header.record_count, index - 1))
                end
                throw(CorruptStore(
                    run_id, generation, path,
                    "record $(index) could not be deserialized ($(typeof(err)))"))
            end
            push!(records, record)
        end
        return header, records
    end
end

"""
    _each_partition_record(f, path, run_id, generation, kind)

Stream the records of a partition to `f`, one record at a time.

The partition is never materialized as a whole, so a query over a large
generation only holds the matching records in memory.
"""
function _each_partition_record(f::Function,
                                path::AbstractString,
                                run_id::AbstractString,
                                generation::Int64,
                                kind::Symbol)
    open(path, "r") do io
        header = try
            deserialize(io)
        catch err
            throw(CorruptStore(
                run_id, generation, path,
                "the partition header could not be deserialized ($(typeof(err)))"))
        end
        header isa NativePartitionHeader || throw(CorruptStore(
            run_id, generation, path,
            "the partition header has the unexpected type $(typeof(header))"))
        header.schema_name == RUNSTORE_SCHEMA_NAME || throw(UnsupportedSchemaVersion(
            header.schema_name, header.schema_version, RUNSTORE_SCHEMA_NAME,
            [RUNSTORE_SCHEMA_VERSION]))
        header.schema_version == RUNSTORE_SCHEMA_VERSION || throw(UnsupportedSchemaVersion(
            header.schema_name, header.schema_version, RUNSTORE_SCHEMA_NAME,
            [RUNSTORE_SCHEMA_VERSION]))
        header.kind === kind || throw(CorruptStore(
            run_id, generation, path,
            "the partition holds $(header.kind) records but $(kind) was expected"))
        header.run_id == run_id || throw(CorruptStore(
            run_id, generation, path,
            "the partition belongs to run $(repr(header.run_id))"))
        header.generation == generation || throw(CorruptStore(
            run_id, generation, path,
            "the partition belongs to generation $(header.generation)"))

        for index in 1:header.record_count
            record = try
                deserialize(io)
            catch err
                if err isa EOFError
                    throw(TruncatedPartition(
                        run_id, generation, path, header.record_count, index - 1))
                end
                throw(CorruptStore(
                    run_id, generation, path,
                    "record $(index) could not be deserialized ($(typeof(err)))"))
            end
            f(record)
        end
    end
    return nothing
end
