using SequentialABC
using Test
using Dates
using Serialization
import SHA

# ============================================================================
# Shared, test-only helpers for the native run-store tests.
#
# The helpers mutate a published run directory so that the store's integrity
# and durability behaviour can be exercised deterministically. They are test
# code: they never touch the production store.
# ============================================================================

# Every directory created here is removed when the test file that owns it
# finishes.
const NATIVE_TEST_DIRECTORIES = String[]

function native_test_directory()
    directory = mktempdir(; cleanup = false)
    push!(NATIVE_TEST_DIRECTORIES, directory)
    return directory
end

native_generation_dir(root::AbstractString, generation::Integer) =
    joinpath(root, "generations", "generation_" * lpad(string(generation), 6, '0'))

native_staging_dirs(root::AbstractString) =
    isdir(joinpath(root, "tmp")) ? sort(readdir(joinpath(root, "tmp"))) : String[]

# --- canonical text files ---------------------------------------------------

function read_canonical(path::AbstractString)
    entries = Dict{String,String}()
    for line in readlines(path)
        key, value = split(line, "="; limit = 2)
        entries[key] = value
    end
    return entries
end

function write_canonical(path::AbstractString, entries::Dict{String,String},
                         order::Vector{String})
    open(path, "w") do io
        for key in order
            haskey(entries, key) && println(io, key, "=", entries[key])
        end
    end
    return path
end

const NATIVE_MANIFEST_ORDER = [
    "schema", "version", "run_id", "generation", "metadata_sha256",
    "population_file", "population_sha256", "population_bytes",
    "population_count", "evaluation_file", "evaluation_sha256",
    "evaluation_bytes", "evaluation_count", "evaluation_key_set_sha256",
    "omission_count", "package_version", "julia_version", "completed_at",
    "expected_plan_sha256",
]

const NATIVE_MARKER_ORDER = [
    "schema", "version", "run_id", "generation", "manifest_file",
    "manifest_sha256", "completed_at",
]

"""
    refresh_generation_digests!(root, generation; updates = Dict{String,String}())

Rewrite the manifest of a published generation.

The partition digests and byte sizes are recomputed from the files on disk,
`updates` are applied to the manifest, and the manifest digest in the
`COMPLETE` marker is refreshed. After this call the byte-integrity chain of
the generation is internally consistent, so a read reaches the *semantic*
checks of the record instead of stopping at a digest mismatch.
"""
function refresh_generation_digests!(root::AbstractString, generation::Integer;
                                      updates = Dict{String,String}())
    directory = native_generation_dir(root, generation)
    manifest_path = joinpath(directory, "manifest")
    marker_path = joinpath(directory, "COMPLETE")
    entries = read_canonical(manifest_path)
    for (file_key, digest_key, size_key) in
        (("population_file", "population_sha256", "population_bytes"),
         ("evaluation_file", "evaluation_sha256", "evaluation_bytes"))
        path = joinpath(directory, entries[file_key])
        entries[digest_key] = SHA.bytes2hex(open(SHA.sha256, path))
        entries[size_key] = string(filesize(path))
    end
    # The updates are applied last, so a test can deliberately record a
    # digest, a size, or a count that does not match the file.
    for (key, value) in updates
        entries[key] = value
    end
    write_canonical(manifest_path, entries, NATIVE_MANIFEST_ORDER)
    marker = read_canonical(marker_path)
    marker["manifest_sha256"] = SHA.bytes2hex(open(SHA.sha256, manifest_path))
    write_canonical(marker_path, marker, NATIVE_MARKER_ORDER)
    return nothing
end

"""
    refresh_marker_digest!(root, generation)

Recompute the manifest digest that the `COMPLETE` marker records.
"""
function refresh_marker_digest!(root::AbstractString, generation::Integer)
    directory = native_generation_dir(root, generation)
    manifest_path = joinpath(directory, "manifest")
    marker_path = joinpath(directory, "COMPLETE")
    marker = read_canonical(marker_path)
    marker["manifest_sha256"] = SHA.bytes2hex(open(SHA.sha256, manifest_path))
    write_canonical(marker_path, marker, NATIVE_MARKER_ORDER)
    return nothing
end

"""
    patch_float!(path, from, to) -> Int

Replace every occurrence of a floating-point value in a file and report how
many occurrences were replaced.
"""
function patch_float!(path::AbstractString, from::Float64, to::Float64)
    data = read(path)
    pattern = collect(reinterpret(UInt8, [from]))
    replacement = collect(reinterpret(UInt8, [to]))
    patched = 0
    index = 1
    while index + length(pattern) - 1 <= length(data)
        if data[index:(index + length(pattern) - 1)] == pattern
            data[index:(index + length(pattern) - 1)] = replacement
            patched += 1
            index += length(pattern)
        else
            index += 1
        end
    end
    open(path, "w") do io
        write(io, data)
    end
    return patched
end

# --- crafted records --------------------------------------------------------

"""
    population_fields(; kwargs...)

A record-shaped value with the fields of a [`PopulationRecord`](@ref).

The store does not trust the stored representation: it re-derives every
record through the validating constructor. These helpers therefore build the
values that the store has to reject, without weakening an invariant.
"""
function population_fields(; generation::Integer=1, particles=[[1.0], [2.0]],
                            weights=[0.5, 0.5], distances=[0.1, 0.2],
                            epsilon=1.0, ess=2.0, attempts=nothing,
                            ancestry=nothing, resampling_count::Integer=0,
                            adaptive=nothing,
                            started_at=DateTime(2026, 1, 2, 3, 4, 5),
                            finished_at=DateTime(2026, 1, 2, 3, 4, 6),
                            provenance=RecordedProvenance())
    return (generation=Int64(generation), particles=particles, weights=weights,
            distances=distances, epsilon=epsilon, ess=ess, attempts=attempts,
            ancestry=ancestry, resampling_count=Int64(resampling_count),
            adaptive=adaptive, started_at=started_at, finished_at=finished_at,
            provenance=provenance)
end

"""
    evaluation_fields(; kwargs...)

A record-shaped value with the fields of an [`EvaluationRecord`](@ref).
"""
function evaluation_fields(; context=EvaluationContext(:smc, 1, 1, 1, 1, 1, UInt64(7)),
                           status::Symbol=:success,
                           summary=recorded_summary([1.0]), failure=nothing,
                           wall_time_s=0.1, provenance=RecordedProvenance(),
                           discrepancy=nothing)
    return (context=context, status=status, summary=summary, failure=failure,
            wall_time_s=wall_time_s, provenance=provenance,
            discrepancy=discrepancy)
end

"""
    craft_partition!(path, run_id, generation, kind, records)

Write a native partition that holds the given record values.

The header states how many records follow, exactly as the store writes it.
"""
function craft_partition!(path::AbstractString, run_id::AbstractString,
                          generation::Integer, kind::Symbol, records)
    header = SequentialABC.NativePartitionHeader(
        "sequentialabc.runstore", UInt64(1), kind, String(run_id),
        Int64(generation), length(records), DateTime(2026, 1, 2, 3, 4, 5))
    open(path, "w") do io
        serialize(io, header)
        for record in records
            serialize(io, record)
        end
    end
    return path
end
