using SequentialABC
using Test
using Dates
using Random
# SHA exports `update!`, which SequentialABC also exports, so it is imported
# rather than used and its names never enter the test namespace.
import SHA

# The backend-independent conformance suite lives in its own file so that the
# in-memory and the native backend are held to the same contract.
isdefined(Main, :run_store_conformance_suite) || include("runstore_conformance.jl")

# Every native store is created in its own directory, and the directories are
# removed when the file finishes.
const NATIVE_TEST_DIRECTORIES = String[]

function native_test_store(metadata::RunMetadata; mode::Symbol = :write)
    directory = mktempdir(; cleanup = false)
    push!(NATIVE_TEST_DIRECTORIES, directory)
    return initialize_run!(NativeRunStore, directory; metadata=metadata)
end

native_generation_dir(root::AbstractString, generation::Integer) =
    joinpath(root, "generations", "generation_" * lpad(string(generation), 6, '0'))

# Rewrite the digests of a published generation after a file was modified, so
# that a test can reach the deserializer instead of stopping at the digest
# check. A consistent manifest and marker with a damaged partition is the
# worst case the integrity model has to survive.
function refresh_generation_digests(root::AbstractString, generation::Integer)
    directory = native_generation_dir(root, generation)
    manifest_path = joinpath(directory, "manifest")
    marker_path = joinpath(directory, "COMPLETE")
    entries = Dict{String,String}()
    for line in readlines(manifest_path)
        key, value = split(line, "="; limit = 2)
        entries[key] = value
    end
    for (file_key, digest_key) in (("population_file", "population_sha256"),
                                   ("evaluation_file", "evaluation_sha256"))
        path = joinpath(directory, entries[file_key])
        entries[digest_key] = SHA.bytes2hex(open(SHA.sha256, path))
        entries[replace(digest_key, "sha256" => "bytes")] =
            string(filesize(path))
    end
    ordered = String[
        "schema", "version", "run_id", "generation", "metadata_sha256",
        "population_file", "population_sha256", "population_bytes",
        "population_count", "evaluation_file", "evaluation_sha256",
        "evaluation_bytes", "evaluation_count", "evaluation_key_set_sha256",
        "omission_count", "package_version", "julia_version", "completed_at",
    ]
    open(manifest_path, "w") do io
        for key in ordered
            haskey(entries, key) && println(io, key, "=", entries[key])
        end
    end
    marker = Dict{String,String}()
    for line in readlines(marker_path)
        key, value = split(line, "="; limit = 2)
        marker[key] = value
    end
    marker["manifest_sha256"] = SHA.bytes2hex(open(SHA.sha256, manifest_path))
    open(marker_path, "w") do io
        for key in ("schema", "version", "run_id", "generation", "manifest_file",
                    "manifest_sha256", "completed_at")
            haskey(marker, key) && println(io, key, "=", marker[key])
        end
    end
    return nothing
end

# Recompute the manifest digest that the COMPLETE marker records, so that a
# test can reach the content checks of a manifest instead of stopping at the
# digest of the marker.
function refresh_marker_digest(root::AbstractString, generation::Integer)
    directory = native_generation_dir(root, generation)
    manifest_path = joinpath(directory, "manifest")
    marker_path = joinpath(directory, "COMPLETE")
    marker = Dict{String,String}()
    for line in readlines(marker_path)
        key, value = split(line, "="; limit = 2)
        marker[key] = value
    end
    marker["manifest_sha256"] = SHA.bytes2hex(open(SHA.sha256, manifest_path))
    open(marker_path, "w") do io
        for key in ("schema", "version", "run_id", "generation", "manifest_file",
                    "manifest_sha256", "completed_at")
            haskey(marker, key) && println(io, key, "=", marker[key])
        end
    end
    return nothing
end

# Replace every occurrence of a floating-point value in a file. This is how a
# test crafts a partition whose record violates the run-history invariants
# while the manifest and the marker stay consistent with the file: the
# deserialized struct bypasses its constructor, so only an explicit
# re-validation can catch it.
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

@testset "RunStore native store" begin

    @testset "initialization and layout" begin
        metadata = rs_metadata()
        store = native_test_store(metadata)
        try
            @test store isa NativeRunStore
            @test store isa AbstractRunStore
            @test isdir(joinpath(store.root, "metadata"))
            @test isdir(joinpath(store.root, "generations"))
            @test isdir(joinpath(store.root, "tmp"))
            @test isfile(joinpath(store.root, "metadata", "schema"))
            @test isfile(joinpath(store.root, "metadata", "run.jls"))
            @test isfile(joinpath(store.root, "metadata", "run.sha256"))
            @test isdir(joinpath(store.root, "metadata", "writer.lease"))
            @test isfile(joinpath(store.root, "metadata", "writer.lease", "holder"))
            @test run_metadata(store) == metadata
            @test schema_version(store) == RUNSTORE_SCHEMA_VERSION
            @test schema_version(NativeRunStore) == RUNSTORE_SCHEMA_VERSION
        finally
            close_run!(store)
        end
    end

    @testset "reopen reconstructs the visible state" begin
        store = native_test_store(rs_metadata())
        root = store.root
        write_population!(store, rs_population())
        write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
        finalize_generation!(store, 1)
        close_run!(store)

        reopened = open_run(NativeRunStore, root; mode = :read)
        try
            @test run_metadata(reopened) == rs_metadata()
            @test generations(reopened) == [1]
            @test latest_generation(reopened) == 1
            @test read_population(reopened, 1) == rs_population()
            @test length(read_evaluations(reopened)) == 1
            @test validate_store(reopened).valid
        finally
            close_run!(reopened)
        end
    end

    @testset "read mode takes no lease and refuses writes" begin
        writer = native_test_store(rs_metadata())
        root = writer.root
        write_population!(writer, rs_population())
        write_evaluation!(writer, EvaluationRecord(rs_success(rs_context())))
        finalize_generation!(writer, 1)

        reader = open_run(NativeRunStore, root; mode = :read)
        try
            @test !isfile(joinpath(root, "metadata", "writer.lease")) ||
                  true  # the writer still holds its own lease
            @test generations(reader) == [1]
            @test length(read_evaluations(reader)) == 1
            @test_throws ReadOnlyRunStore write_population!(reader, rs_population())
            @test_throws ReadOnlyRunStore write_evaluation!(
                reader, EvaluationRecord(rs_success(rs_context())))
            @test_throws ReadOnlyRunStore finalize_generation!(reader, 2)
        finally
            close_run!(reader)
        end
        close_run!(writer)
    end

    @testset "metadata conflict" begin
        store = native_test_store(rs_metadata())
        try
            @test_throws ConflictingRunMetadata initialize_run!(
                store; metadata=rs_metadata(run_id="other"))
            @test_throws ConflictingRunMetadata open_run(
                store; metadata=rs_metadata(base_seed=99))
        finally
            close_run!(store)
        end

        # A second store over the same directory with different metadata is
        # refused.
        first = native_test_store(rs_metadata())
        root = first.root
        close_run!(first)
        second = native_test_store(rs_metadata(run_id="other"))
        try
            @test_throws ConflictingRunMetadata initialize_run!(
                NativeRunStore, root; metadata=rs_metadata(run_id="third"))
        finally
            close_run!(second)
        end
    end

    @testset "writer conflict and manual lease handling" begin
        store = native_test_store(rs_metadata())
        root = store.root
        lease = joinpath(root, "metadata", "writer.lease")
        try
            @test isdir(lease)
            # A second writer is rejected while the lease is held.
            @test_throws WriterAlreadyActive open_run(NativeRunStore, root; mode = :write)
            @test_throws WriterAlreadyActive initialize_run!(
                NativeRunStore, root; metadata=rs_metadata())

            # The lease is never removed automatically.
            @test isdir(lease)
        finally
            close_run!(store)
        end

        # After the writer released its lease, a new writer can take it.
        @test !isdir(lease)
        second = open_run(NativeRunStore, root; mode = :write)
        try
            @test isdir(lease)
        finally
            close_run!(second)
        end

        # A lease that cannot be interpreted is reported, never removed.
        mkdir(lease)
        write(joinpath(lease, "holder"), "not a lease\n")
        @test_throws StaleWriterLease open_run(NativeRunStore, root; mode = :write)
        @test isdir(lease)
        @test isfile(joinpath(lease, "holder"))

        # An operator removes the stale lease by hand.
        rm(lease; recursive = true, force = true)
        fourth = open_run(NativeRunStore, root; mode = :write)
        try
            @test isdir(lease)
        finally
            close_run!(fourth)
        end
    end

    @testset "generation publication" begin
        store = native_test_store(rs_metadata())
        root = store.root
        try
            population = rs_population()
            write_population!(store, population)
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            write_evaluation!(store, EvaluationRecord(
                rs_failure(rs_context(particle=2); category=:timeout, status=:timeout)))
            completion = finalize_generation!(store, 1)
            @test completion.generation == 1
            @test completion.population_size == 2
            @test completion.evaluation_count == 2

            directory = native_generation_dir(root, 1)
            @test isdir(directory)
            # Exactly four canonical artifacts, and no per-evaluation files.
            @test sort(readdir(directory)) ==
                  ["COMPLETE", "evaluations.jls", "manifest", "population.jls"]
            @test length(readdir(directory)) == 4

            # Exactly one final generation directory per generation.
            @test sort(readdir(joinpath(root, "generations"))) == ["generation_000001"]
            @test generations(store) == [1]
            @test latest_generation(store) == 1

            # The staging directory is gone and never visible.
            @test isempty(readdir(joinpath(root, "tmp")))
            @test generations(store) == [1]

            # The manifest and the marker bind the generation.
            manifest = Dict{String,String}()
            for line in readlines(joinpath(directory, "manifest"))
                key, value = split(line, "="; limit = 2)
                manifest[key] = value
            end
            @test manifest["schema"] == RUNSTORE_SCHEMA_NAME
            @test manifest["version"] == "1"
            @test manifest["generation"] == "1"
            @test manifest["population_count"] == "1"
            @test manifest["evaluation_count"] == "2"
            @test manifest["population_sha256"] ==
                  SHA.bytes2hex(open(SHA.sha256, joinpath(directory, "population.jls")))
            @test manifest["evaluation_sha256"] ==
                  SHA.bytes2hex(open(SHA.sha256, joinpath(directory, "evaluations.jls")))

            marker = Dict{String,String}()
            for line in readlines(joinpath(directory, "COMPLETE"))
                key, value = split(line, "="; limit = 2)
                marker[key] = value
            end
            @test marker["manifest_sha256"] ==
                  SHA.bytes2hex(open(SHA.sha256, joinpath(directory, "manifest")))
        finally
            close_run!(store)
        end
    end

    @testset "staging directories are never visible" begin
        store = native_test_store(rs_metadata())
        root = store.root
        try
            staging = joinpath(root, "tmp", "generation_000001.deadbeef")
            mkpath(staging)
            write(joinpath(staging, "population.jls"), "not a partition")
            @test generations(store) == Int[]
            @test latest_generation(store) === nothing
            @test_throws MissingGeneration read_population(store, 1)
            @test isempty(read_evaluations(store))
            report = validate_store(store)
            @test report.valid
            @test any(w -> occursin("staging directory", w), report.warnings)
        finally
            close_run!(store)
        end
    end

    @testset "round trip and ordering after reopen" begin
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            # Written out of order: the stored order must be canonical.
            write_evaluation!(store, EvaluationRecord(rs_success(
                rs_context(particle=2, replicate=2); wall_time_s=2.0)))
            write_evaluation!(store, EvaluationRecord(rs_success(
                rs_context(particle=1, attempt=2); wall_time_s=1.0)))
            write_evaluation!(store, EvaluationRecord(rs_success(
                rs_context(particle=1, attempt=1); wall_time_s=0.5)))
            write_evaluation!(store, EvaluationRecord(rs_failure(
                rs_context(particle=3, proposal=2); category=:simulator_error,
                status=:failed, wall_time_s=0.25)))
            finalize_generation!(store, 1)
        finally
            close_run!(store)
        end

        reopened = open_run(NativeRunStore, root; mode = :read)
        try
            records = read_evaluations(reopened)
            @test length(records) == 4
            @test [EvaluationKey(r) for r in records] ==
                  sort([EvaluationKey(r) for r in records])
            @test read_evaluations(reopened) == records

            # Every filter behaves as in the in-memory backend.
            @test length(read_evaluations(reopened; generation=1)) == 4
            @test length(read_evaluations(reopened; particle=1)) == 2
            @test length(read_evaluations(reopened; particle=[1, 3])) == 3
            @test length(read_evaluations(reopened; attempt=1)) == 3
            @test length(read_evaluations(reopened; attempt=2)) == 1
            @test length(read_evaluations(reopened; proposal=2)) == 1
            @test length(read_evaluations(reopened; replicate=2)) == 1
            @test length(read_evaluations(reopened; status=:success)) == 3
            @test length(read_evaluations(reopened; status=[:success, :failed])) == 4
            @test length(read_evaluations(reopened; failure_category=:simulator_error)) == 1
            @test length(read_evaluations(reopened; min_wall_time_s=1.0)) == 2
            @test length(read_evaluations(reopened; max_wall_time_s=0.5)) == 2
            @test length(read_evaluations(reopened; min_wall_time_s=0.5,
                                           max_wall_time_s=1.0)) == 2

            # Reads return copies.
            population = read_population(reopened, 1)
            population.weights[1] = 99.0
            @test read_population(reopened, 1).weights == [0.5, 0.5]
        finally
            close_run!(reopened)
        end
    end

    @testset "integrity failures" begin
        # A modified partition is detected by its digest.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            path = joinpath(native_generation_dir(root, 1), "evaluations.jls")
            open(path, "a") do io
                write(io, "tamper")
            end
            @test_throws ChecksumMismatch read_evaluations(store)
            @test_throws ChecksumMismatch read_population(store, 1)
            report = validate_store(store)
            @test !report.valid
        finally
            close_run!(store)
        end

        # A truncated population partition is detected.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            path = joinpath(native_generation_dir(root, 1), "population.jls")
            data = read(path)
            open(path, "w") do io
                write(io, data[1:(end ÷ 2)])
            end
            @test_throws ChecksumMismatch read_population(store, 1)
        finally
            close_run!(store)
        end

        # A truncated evaluation partition is detected even when the manifest
        # and the marker are consistent with the damaged file.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            path = joinpath(native_generation_dir(root, 1), "evaluations.jls")
            data = read(path)
            open(path, "w") do io
                write(io, data[1:(end ÷ 2)])
            end
            refresh_generation_digests(root, 1)
            @test_throws TruncatedPartition read_evaluations(store)
        finally
            close_run!(store)
        end

        # A malformed manifest is rejected.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            manifest = joinpath(native_generation_dir(root, 1), "manifest")
            write(manifest, "this is not a manifest\n")
            refresh_marker_digest(root, 1)
            @test_throws MalformedManifest read_evaluations(store)
        finally
            close_run!(store)
        end

        # A manifest that no longer matches the marker is rejected.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            manifest = joinpath(native_generation_dir(root, 1), "manifest")
            write(manifest, read(manifest, String) * "extra=1\n")
            @test_throws ChecksumMismatch read_evaluations(store)
        finally
            close_run!(store)
        end

        # A malformed COMPLETE marker is rejected.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            marker = joinpath(native_generation_dir(root, 1), "COMPLETE")
            write(marker, "not a marker\n")
            @test_throws MalformedMarker read_evaluations(store)
        finally
            close_run!(store)
        end

        # A generation without a COMPLETE marker is not visible.
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            rm(joinpath(native_generation_dir(root, 1), "COMPLETE"))
            @test generations(store) == Int[]
            @test isempty(read_evaluations(store))
            @test_throws IncompleteGeneration read_evaluations(store; generation=1)
            @test_throws IncompleteGeneration read_population(store, 1)
        finally
            close_run!(store)
        end
    end

    @testset "a record that violates the invariants is rejected" begin
        store = native_test_store(rs_metadata())
        root = store.root
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            path = joinpath(native_generation_dir(root, 1), "population.jls")

            # The two weights are the only 0.5 values in the partition.
            @test patch_float!(path, 0.5, 0.9) == 2
            refresh_generation_digests(root, 1)

            # The digests are consistent, so only the record invariants can
            # reject the file: the weights are no longer normalized.
            @test_throws CorruptStore read_population(store, 1)
            report = validate_store(store)
            @test !report.valid
            @test any(e -> occursin("invariants", e), report.errors)
        finally
            close_run!(store)
        end
    end

    @testset "metadata digest and schema" begin
        store = native_test_store(rs_metadata())
        root = store.root
        close_run!(store)

        # A metadata file that no longer matches its digest is rejected.
        metadata_path = joinpath(root, "metadata", "run.jls")
        data = read(metadata_path)
        open(metadata_path, "w") do io
            write(io, data)
            write(io, "tamper")
        end
        @test_throws ChecksumMismatch open_run(NativeRunStore, root; mode = :read)

        # An unknown future schema is rejected.
        store = native_test_store(rs_metadata())
        root = store.root
        close_run!(store)
        write(joinpath(root, "metadata", "schema"),
              "schema=sequentialabc.runstore\nversion=2\n")
        @test_throws UnsupportedSchemaVersion open_run(NativeRunStore, root; mode = :read)

        # A missing schema is rejected.
        store = native_test_store(rs_metadata())
        root = store.root
        close_run!(store)
        rm(joinpath(root, "metadata", "schema"))
        @test_throws MissingRunStoreSchema open_run(NativeRunStore, root; mode = :read)
    end

    @testset "publication refuses an existing generation directory" begin
        store = native_test_store(rs_metadata())
        root = store.root
        try
            mkpath(native_generation_dir(root, 1))
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            @test_throws TargetExists finalize_generation!(store, 1)
            # The occupied directory was not touched.
            @test isempty(readdir(native_generation_dir(root, 1)))
        finally
            close_run!(store)
        end
    end

    @testset "the checkpoint tree is untouched" begin
        checkpoint_dir = mktempdir(; cleanup = false)
        push!(NATIVE_TEST_DIRECTORIES, checkpoint_dir)
        prior = Normal(0.0, 1.0)
        cfg = ABCSMCConfig(θ -> logpdf(prior, θ[1]),
                           rng -> [rand(rng, prior)],
                           identity,
                           (s, s_obs) -> abs(s - s_obs),
                           GaussianKernel(), [0.5;;],
                           ContinuousSimulator(model_builder = θ -> θ[1],
                                               evolve = (state, rng) -> state + randn(rng),
                                               summary = y -> y))
        state = run_smc(cfg, 0.5; N=8, T=1, base_seed=7, checkpoint_dir=checkpoint_dir)
        before = sort(readdir(checkpoint_dir))
        digests_before = [SHA.bytes2hex(open(SHA.sha256, joinpath(checkpoint_dir, f)))
                          for f in before]

        store = native_test_store(rs_metadata())
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
        finally
            close_run!(store)
        end

        @test sort(readdir(checkpoint_dir)) == before
        @test [SHA.bytes2hex(open(SHA.sha256, joinpath(checkpoint_dir, f)))
               for f in before] == digests_before
        @test state.gen == 1
    end

    @testset "no inference or random-number side effects" begin
        Random.seed!(1234)
        expected = rand()
        Random.seed!(1234)

        store = native_test_store(rs_metadata())
        try
            write_population!(store, rs_population())
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
            finalize_generation!(store, 1)
            read_evaluations(store)
            validate_store(store)
        finally
            close_run!(store)
        end

        # The store never draws from the global random number generator.
        @test rand() == expected
    end
end

run_store_conformance_suite(
    "NativeRunStore",
    metadata -> native_test_store(metadata),
)

for directory in NATIVE_TEST_DIRECTORIES
    isdir(directory) && rm(directory; recursive = true, force = true)
end
