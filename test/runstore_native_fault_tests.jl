using SequentialABC
using Test
using Dates
using Logging
import SHA

# The shared conformance suite and the shared, test-only mutation helpers live
# in their own files.
isdefined(Main, :run_store_conformance_suite) || include("runstore_conformance.jl")
isdefined(Main, :native_test_directory) || include("runstore_native_helpers.jl")

# ============================================================================
# Durability and fault-injection matrix for the native run store.
#
# The suite covers four distinct situations:
#
#   A. an interruption before the final directory rename leaves only
#      invisible staging evidence;
#   B. an interruption after the rename leaves a visible, complete generation
#      whose acknowledgement was interrupted;
#   C. ordinary corruption is caught by the byte-integrity chain;
#   D. checksum-consistent but semantically invalid records are caught by the
#      constructor revalidation.
#
# The fault injector is internal (`SequentialABC._native_runstore` /
# `SequentialABC._InjectedInterruption`); it is not part of the public API.
# ============================================================================

const PRE_RENAME_PHASES = (
    :before_staging,
    :after_staging_created,
    :after_population_write,
    :after_evaluation_write,
    :after_partition_reopen_validation,
    :after_digest_computation,
    :after_manifest_write,
    :after_complete_write,
    :after_staging_validation,
    :before_publish_rename,
)

const POST_RENAME_PHASES = (:after_publish_rename, :after_final_validation)

# A store with `generations` committed generations, plus its directory.
function durability_store(; generations::Int = 1, metadata = rs_metadata())
    directory = native_test_directory()
    store = initialize_run!(NativeRunStore, directory; metadata=metadata)
    for generation in 1:generations
        write_population!(store, rs_population(generation=generation))
        write_evaluation!(store, EvaluationRecord(
            rs_success(rs_context(generation=generation))))
        finalize_generation!(store, generation)
    end
    return store, directory
end

# Arm the internal injector so that it throws at `target` and records every
# phase it observes.
function arm_fault!(store::NativeRunStore, target::Symbol, observed::Vector{Symbol})
    store.fault = function (phase::Symbol)
        push!(observed, phase)
        phase === target && throw(SequentialABC._InjectedInterruption(phase))
        return nothing
    end
    return store
end

# The bytes of every committed generation: path, size, and digest.
function generation_snapshot(root::AbstractString)
    snapshot = Dict{String,Any}()
    for entry in sort(readdir(joinpath(root, "generations")))
        directory = joinpath(root, "generations", entry)
        isdir(directory) || continue
        for file in sort(readdir(directory))
            path = joinpath(directory, file)
            snapshot["$entry/$file"] = (
                filesize(path),
                SHA.bytes2hex(open(SHA.sha256, path)),
            )
        end
    end
    return snapshot
end

# Close a store whose pending generation was deliberately left behind. The
# discard is reported by the store; the tests that care about the report
# assert it explicitly, so the noise is suppressed here.
function close_quietly!(store::NativeRunStore)
    Logging.with_logger(Logging.NullLogger()) do
        close_run!(store)
    end
    return nothing
end

@testset "RunStore native durability" begin

    @testset "publication phases match the production flow" begin
        phases = SequentialABC.NATIVE_PUBLICATION_PHASES
        @test phases == (
            :before_staging,
            :after_staging_created,
            :after_population_write,
            :after_evaluation_write,
            :after_partition_reopen_validation,
            :after_digest_computation,
            :after_manifest_write,
            :after_complete_write,
            :after_staging_validation,
            :before_publish_rename,
            :after_publish_rename,
            :after_final_validation,
        )

        # The injector observes exactly the declared phases, in order.
        observed = Symbol[]
        store, directory = durability_store()
        try
            arm_fault!(store, :after_final_validation, observed)
            write_population!(store, rs_population(generation=2))
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context(generation=2))))
            @test_throws SequentialABC._InjectedInterruption finalize_generation!(store, 2)
            @test observed == collect(phases)
        finally
            close_run!(store)
        end
    end

    @testset "interruption before the publication rename" begin
        for phase in PRE_RENAME_PHASES
            @testset "$(phase)" begin
                store, directory = durability_store(generations=1)
                try
                    before = generation_snapshot(directory)
                    observed = Symbol[]
                    arm_fault!(store, phase, observed)

                    write_population!(store, rs_population(generation=2))
                    write_evaluation!(store, EvaluationRecord(
                        rs_success(rs_context(generation=2))))
                    @test_throws SequentialABC._InjectedInterruption finalize_generation!(store, 2)

                    # The attempted generation is not visible.
                    @test generations(store) == [1]
                    @test latest_generation(store) == 1
                    @test_throws MissingGeneration read_evaluations(store; generation=2)
                    @test_throws MissingGeneration read_population(store, 2)

                    # The staging evidence remains; nothing was deleted. The
                    # staging directory exists from `:after_staging_created`
                    # onwards.
                    staging = native_staging_dirs(directory)
                    if phase === :before_staging
                        @test isempty(staging)
                    else
                        @test length(staging) == 1
                        @test startswith(staging[1], "generation_000002.")
                    end

                    # The previous generation is byte-identical and readable.
                    @test generation_snapshot(directory)["generation_000001/population.jls"] ==
                          before["generation_000001/population.jls"]
                    @test generation_snapshot(directory)["generation_000001/evaluations.jls"] ==
                          before["generation_000001/evaluations.jls"]
                    @test read_population(store, 1) == rs_population()
                    @test length(read_evaluations(store)) == 1

                    # The store reports the leftover staging as a warning, and
                    # the store itself stays valid. The pending generation is
                    # still in memory, so it is reported as well.
                    report = validate_store(store)
                    @test report.valid
                    @test any(w -> occursin("pending", w), report.warnings)
                    if phase !== :before_staging
                        @test any(w -> occursin("staging directory", w), report.warnings)
                    end

                    # The session still holds its lease, and a second writer is
                    # rejected.
                    @test isdir(joinpath(directory, "metadata", "writer.lease"))
                    @test_throws WriterAlreadyActive open_run(
                        NativeRunStore, directory; mode=:write)

                    # A reader can still inspect the committed generations.
                    reader = open_run(NativeRunStore, directory; mode=:read)
                    try
                        @test generations(reader) == [1]
                        @test length(read_evaluations(reader)) == 1
                    finally
                        close_run!(reader)
                    end
                finally
                    close_quietly!(store)
                end
            end
        end
    end

    @testset "interruption after the publication rename" begin
        for phase in POST_RENAME_PHASES
            @testset "$(phase)" begin
                store, directory = durability_store(generations=1)
                try
                    observed = Symbol[]
                    arm_fault!(store, phase, observed)

                    write_population!(store, rs_population(generation=2))
                    write_evaluation!(store, EvaluationRecord(
                        rs_success(rs_context(generation=2))))
                    @test_throws SequentialABC._InjectedInterruption finalize_generation!(store, 2)

                    # The publication completed: the generation is canonical,
                    # visible, and complete.
                    @test generations(store) == [1, 2]
                    @test latest_generation(store) == 2
                    @test isempty(native_staging_dirs(directory))
                    @test isdir(native_generation_dir(directory, 2))

                    # The pending state is stale and was dropped: the writer
                    # cannot publish the same generation again.
                    @test store.pending_generation === nothing
                    @test store.pending_population === nothing
                    @test isempty(store.pending_evaluations)
                    @test_throws GenerationAlreadyComplete finalize_generation!(store, 2)

                    # A fresh reader validates and reads the generation.
                    reader = open_run(NativeRunStore, directory; mode=:read)
                    try
                        @test validate_store(reader).valid
                        @test read_population(reader, 2) == rs_population(generation=2)
                        @test length(read_evaluations(reader; generation=2)) == 1
                    finally
                        close_run!(reader)
                    end
                finally
                    close_run!(store)
                end

                # Closing the writer does not remove the published generation.
                @test isdir(native_generation_dir(directory, 2))
                @test isdir(native_generation_dir(directory, 1))
            end
        end
    end

    @testset "corrupted bytes" begin
        # Every case damages a committed generation and must fail closed with
        # a contextual error, while the previous generation stays readable.
        # `reads` names the reads that must fail; the others must still work,
        # because the damage is confined to one partition.
        function corrupt_case(mutate!, expected; reads = (:population, :evaluations))
            store, directory = durability_store(generations=2)
            try
                mutate!(store, directory)
                if :population in reads
                    @test_throws expected read_population(store, 2)
                else
                    @test read_population(store, 2) == rs_population(generation=2)
                end
                if :evaluations in reads
                    @test_throws expected read_evaluations(store; generation=2)
                else
                    @test length(read_evaluations(store; generation=2)) == 1
                end
                # The previous generation is untouched and readable.
                @test read_population(store, 1) == rs_population()
                @test length(read_evaluations(store; generation=1)) == 1
            finally
                close_run!(store)
            end
        end

        @testset "population bit flip" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                data = read(path)
                data[10] = data[10] ⊻ 0x01
                open(path, "w") do io
                    write(io, data)
                end
            end, ChecksumMismatch; reads = (:population, :evaluations))
        end

        @testset "evaluation bit flip" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                data = read(path)
                data[10] = data[10] ⊻ 0x01
                open(path, "w") do io
                    write(io, data)
                end
            end, ChecksumMismatch; reads = (:population, :evaluations))
        end

        @testset "population truncation" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                data = read(path)
                open(path, "w") do io
                    write(io, data[1:(end ÷ 2)])
                end
            end, ChecksumMismatch; reads = (:population, :evaluations))
        end

        @testset "evaluation truncation" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                data = read(path)
                open(path, "w") do io
                    write(io, data[1:(end ÷ 2)])
                end
            end, ChecksumMismatch; reads = (:population, :evaluations))
        end

        @testset "truncated partition with a consistent manifest" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                data = read(path)
                open(path, "w") do io
                    write(io, data[1:(end ÷ 2)])
                end
                refresh_generation_digests!(directory, 2)
            end, TruncatedPartition; reads = (:evaluations,))
        end

        @testset "corrupt serialized header" begin
            # The header is cut short, so the deserializer runs into the end
            # of the file instead of reconstructing a type. (Corrupting the
            # *type* bytes of a serialized header is not recoverable inside
            # the process: Julia's `Serialization` is not safe against a
            # corrupted type header, which is why the native format is
            # documented as trusted-input only and why the digests are
            # verified before anything is deserialized.)
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                data = read(path)
                open(path, "w") do io
                    write(io, data[1:6])
                end
                refresh_generation_digests!(directory, 2)
            end, CorruptStore; reads = (:population,))
        end

        @testset "wrong partition kind" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 2, :evaluations,
                                 [population_fields()])
                refresh_generation_digests!(directory, 2)
            end, CorruptStore; reads = (:population,))
        end

        @testset "wrong generation in the partition header" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 99, :population,
                                 [population_fields(generation=99)])
                refresh_generation_digests!(directory, 2)
            end, CorruptStore; reads = (:population,))
        end

        @testset "manifest hash mismatch" begin
            corrupt_case((store, directory) -> begin
                refresh_generation_digests!(directory, 2;
                    updates = Dict("population_sha256" => repeat("0", 64)))
            end, ChecksumMismatch; reads = (:population, :evaluations))
        end

        @testset "manifest byte-size mismatch" begin
            corrupt_case((store, directory) -> begin
                refresh_generation_digests!(directory, 2;
                    updates = Dict("population_bytes" => "1"))
            end, CorruptStore; reads = (:population, :evaluations))
        end

        @testset "manifest record-count mismatch" begin
            corrupt_case((store, directory) -> begin
                refresh_generation_digests!(directory, 2;
                    updates = Dict("evaluation_count" => "7"))
            end, CorruptStore; reads = (:evaluations,))
        end

        @testset "evaluation-key-set hash mismatch" begin
            corrupt_case((store, directory) -> begin
                refresh_generation_digests!(directory, 2;
                    updates = Dict("evaluation_key_set_sha256" => repeat("1", 64)))
            end, ChecksumMismatch; reads = (:evaluations,))
        end

        @testset "metadata hash mismatch" begin
            store, directory = durability_store(generations=1)
            try
                path = joinpath(directory, "metadata", "run.jls")
                open(path, "a") do io
                    write(io, "tamper")
                end
                @test_throws ChecksumMismatch open_run(
                    NativeRunStore, directory; mode=:read)
            finally
                close_run!(store)
            end
        end

        @testset "malformed COMPLETE marker" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "COMPLETE")
                write(path, "not a marker\n")
            end, MalformedMarker; reads = (:population, :evaluations))
        end

        @testset "COMPLETE refers to the wrong manifest" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "COMPLETE")
                entries = read_canonical(path)
                entries["manifest_file"] = "other.manifest"
                write_canonical(path, entries, NATIVE_MARKER_ORDER)
            end, MissingManifest; reads = (:population, :evaluations))
        end

        @testset "unsupported storage schema" begin
            store, directory = durability_store(generations=1)
            try
                write(joinpath(directory, "metadata", "schema"),
                      "schema=sequentialabc.runstore\nversion=99\n")
                @test_throws UnsupportedSchemaVersion open_run(
                    NativeRunStore, directory; mode=:read)
            finally
                close_run!(store)
            end
        end

        @testset "conflicting run identity" begin
            corrupt_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "manifest")
                entries = read_canonical(path)
                entries["run_id"] = "another-run"
                write_canonical(path, entries, NATIVE_MANIFEST_ORDER)
                refresh_marker_digest!(directory, 2)
            end, CorruptStore; reads = (:population, :evaluations))
        end
    end

    @testset "checksum-consistent semantic corruption" begin
        # The bytes are rewritten and every digest is recomputed, so the
        # byte-integrity chain passes and only the constructor revalidation can
        # reject the file.
        function semantic_case(mutate!; reads = (:population,))
            store, directory = durability_store(generations=2)
            try
                mutate!(store, directory)
                # The digests are consistent: the byte chain passes.
                manifest = read_canonical(joinpath(
                    native_generation_dir(directory, 2), "manifest"))
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                @test manifest["population_sha256"] ==
                      SHA.bytes2hex(open(SHA.sha256, path))
                # The semantic validation fails.
                if :population in reads
                    @test_throws CorruptStore read_population(store, 2)
                else
                    @test read_population(store, 2) == rs_population(generation=2)
                end
                if :evaluations in reads
                    @test_throws CorruptStore read_evaluations(store; generation=2)
                else
                    @test length(read_evaluations(store; generation=2)) == 1
                end
                # The previous generation is untouched and readable, and the
                # invalid generation is never silently replaced by it.
                @test read_population(store, 1) == rs_population()
                @test length(read_evaluations(store; generation=1)) == 1
                @test !isempty(validate_store(store).errors)
            finally
                close_run!(store)
            end
        end

        @testset "non-normalized weights" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 2, :population,
                                 [population_fields(weights=[0.9, 0.9])])
                refresh_generation_digests!(directory, 2)
            end)
        end

        @testset "negative weight" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 2, :population,
                                 [population_fields(weights=[-0.5, 1.5])])
                refresh_generation_digests!(directory, 2)
            end)
        end

        @testset "nonfinite weight" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 2, :population,
                                 [population_fields(weights=[NaN, 1.0])])
                refresh_generation_digests!(directory, 2)
            end)
        end

        @testset "invalid effective sample size" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 2, :population,
                                 [population_fields(ess=99.0)])
                refresh_generation_digests!(directory, 2)
            end)
        end

        @testset "invalid ancestry index" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "population.jls")
                craft_partition!(path, "run-1", 2, :population,
                                 [population_fields(ancestry=[0, 1])])
                refresh_generation_digests!(directory, 2)
            end)
        end

        @testset "duplicate evaluation identity" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                craft_partition!(path, "run-1", 2, :evaluations,
                                 [evaluation_fields(), evaluation_fields()])
                refresh_generation_digests!(directory, 2;
                    updates = Dict("evaluation_count" => "2"))
            end; reads = (:evaluations,))
        end

        @testset "invalid success and failure relationship" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                craft_partition!(path, "run-1", 2, :evaluations,
                                 [evaluation_fields(status=:failed, failure=nothing)])
                refresh_generation_digests!(directory, 2)
            end; reads = (:evaluations,))
        end

        @testset "discrepancy attached to a failure" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                craft_partition!(path, "run-1", 2, :evaluations,
                                 [evaluation_fields(status=:failed,
                                                    summary=not_applicable_summary(),
                                                    failure=EvaluationFailure(
                                                        :simulator_error, "E", "m"),
                                                    discrepancy=0.5)])
                refresh_generation_digests!(directory, 2)
            end; reads = (:evaluations,))
        end

        @testset "nonfinite wall time" begin
            semantic_case((store, directory) -> begin
                path = joinpath(native_generation_dir(directory, 2), "evaluations.jls")
                craft_partition!(path, "run-1", 2, :evaluations,
                                 [evaluation_fields(wall_time_s=NaN)])
                refresh_generation_digests!(directory, 2)
            end; reads = (:evaluations,))
        end
    end

    @testset "orphan and incomplete artifacts" begin
        @testset "staging directory only is a warning" begin
            store, directory = durability_store(generations=1)
            try
                mkpath(joinpath(directory, "tmp", "generation_000002.deadbeef"))
                report = validate_store(store)
                @test report.valid
                @test any(w -> occursin("staging directory", w), report.warnings)
            finally
                close_run!(store)
            end
        end

        @testset "stale staging directories of other generations are warnings" begin
            store, directory = durability_store(generations=1)
            try
                mkpath(joinpath(directory, "tmp", "generation_000009.deadbeef"))
                mkpath(joinpath(directory, "tmp", "generation_000010.deadbeef"))
                report = validate_store(store)
                @test report.valid
                @test count(w -> occursin("staging directory", w), report.warnings) == 2
            finally
                close_run!(store)
            end
        end

        @testset "a generation directory without COMPLETE is an error" begin
            store, directory = durability_store(generations=1)
            try
                rm(joinpath(native_generation_dir(directory, 1), "COMPLETE"))
                report = validate_store(store)
                @test !report.valid
                @test any(e -> occursin("no COMPLETE marker", e), report.errors)
                @test_throws IncompleteGeneration read_population(store, 1)
            finally
                close_run!(store)
            end
        end

        @testset "COMPLETE without manifest is an error" begin
            store, directory = durability_store(generations=1)
            try
                rm(joinpath(native_generation_dir(directory, 1), "manifest"))
                report = validate_store(store)
                @test !report.valid
                @test_throws MissingManifest read_population(store, 1)
            finally
                close_run!(store)
            end
        end

        @testset "manifest without partitions is an error" begin
            store, directory = durability_store(generations=1)
            try
                rm(joinpath(native_generation_dir(directory, 1), "population.jls"))
                rm(joinpath(native_generation_dir(directory, 1), "evaluations.jls"))
                report = validate_store(store)
                @test !report.valid
                @test_throws CorruptStore read_population(store, 1)
            finally
                close_run!(store)
            end
        end

        @testset "a population without an evaluation partition is an error" begin
            store, directory = durability_store(generations=1)
            try
                rm(joinpath(native_generation_dir(directory, 1), "evaluations.jls"))
                report = validate_store(store)
                @test !report.valid
                @test_throws CorruptStore read_evaluations(store; generation=1)
            finally
                close_run!(store)
            end
        end

        @testset "an unexpected artifact in a generation directory is an error" begin
            store, directory = durability_store(generations=1)
            try
                write(joinpath(native_generation_dir(directory, 1), "surprise.txt"),
                      "unexpected\n")
                report = validate_store(store)
                @test !report.valid
                @test any(e -> occursin("unexpected artifact", e), report.errors)
            finally
                close_run!(store)
            end
        end

        @testset "a gap between committed generations is an error" begin
            store, directory = durability_store(generations=2)
            try
                mv(native_generation_dir(directory, 2),
                   native_generation_dir(directory, 3))
                report = validate_store(store)
                @test !report.valid
                @test any(e -> occursin("contiguous", e), report.errors)
            finally
                close_run!(store)
            end
        end

        @testset "a malformed generation directory name is an error" begin
            store, directory = durability_store(generations=1)
            try
                mkpath(joinpath(directory, "generations", "generation_1"))
                report = validate_store(store)
                @test !report.valid
                @test any(e -> occursin("malformed", e), report.errors)
            finally
                close_run!(store)
            end
        end

        @testset "a duplicate generation representation is an error" begin
            store, directory = durability_store(generations=1)
            try
                mkpath(joinpath(directory, "generations", "generation_000001"))
                write(joinpath(directory, "generations", "generation_000001",
                               "surprise.txt"), "duplicate\n")
                report = validate_store(store)
                @test !report.valid
                @test any(e -> occursin("unexpected artifact", e) ||
                                occursin("malformed", e), report.errors)
            finally
                close_run!(store)
            end
        end
    end

    @testset "lease matrix" begin
        @testset "a second writer is rejected and the lease survives" begin
            store, directory = durability_store(generations=1)
            try
                @test_throws WriterAlreadyActive open_run(
                    NativeRunStore, directory; mode=:write)
                @test isdir(joinpath(directory, "metadata", "writer.lease"))
            finally
                close_run!(store)
            end
        end

        @testset "a reader can read while the lease is active" begin
            store, directory = durability_store(generations=1)
            try
                reader = open_run(NativeRunStore, directory; mode=:read)
                try
                    @test generations(reader) == [1]
                    @test length(read_evaluations(reader)) == 1
                finally
                    close_run!(reader)
                end
                # Closing the reader did not touch the writer's lease.
                @test isdir(joinpath(directory, "metadata", "writer.lease"))
            finally
                close_run!(store)
            end
        end

        @testset "a malformed holder record is reported and never removed" begin
            store, directory = durability_store(generations=1)
            try
                holder = joinpath(directory, "metadata", "writer.lease", "holder")
                write(holder, "not a holder\n")
                @test_throws StaleWriterLease open_run(
                    NativeRunStore, directory; mode=:write)
                @test isfile(holder)
            finally
                close_run!(store)
            end
        end

        @testset "a foreign nonce does not grant ownership" begin
            store, directory = durability_store(generations=1)
            try
                holder = joinpath(directory, "metadata", "writer.lease", "holder")
                entries = read_canonical(holder)
                entries["nonce"] = repeat("f", 16)
                write_canonical(holder, entries,
                                ["schema", "version", "run_id", "hostname", "pid",
                                 "acquired_at", "nonce"])
                @test_throws WriterAlreadyActive open_run(
                    NativeRunStore, directory; mode=:write)
                @test read_canonical(holder)["nonce"] == repeat("f", 16)
            finally
                close_run!(store)
            end
        end

        @testset "close_run! removes only the lease it owns" begin
            store, directory = durability_store(generations=1)
            reader = open_run(NativeRunStore, directory; mode=:read)
            try
                close_run!(reader)
                @test isdir(joinpath(directory, "metadata", "writer.lease"))
            finally
                close_run!(store)
            end
            @test !isdir(joinpath(directory, "metadata", "writer.lease"))
        end

        @testset "writes after close are rejected" begin
            store, directory = durability_store(generations=1)
            close_run!(store)
            @test_throws StoreClosed write_population!(store, rs_population(generation=2))
            @test_throws StoreClosed write_evaluation!(
                store, EvaluationRecord(rs_success(rs_context(generation=2))))
            @test_throws StoreClosed finalize_generation!(store, 2)
        end

        @testset "discarded pending records are reported" begin
            store, directory = durability_store(generations=1)
            write_population!(store, rs_population(generation=2))
            write_evaluation!(store, EvaluationRecord(rs_success(rs_context(generation=2))))
            @test_logs (:warn, r"discards a pending generation") match_mode = :any close_run!(store)
        end

        @testset "a stale lease is removed only by hand" begin
            store, directory = durability_store(generations=1)
            lease = joinpath(directory, "metadata", "writer.lease")
            close_run!(store)
            mkpath(lease)
            write(joinpath(lease, "holder"), "not a holder\n")
            @test_throws StaleWriterLease open_run(
                NativeRunStore, directory; mode=:write)
            @test isdir(lease)
            rm(lease; recursive = true, force = true)
            writer = open_run(NativeRunStore, directory; mode=:write)
            try
                @test isdir(lease)
            finally
                close_run!(writer)
            end
        end
    end

    @testset "previous-generation retention" begin
        for phase in PRE_RENAME_PHASES
            @testset "$(phase)" begin
                store, directory = durability_store(generations=2)
                try
                    before = generation_snapshot(directory)
                    queries_before = read_evaluations(store)
                    observed = Symbol[]
                    arm_fault!(store, phase, observed)

                    write_population!(store, rs_population(generation=3))
                    write_evaluation!(store, EvaluationRecord(
                        rs_success(rs_context(generation=3))))
                    @test_throws SequentialABC._InjectedInterruption finalize_generation!(store, 3)

                    # Nothing committed was rewritten.
                    @test generation_snapshot(directory)["generation_000001/population.jls"] ==
                          before["generation_000001/population.jls"]
                    @test generation_snapshot(directory)["generation_000001/evaluations.jls"] ==
                          before["generation_000001/evaluations.jls"]
                    @test generation_snapshot(directory)["generation_000002/population.jls"] ==
                          before["generation_000002/population.jls"]
                    @test generation_snapshot(directory)["generation_000002/evaluations.jls"] ==
                          before["generation_000002/evaluations.jls"]
                    @test generation_snapshot(directory)["generation_000002/manifest"] ==
                          before["generation_000002/manifest"]
                    @test generation_snapshot(directory)["generation_000002/COMPLETE"] ==
                          before["generation_000002/COMPLETE"]

                    # The committed history is unchanged.
                    @test generations(store) == [1, 2]
                    @test latest_generation(store) == 2
                    @test read_evaluations(store) == queries_before
                    @test read_population(store, 1) == rs_population()
                    @test read_population(store, 2) == rs_population(generation=2)

                    # The failed attempt is reported as staging evidence only
                    # (the staging directory exists from `:after_staging_created`
                    # onwards); the pending generation is still in memory.
                    report = validate_store(store)
                    @test report.valid
                    @test any(w -> occursin("pending", w), report.warnings)
                    if phase !== :before_staging
                        @test any(w -> occursin("staging directory", w), report.warnings)
                    end
                    @test isempty(report.errors)
                finally
                    close_quietly!(store)
                end
            end
        end
    end

    @testset "canonical encoding golden fixtures" begin
        # The canonical text encoding is platform independent: fixed key
        # order, LF line endings, and no path separators.
        directory = native_test_directory()
        schema_path = joinpath(directory, "schema")
        SequentialABC._write_canonical_kv(schema_path,
            Pair{String,String}[
                "schema" => RUNSTORE_SCHEMA_NAME,
                "version" => string(RUNSTORE_SCHEMA_VERSION),
            ])
        @test read(schema_path, String) == "schema=sequentialabc.runstore\nversion=1\n"
        @test !occursin("\r", read(schema_path, String))

        manifest_path = joinpath(directory, "manifest")
        SequentialABC._write_canonical_kv(manifest_path, Pair{String,String}[
            "schema" => RUNSTORE_SCHEMA_NAME,
            "version" => "1",
            "run_id" => "run-1",
            "generation" => "1",
            "metadata_sha256" => repeat("a", 64),
            "population_file" => "population.jls",
            "population_sha256" => repeat("b", 64),
            "population_bytes" => "128",
            "population_count" => "1",
            "evaluation_file" => "evaluations.jls",
            "evaluation_sha256" => repeat("c", 64),
            "evaluation_bytes" => "256",
            "evaluation_count" => "2",
            "evaluation_key_set_sha256" => repeat("d", 64),
            "omission_count" => "0",
            "package_version" => "0.4.0",
            "julia_version" => "1.10.0",
            "completed_at" => "2026-01-02T03:04:05.000",
        ])
        expected_manifest = """schema=sequentialabc.runstore
version=1
run_id=run-1
generation=1
metadata_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
population_file=population.jls
population_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
population_bytes=128
population_count=1
evaluation_file=evaluations.jls
evaluation_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
evaluation_bytes=256
evaluation_count=2
evaluation_key_set_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
omission_count=0
package_version=0.4.0
julia_version=1.10.0
completed_at=2026-01-02T03:04:05.000
"""
        @test read(manifest_path, String) == expected_manifest
        @test !occursin("\r", read(manifest_path, String))
        manifest_text = read(manifest_path, String)
        @test !occursin("\\", manifest_text)
        @test occursin("population_file=population.jls\n", manifest_text)
        @test occursin("evaluation_file=evaluations.jls\n", manifest_text)

        marker_path = joinpath(directory, "COMPLETE")
        SequentialABC._write_canonical_kv(marker_path, Pair{String,String}[
            "schema" => RUNSTORE_SCHEMA_NAME,
            "version" => "1",
            "run_id" => "run-1",
            "generation" => "1",
            "manifest_file" => "manifest",
            "manifest_sha256" => repeat("e", 64),
            "completed_at" => "2026-01-02T03:04:05.000",
        ])
        expected_marker = """schema=sequentialabc.runstore
version=1
run_id=run-1
generation=1
manifest_file=manifest
manifest_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
completed_at=2026-01-02T03:04:05.000
"""
        @test read(marker_path, String) == expected_marker
        @test !occursin("\r", read(marker_path, String))

        # The identity encoding and its digest are fixed.
        key = EvaluationKey(rs_context(generation=1, particle=2, proposal=3,
                                       attempt=4, replicate=5))
        @test SequentialABC._evaluation_key_line(key) == "smc|1|2|3|4|5\n"
        @test SequentialABC._evaluation_key_set_hash([key]) ==
              SequentialABC._evaluation_key_set_hash([key])
        @test length(SequentialABC._evaluation_key_set_hash([key])) == 64
        # The expected plan uses the same encoding, so a matching plan hashes
        # like the recorded identities.
        @test SequentialABC._evaluation_key_set_hash([key]) ==
              SequentialABC._evaluation_key_set_hash(
                  [EvaluationKey(rs_context(generation=1, particle=2, proposal=3,
                                             attempt=4, replicate=5))])
    end

    @testset "native partitions round trip and detect damage" begin
        store, directory = durability_store(generations=1)
        try
            generation = native_generation_dir(directory, 1)
            manifest = read_canonical(joinpath(generation, "manifest"))
            # The digest of the partition matches its local manifest.
            @test manifest["population_sha256"] ==
                  SHA.bytes2hex(open(SHA.sha256, joinpath(generation, "population.jls")))
            @test manifest["evaluation_sha256"] ==
                  SHA.bytes2hex(open(SHA.sha256, joinpath(generation, "evaluations.jls")))
            # The semantic round trip holds.
            @test read_population(store, 1) == rs_population()
            @test length(read_evaluations(store; generation=1)) == 1
        finally
            close_run!(store)
        end
    end
end

for directory in NATIVE_TEST_DIRECTORIES
    isdir(directory) && rm(directory; recursive = true, force = true)
end
