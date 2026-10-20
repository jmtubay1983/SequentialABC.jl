using SequentialABC
using Test
using Dates
using Distributions
using Random

# The backend-independent conformance suite lives in its own file so that the
# in-memory and the native backend are held to the same contract.
isdefined(Main, :run_store_conformance_suite) || include("runstore_conformance.jl")

@testset "RunStore memory store specifics" begin
    @testset "construction and identity" begin
        metadata = rs_metadata()
        store = initialize_run!(MemoryRunStore; metadata=metadata)
        @test store isa MemoryRunStore
        @test store isa AbstractRunStore
        @test run_metadata(store) == metadata
        @test schema_version(store) == RUNSTORE_SCHEMA_VERSION
        @test schema_version(MemoryRunStore) == RUNSTORE_SCHEMA_VERSION
        @test occursin("MemoryRunStore", sprint(show, store))

        # The metadata is copied out, so a caller cannot mutate the identity.
        returned = run_metadata(store)
        @test returned == metadata

        # Closing an in-memory store is a no-op: it owns no external resource.
        @test close_run!(store) === nothing
    end

    @testset "pending state is in memory and reported" begin
        store = initialize_run!(MemoryRunStore; metadata=rs_metadata())
        write_population!(store, rs_population())
        write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))

        # The pending generation is visible in the store but not as a
        # committed generation.
        @test generations(store) == Int[]
        @test latest_generation(store) === nothing
        @test length(store.pending_evaluations[1]) == 1
        @test haskey(store.pending_populations, 1)

        report = validate_store(store)
        @test report.valid
        @test any(w -> occursin("generation 1", w), report.warnings)
        @test any(w -> occursin("never finalized", w), report.warnings)
    end

    @testset "the in-memory backend never touches the filesystem" begin
        store = initialize_run!(MemoryRunStore; metadata=rs_metadata())
        write_population!(store, rs_population())
        write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
        finalize_generation!(store, 1)
        @test read_population(store, 1) == rs_population()
        @test length(read_evaluations(store)) == 1
    end
end

run_store_conformance_suite(
    "MemoryRunStore",
    metadata -> initialize_run!(MemoryRunStore; metadata=metadata),
)
