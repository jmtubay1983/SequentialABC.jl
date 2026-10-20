using SequentialABC
using Test
using Dates
using Distributions
using Random

# ============================================================================
# Backend conformance suite for the run-history contract.
#
# The suite is written against the public `AbstractRunStore` semantics only,
# so every backend is held to the same contract. It is executed for the
# in-memory reference backend and for the native, file-backed backend.
#
# `make_store(metadata)` must return a fresh store for the given metadata; the
# suite closes every store it opens, so a backend that owns an external
# resource (such as a writer lease) releases it.
# ============================================================================

const RS_METHOD = :smc

# A custom summary type that has no canonical representation. It is used to
# check that unsupported values fail by default and that an explicit omission
# stays visible.
struct RSOpaqueSummary end

rs_metadata(; run_id="run-1", method=RS_METHOD, base_seed=42,
            package_version="0.4.0", julia_version="1.10.0",
            source_revision=nothing,
            user_provenance=RecordedProvenance(),
            seed_schema_name=SequentialABC.EVALUATION_SEED_SCHEMA_NAME,
            seed_schema_version=EVALUATION_SEED_SCHEMA_VERSION,
            storage_schema_name=RUNSTORE_SCHEMA_NAME,
            storage_schema_version=RUNSTORE_SCHEMA_VERSION,
            created_at=DateTime(2026, 1, 2, 3, 4, 5)) = RunMetadata(
    run_id, method, package_version, julia_version, source_revision,
    seed_schema_name, seed_schema_version, base_seed, created_at,
    user_provenance, storage_schema_name, storage_schema_version,
)

rs_context(; generation=1, particle=1, proposal=1, attempt=1, replicate=1,
           method=RS_METHOD, base_seed=UInt64(42)) = EvaluationContext(
    method, generation, particle, proposal, attempt, replicate, base_seed)

rs_success(context; summary=[1.0, 2.0], provenance=NamedTuple(),
            wall_time_s=0.25) = EvaluationResult(
    context, :success, summary, provenance; wall_time_s=wall_time_s)

rs_failure(context; category=:simulator_error, status=:failed,
            provenance=NamedTuple(), wall_time_s=0.5) = EvaluationResult(
    context, status, nothing, provenance;
    failure=EvaluationFailure(category, "TestError", "deliberate failure"),
    wall_time_s=wall_time_s,
)

rs_population(; generation=1, particles=[[1.0], [2.0]], weights=[0.5, 0.5],
              distances=[0.1, 0.2], epsilon=1.0, ess=2.0, attempts=nothing,
              ancestry=nothing, resampling_count=0, adaptive=nothing,
              started_at=DateTime(2026, 1, 2, 3, 4, 5),
              finished_at=DateTime(2026, 1, 2, 3, 4, 6),
              provenance=RecordedProvenance()) = PopulationRecord(
    generation, particles, weights, distances, epsilon, ess, attempts,
    ancestry, resampling_count, adaptive, started_at, finished_at, provenance,
)

"""
    run_store_conformance_suite(label::AbstractString, make_store)

Run the backend-independent conformance suite against one run-store backend.

`make_store(metadata)` must return a fresh store. The suite closes every
store it opens.
"""
function run_store_conformance_suite(label::AbstractString, make_store)
    @testset "RunStore conformance ($label)" begin

        @testset "canonical recorded values" begin
            @test recorded_value(nothing) == RecordedNull()
            @test recorded_value(true) == RecordedBool(true)
            @test recorded_value(3) == RecordedInt(3)
            @test recorded_value(Int32(3)) == RecordedInt(3)
            @test recorded_value(UInt64(3)) == RecordedUInt(3)
            @test recorded_value(1.5) == RecordedFloat(1.5)
            @test recorded_value(Float32(1.5)) == RecordedFloat(1.5)
            @test recorded_value("text") == RecordedString("text")
            @test recorded_value(:sym) == RecordedSymbol(:sym)

            vec_rec = recorded_value([1.0, 2.0])
            @test vec_rec isa RecordedVector
            @test vec_rec.values == RecordedValue[RecordedFloat(1.0), RecordedFloat(2.0)]

            tup_rec = recorded_value((1.0, "a"))
            @test tup_rec isa RecordedTuple
            @test tup_rec.values == RecordedValue[RecordedFloat(1.0), RecordedString("a")]

            nt_rec = recorded_value((a=1, b=[1.0]))
            @test nt_rec isa RecordedNamedTuple
            @test nt_rec.names == [:a, :b]
            @test nt_rec.values == RecordedValue[RecordedInt(1),
                                                 RecordedVector(RecordedValue[RecordedFloat(1.0)])]

            # Conversion is deterministic and recursively validated.
            @test recorded_value([1.0, [2.0, 3.0]]) == recorded_value([1.0, [2.0, 3.0]])
            @test recorded_value([1.0, [2.0, 3.0]]) isa RecordedVector
            @test recorded_value([1.0, [2.0, 3.0]]).values[2] isa RecordedVector

            # Integers are checked against the fixed-width ranges.
            @test recorded_value(big(2)^63) == RecordedUInt(big(2)^63)
            @test recorded_value(Int128(big(2)^63)) == RecordedUInt(big(2)^63)
            @test_throws UnsupportedRecordedValue recorded_value(big(2)^100)
            @test_throws UnsupportedRecordedValue recorded_value(UInt128(big(2)^100))

            # Nonfinite numbers fail; only a tagged sentinel represents one.
            @test_throws NonfiniteRecordedValue recorded_value(NaN)
            @test_throws NonfiniteRecordedValue recorded_value(Inf)
            @test_throws NonfiniteRecordedValue recorded_value(-Inf)
            @test_throws NonfiniteRecordedValue recorded_value([1.0, NaN])
            @test recorded_value(RecordedSentinel(:nan)) == RecordedSentinel(:nan)
            @test recorded_value(RecordedSentinel(:positive_infinity)) ==
                RecordedSentinel(:positive_infinity)
            @test_throws ArgumentError RecordedSentinel(:not_a_kind)
            @test RECORDED_SENTINEL_KINDS == (:nan, :positive_infinity, :negative_infinity)

            # Unsupported values fail by default and are never stored implicitly.
            @test_throws UnsupportedRecordedValue recorded_value(Dict(:a => 1))
            @test_throws UnsupportedRecordedValue recorded_value(1 + 2im)
            @test_throws UnsupportedRecordedValue recorded_value(missing)

            # An explicit omission is visible and carries a type and a reason.
            omitted = recorded_value(Dict(:a => 1); omit=true, reason="not representable")
            @test omitted isa RecordedOmitted
            @test omitted.type_name == "Dict{Symbol, Int64}"
            @test omitted.reason == "not representable"

            # An already canonical value passes through unchanged.
            @test recorded_value(RecordedOmitted("T", "r")) == RecordedOmitted("T", "r")
        end

        @testset "provenance canonicalization" begin
            prov = RecordedProvenance((model="toy", revision=3))
            @test prov.entries == Pair{String,RecordedValue}[
                "model" => RecordedString("toy"),
                "revision" => RecordedInt(3),
            ]

            @test isempty(RecordedProvenance().entries)
            @test RecordedProvenance(["a" => 1]).entries ==
                Pair{String,RecordedValue}["a" => RecordedInt(1)]

            # Keys must be nonempty and unique.
            @test_throws ArgumentError RecordedProvenance(["" => 1])
            @test_throws ArgumentError RecordedProvenance(["a" => 1, "a" => 2])

            # Only a NamedTuple or a RecordedProvenance is a canonical provenance.
            single = (a=1,)
            @test recorded_provenance(single) == RecordedProvenance(single)
            @test recorded_provenance(prov) == prov
            @test_throws ArgumentError recorded_provenance("not a provenance")

            # Unsupported provenance values fail by default and can be omitted.
            @test_throws UnsupportedRecordedValue RecordedProvenance((a=Dict(:b => 1),))
            omitted = RecordedProvenance((a=Dict(:b => 1),); omit=true, reason="opaque")
            @test omitted.entries[1].second isa RecordedOmitted
            @test omitted.entries[1].second.reason == "opaque"
        end

        @testset "metadata invariants" begin
            metadata = rs_metadata()
            @test metadata.run_id == "run-1"
            @test metadata.method == :smc
            @test metadata.package_version == "0.4.0"
            @test metadata.julia_version == "1.10.0"
            @test metadata.source_revision === nothing
            @test metadata.seed_schema_name == SequentialABC.EVALUATION_SEED_SCHEMA_NAME
            @test metadata.seed_schema_version == EVALUATION_SEED_SCHEMA_VERSION
            @test metadata.base_seed == 42
            @test metadata.created_at == DateTime(2026, 1, 2, 3, 4, 5)
            @test metadata.storage_schema_name == RUNSTORE_SCHEMA_NAME
            @test metadata.storage_schema_version == RUNSTORE_SCHEMA_VERSION

            @test_throws ArgumentError rs_metadata(run_id="")
            @test_throws ArgumentError rs_metadata(method=Symbol(""))
            @test_throws ArgumentError rs_metadata(base_seed=big(2)^100)
            @test_throws ArgumentError rs_metadata(package_version="")

            # The storage schema and the seed schema are independent concepts.
            legacy = rs_metadata(
                seed_schema_name=SequentialABC.LEGACY_PARTICLE_SEED_SCHEMA_NAME,
                seed_schema_version=UInt64(0))
            @test legacy.seed_schema_name == SequentialABC.LEGACY_PARTICLE_SEED_SCHEMA_NAME
            @test legacy.storage_schema_version == RUNSTORE_SCHEMA_VERSION
            @test legacy.seed_schema_version != legacy.storage_schema_version

            # The current metadata helper fills in the package and Julia versions
            # and a UTC creation timestamp; it never consults Git.
            current = current_run_metadata(run_id="run-2", method=:smc, base_seed=7)
            @test current.run_id == "run-2"
            @test !isempty(current.package_version)
            @test !isempty(current.julia_version)
            @test current.source_revision === nothing
            @test current.storage_schema_name == RUNSTORE_SCHEMA_NAME
            @test current.storage_schema_version == RUNSTORE_SCHEMA_VERSION
            @test current.created_at isa DateTime
        end

        @testset "evaluation identity" begin
            context = rs_context(generation=2, particle=3, proposal=4, attempt=5,
                                 replicate=6)
            key = EvaluationKey(context)
            @test key.method == :smc
            @test key.generation == 2
            @test key.particle == 3
            @test key.proposal == 4
            @test key.attempt == 5
            @test key.replicate == 6

            @test key == EvaluationKey(rs_context(generation=2, particle=3, proposal=4,
                                                  attempt=5, replicate=6))
            @test key != EvaluationKey(rs_context(generation=2, particle=3, proposal=4,
                                                   attempt=5, replicate=7))
            @test hash(key) == hash(EvaluationKey(rs_context(generation=2, particle=3,
                                                             proposal=4, attempt=5,
                                                             replicate=6)))
            @test length(Set([key, EvaluationKey(rs_context(generation=2, particle=3,
                                                              proposal=4, attempt=5,
                                                              replicate=6))])) == 1

            # Deterministic total order.
            a = EvaluationKey(rs_context(generation=1, particle=1, proposal=1, attempt=1,
                                         replicate=1))
            b = EvaluationKey(rs_context(generation=1, particle=1, proposal=1, attempt=1,
                                         replicate=2))
            @test isless(a, b)
            @test !isless(b, a)
            @test sort([b, a]) == [a, b]

            @test_throws ArgumentError EvaluationKey(:smc, 0, 1, 1, 1, 1)
            @test_throws ArgumentError EvaluationKey(:smc, 1, 1, 1, 1, 0)
            @test_throws ArgumentError EvaluationKey(:smc, 1, 1, 1, 1, big(2)^100)
        end

        @testset "evaluation record invariants" begin
            context = rs_context()
            result = rs_success(context; summary=(a=1.0, b=[2.0]),
                                 provenance=(model="toy",), wall_time_s=0.25)
            record = EvaluationRecord(result)

            @test record.context == context
            @test record.status == :success
            @test record.summary.state == :recorded
            @test record.summary.value == recorded_value((a=1.0, b=[2.0]))
            @test occursin("NamedTuple", record.summary.type_name)
            @test record.failure === nothing
            @test record.wall_time_s == 0.25
            @test record.provenance == RecordedProvenance((model="toy",))
            @test record.discrepancy === nothing
            @test EvaluationKey(record) == EvaluationKey(context)

            # A discrepancy is a value the caller calculated; the store never
            # derives it.
            with_discrepancy = EvaluationRecord(result; discrepancy=0.75)
            @test with_discrepancy.discrepancy == 0.75
            @test_throws ArgumentError EvaluationRecord(result; discrepancy=NaN)
            @test_throws ArgumentError EvaluationRecord(result; discrepancy=-1.0)

            # Failures keep their structured failure and never carry a discrepancy.
            failure = rs_failure(context; category=:timeout, status=:timeout)
            failure_record = EvaluationRecord(failure)
            @test failure_record.status == :timeout
            @test failure_record.summary.state == :not_applicable
            @test failure_record.failure.category == :timeout
            @test failure_record.failure.exception_type == "TestError"
            @test failure_record.discrepancy === nothing
            @test_throws ArgumentError EvaluationRecord(failure; discrepancy=0.1)

            # Non-success statuses other than :failed are supported.
            @test EvaluationRecord(rs_failure(context; category=:nonfinite_summary,
                                              status=:nonfinite)).status == :nonfinite

            # Wall time must be finite and nonnegative.
            @test_throws ArgumentError EvaluationRecord(
                context, :success, recorded_summary([1.0]), nothing, -1.0,
                RecordedProvenance(), nothing)
            @test_throws ArgumentError EvaluationRecord(
                context, :success, recorded_summary([1.0]), nothing, Inf,
                RecordedProvenance(), nothing)

            # A success may not carry a failure, and a failure may not carry a summary.
            @test_throws ArgumentError EvaluationRecord(
                context, :success, recorded_summary([1.0]),
                EvaluationFailure(:simulator_error, "E", "m"), 0.1,
                RecordedProvenance(), nothing)
            @test_throws ArgumentError EvaluationRecord(
                context, :failed, recorded_summary([1.0]),
                EvaluationFailure(:simulator_error, "E", "m"), 0.1,
                RecordedProvenance(), nothing)
            @test_throws ArgumentError EvaluationRecord(
                context, :failed, not_applicable_summary(), nothing, 0.1,
                RecordedProvenance(), nothing)

            # An unsupported summary fails by default and can be omitted
            # explicitly; an omission stays visible.
            opaque_result = EvaluationResult(context, :success, RSOpaqueSummary(),
                                              NamedTuple(); wall_time_s=0.1)
            @test_throws UnsupportedRecordedValue EvaluationRecord(opaque_result)
            omitted = EvaluationRecord(opaque_result; omit_unsupported_summary=true,
                                        omission_reason="opaque summary type")
            @test omitted.summary.state == :omitted
            @test omitted.summary.type_name == string(RSOpaqueSummary)
            @test omitted.summary.reason == "opaque summary type"
            @test omitted.summary.value === nothing

            # A non-canonical provenance object is rejected.
            odd_result = EvaluationResult(context, :success, [1.0], "not a provenance";
                                          wall_time_s=0.1)
            @test_throws ArgumentError EvaluationRecord(odd_result)
        end

        @testset "population record invariants" begin
            population = rs_population()
            @test population.generation == 1
            @test population.particles == [[1.0], [2.0]]
            @test population.weights == [0.5, 0.5]
            @test population.distances == [0.1, 0.2]
            @test population.epsilon == 1.0
            @test population.ess == 2.0
            @test population.attempts === nothing
            @test population.ancestry === nothing
            @test population.resampling_count == 0
            @test population.adaptive === nothing
            @test population.started_at <= population.finished_at
            @test population.provenance == RecordedProvenance()

            # The generation-one positive-infinity epsilon sentinel is
            # documented and allowed; it is not allowed anywhere else.
            @test rs_population(epsilon=Inf).epsilon == Inf
            @test_throws ArgumentError rs_population(generation=2, epsilon=Inf)
            @test_throws ArgumentError rs_population(epsilon=NaN)
            @test_throws ArgumentError rs_population(epsilon=-1.0)

            # Aligned vector lengths.
            @test_throws ArgumentError rs_population(weights=[1.0])
            @test_throws ArgumentError rs_population(distances=[0.1])
            @test_throws ArgumentError rs_population(attempts=[1])
            @test_throws ArgumentError rs_population(ancestry=[1])

            # Weights must be finite, nonnegative, and normalized.
            @test_throws ArgumentError rs_population(weights=[0.5, 0.6])
            @test_throws ArgumentError rs_population(weights=[-0.5, 1.5])
            @test_throws ArgumentError rs_population(weights=[NaN, 1.0])

            # Distances must be finite and nonnegative.
            @test_throws ArgumentError rs_population(distances=[0.1, -0.2])
            @test_throws ArgumentError rs_population(distances=[0.1, Inf])

            # The effective sample size must lie in [1, N] up to the tolerance.
            @test rs_population(ess=1.0).ess == 1.0
            @test_throws ArgumentError rs_population(ess=0.5)
            @test_throws ArgumentError rs_population(ess=3.0)
            @test_throws ArgumentError rs_population(ess=NaN)

            # Optional attempt counts and ancestry indices.
            @test rs_population(attempts=[1, 4]).attempts == [1, 4]
            @test_throws ArgumentError rs_population(attempts=[0, 1])
            @test rs_population(ancestry=[1, 2]).ancestry == [1, 2]
            @test_throws ArgumentError rs_population(ancestry=[0, 1])
            @test_throws ArgumentError rs_population(ancestry=[1, 3])

            # Timestamps must be ordered.
            @test_throws ArgumentError rs_population(
                started_at=DateTime(2026, 1, 2, 3, 4, 6),
                finished_at=DateTime(2026, 1, 2, 3, 4, 5))

            # Acceptance consistency is a method-specific property and is
            # therefore not enforced by the generic constructor.
            loose = rs_population(distances=[0.1, 5.0], epsilon=1.0)
            @test loose.distances[2] == 5.0
            @test_throws InvalidRunStoreRecord check_acceptance_consistency(loose)
            @test check_acceptance_consistency(rs_population()) == true
            @test check_acceptance_consistency(rs_population(epsilon=Inf)) == true

            # Optional adaptive diagnostics are validated too.
            adaptive = AdaptiveDiagnostics([:a, :b], [1.0, 2.0], 1, [0.1, 0.2], 0.5)
            @test rs_population(adaptive=adaptive).adaptive == adaptive
            @test_throws ArgumentError AdaptiveDiagnostics([:a, :b], [1.0], 1, [0.1], 0.5)
            @test_throws ArgumentError AdaptiveDiagnostics([:a, :a], [1.0], 1, [0.1], 0.5)
            @test_throws ArgumentError AdaptiveDiagnostics([:a], [0.0], 1, [0.1], 0.5)
            @test_throws ArgumentError AdaptiveDiagnostics([:a], [1.0], 0, [0.1], 0.5)
            @test_throws ArgumentError AdaptiveDiagnostics([:a], [1.0], 1, [-0.1], 0.5)
            @test_throws ArgumentError AdaptiveDiagnostics([:a], [1.0], 1, [0.1], -1.0)
            @test_throws ArgumentError rs_population(
                adaptive=AdaptiveDiagnostics([:a], [1.0], 1, [0.1, 0.2, 0.3], 0.5))
            @test rs_population(
                adaptive=AdaptiveDiagnostics([:a], [1.0], 1, Float64[], 0.5)).adaptive != nothing
        end

        @testset "store lifecycle" begin
            metadata = rs_metadata()
            store = make_store(metadata)
            try
                @test store isa AbstractRunStore
                @test run_metadata(store) == metadata
                @test schema_version(store) == RUNSTORE_SCHEMA_VERSION
                @test latest_generation(store) === nothing
                @test generations(store) == Int[]
                @test isempty(read_evaluations(store))

                population = rs_population()
                write_population!(store, population)
                write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
                write_evaluation!(store, EvaluationRecord(
                    rs_failure(rs_context(particle=2); category=:timeout,
                                status=:timeout)))
                completion = finalize_generation!(store, 1)
                @test completion.generation == 1
                @test completion.population_size == 2
                @test completion.evaluation_count == 2
                @test completion.finalized_at isa DateTime

                @test latest_generation(store) == 1
                @test generations(store) == [1]
                @test read_population(store, 1) == population
                @test length(read_evaluations(store)) == 2

                # Reads return copies: a caller cannot mutate the stored history.
                returned = read_population(store, 1)
                returned.weights[1] = 99.0
                @test read_population(store, 1).weights == [0.5, 0.5]

                # A second generation follows the first.
                write_population!(store, rs_population(generation=2, particles=[[3.0], [4.0]],
                                                       weights=[0.5, 0.5],
                                                       distances=[0.3, 0.4]))
                write_evaluation!(store, EvaluationRecord(
                    rs_success(rs_context(generation=2, particle=1))))
                finalize_generation!(store, 2)
                @test latest_generation(store) == 2
                @test generations(store) == [1, 2]

                # The store can be reused under identical metadata and refuses
                # to be reused under conflicting metadata.
                @test initialize_run!(store; metadata=metadata) === store
                @test open_run(store; metadata=metadata) === store
                @test_throws ConflictingRunMetadata initialize_run!(
                    store; metadata=rs_metadata(run_id="other"))
                @test_throws ConflictingRunMetadata open_run(
                    store; metadata=rs_metadata(base_seed=99))
            finally
                close_run!(store)
            end
        end

        @testset "write-time enforcement" begin
            store = make_store(rs_metadata())
            try
                # Generations are contiguous and start at 1.
                @test_throws GenerationGap write_population!(store, rs_population(generation=2))

                write_population!(store, rs_population())

                # A generation holds exactly one population.
                @test_throws DuplicatePopulation write_population!(store, rs_population())

                # Evaluation identities are unique inside a generation.
                write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
                @test_throws DuplicateEvaluation write_evaluation!(
                    store, EvaluationRecord(rs_success(rs_context())))

                # The method of an evaluation must match the method of the run.
                other = make_store(rs_metadata(method=:pmc))
                try
                    write_population!(other, rs_population())
                    @test_throws MethodMismatch write_evaluation!(
                        other, EvaluationRecord(rs_success(rs_context())))
                finally
                    close_run!(other)
                end

                # A generation without a population cannot be completed.
                empty_store = make_store(rs_metadata())
                try
                    @test_throws MissingPopulation finalize_generation!(empty_store, 1)
                finally
                    close_run!(empty_store)
                end

                # An expected plan detects missing and extra evaluations.
                expected = [EvaluationKey(rs_context()), EvaluationKey(rs_context(particle=2))]
                @test_throws MissingExpectedEvaluation finalize_generation!(
                    store, 1; expected_evaluations=expected)
                write_evaluation!(store, EvaluationRecord(rs_success(rs_context(particle=2))))
                @test_throws UnexpectedEvaluation finalize_generation!(
                    store, 1; expected_evaluations=[EvaluationKey(rs_context())])
                @test finalize_generation!(store, 1; expected_evaluations=expected) isa
                    GenerationCompletion

                # A complete generation is immutable.
                @test_throws GenerationAlreadyComplete write_population!(
                    store, rs_population())
                @test_throws GenerationAlreadyComplete write_evaluation!(
                    store, EvaluationRecord(rs_success(rs_context(particle=3))))
                @test_throws GenerationAlreadyComplete finalize_generation!(store, 1)
            finally
                close_run!(store)
            end
        end

        @testset "queries" begin
            store = make_store(rs_metadata())
            try
                write_population!(store, rs_population())
                # Written out of order on purpose: the query order must not
                # depend on it.
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

                all_records = read_evaluations(store)
                @test length(all_records) == 4
                @test [EvaluationKey(r) for r in all_records] ==
                    sort([EvaluationKey(r) for r in all_records])
                @test read_evaluations(store) == all_records  # deterministic

                @test length(read_evaluations(store; generation=1)) == 4
                @test_throws MissingGeneration read_evaluations(store; generation=2)

                @test length(read_evaluations(store; particle=1)) == 2
                @test length(read_evaluations(store; particle=[1, 3])) == 3
                @test length(read_evaluations(store; attempt=1)) == 3
                @test length(read_evaluations(store; attempt=2)) == 1
                @test length(read_evaluations(store; proposal=2)) == 1
                @test length(read_evaluations(store; replicate=2)) == 1
                @test length(read_evaluations(store; status=:success)) == 3
                @test length(read_evaluations(store; status=[:success, :failed])) == 4
                @test length(read_evaluations(store; failure_category=:simulator_error)) == 1
                @test length(read_evaluations(store; failure_category=[:timeout])) == 0

                @test length(read_evaluations(store; min_wall_time_s=1.0)) == 2
                @test length(read_evaluations(store; max_wall_time_s=0.5)) == 2
                @test length(read_evaluations(store; min_wall_time_s=0.5,
                                               max_wall_time_s=1.0)) == 2
                @test_throws ArgumentError read_evaluations(store; min_wall_time_s=1.0,
                                                             max_wall_time_s=0.5)
                @test_throws ArgumentError read_evaluations(store; min_wall_time_s=-1.0)

                # Combined filters.
                @test length(read_evaluations(store; particle=1, status=:success)) == 2
            finally
                close_run!(store)
            end
        end

        @testset "validation" begin
            store = make_store(rs_metadata())
            try
                write_population!(store, rs_population())
                write_evaluation!(store, EvaluationRecord(rs_success(rs_context())))
                finalize_generation!(store, 1)

                report = validate_store(store)
                @test report.valid
                @test isempty(report.errors)
                @test isempty(report.warnings)
                @test report.generations == [1]
                @test report.pending_generations == Int[]
                @test isempty(report.omitted_values)

                # A generation that was started but never completed is reported.
                write_population!(store, rs_population(generation=2, particles=[[3.0], [4.0]],
                                                       weights=[0.5, 0.5],
                                                       distances=[0.3, 0.4]))
                pending_report = validate_store(store)
                @test pending_report.valid
                @test !isempty(pending_report.warnings)
                @test any(w -> occursin("generation 2", w), pending_report.warnings)

                # Omissions stay visible to validation.
                opaque_store = make_store(rs_metadata())
                try
                    write_population!(opaque_store, rs_population())
                    opaque_result = EvaluationResult(rs_context(), :success, RSOpaqueSummary(),
                                                      NamedTuple(); wall_time_s=0.1)
                    write_evaluation!(opaque_store, EvaluationRecord(
                        opaque_result; omit_unsupported_summary=true, omission_reason="opaque"))
                    finalize_generation!(opaque_store, 1)
                    opaque_report = validate_store(opaque_store)
                    @test opaque_report.valid
                    @test length(opaque_report.omitted_values) == 1
                    @test occursin("was omitted", opaque_report.omitted_values[1])
                finally
                    close_run!(opaque_store)
                end

                # An unsupported storage schema is an error, not a warning.
                odd_store = make_store(rs_metadata(
                    storage_schema_version=UInt64(99)))
                try
                    odd_report = validate_store(odd_store)
                    @test !odd_report.valid
                    @test any(e -> occursin("unsupported storage schema", e),
                              odd_report.errors)
                finally
                    close_run!(odd_store)
                end
            finally
                close_run!(store)
            end
        end

        @testset "separation from inference" begin
            # The run store records history; it never resumes inference. No
            # public symbol reconstructs sampler state from a store.
            exported = names(SequentialABC)
            @test !any(n -> occursin("resume", lowercase(String(n))) ||
                               occursin("load_state", lowercase(String(n))),
                       exported)

            # The store holds no configuration and no sampler state.
            store = make_store(rs_metadata())
            try
                @test !(store isa ABCSMCState)
                @test !hasproperty(store, :cfg)
            finally
                close_run!(store)
            end

            # Inference is untouched by the run-history contract: the same
            # seed still reproduces the same population.
            prior = Normal(0.0, 1.0)
            cfg = ABCSMCConfig(θ -> logpdf(prior, θ[1]),
                               rng -> [rand(rng, prior)],
                               identity,
                               (s, s_obs) -> abs(s - s_obs),
                               GaussianKernel(), [0.5;;],
                               ContinuousSimulator(model_builder = θ -> θ[1],
                                                   evolve = (state, rng) -> state + randn(rng),
                                                   summary = y -> y))
            a = run_smc(cfg, 0.5; N=12, T=2, base_seed=2026)
            b = run_smc(cfg, 0.5; N=12, T=2, base_seed=2026)
            @test a.θ == b.θ
            @test a.w == b.w
            @test a.dists == b.dists
            @test a.gen == b.gen == 2
        end
    end
    return nothing
end
