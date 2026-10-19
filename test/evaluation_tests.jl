using SequentialABC
using Test
using Random
using LinearAlgebra
using Distributions

"""
    test/evaluation_tests.jl

Focused tests for the generic simulator evaluation contract (src/evaluation.jl).

All assertions live inside named `@testset` blocks.  `@test_throws` is used only
for documented validation or propagation behavior.
"""

# ---------------------------------------------------------------------------
# Test fixtures and helpers (definitions only; no assertions here)
# ---------------------------------------------------------------------------

struct EvalContractSimulator{F} <: ABCSimulator
    f::F
end

SequentialABC.run_sim_once(sim::EvalContractSimulator, θ, rng) = sim.f(θ, rng)

struct EvalContractOpaqueSummary
    value::Float64
end

struct EvalContractCustomError <: Exception
    msg::String
end

Base.showerror(io::IO, e::EvalContractCustomError) = print(io, e.msg)

struct EvalContractScriptedPolicy <: FailurePolicy
    action::Symbol
end

(policy::EvalContractScriptedPolicy)(ctx, result) = policy.action

_eval_contract_ctx() = EvaluationContext(
    :smc, 3, 5, 2, 4, 6, UInt64(0x0000000000000001), UInt64(0x0000000000000042))

_eval_contract_failed(ctx::EvaluationContext) = EvaluationResult(
    ctx,
    :failed,
    nothing,
    NamedTuple();
    failure=EvaluationFailure(:simulator_error, "TestError", "boom", nothing),
    wall_time_s=0.5,
)

function _eval_contract_cfg()
    prior_logpdf(θ) = sum(logpdf(Normal(0.0, 4.0), x) for x in θ)
    prior_sample(rng) = 4.0 .* randn(rng, 2)
    distance(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
    simulator = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ randn(rng, 2),
        summary = identity,
    )
    return ABCSMCConfig(
        prior_logpdf,
        prior_sample,
        identity,
        distance,
        GaussianKernel(),
        Matrix{Float64}(I, 2, 2),
        simulator,
    )
end

const _EVAL_CONTRACT_OBS = [1.0, -1.0]
const _EVAL_CONTRACT_SEED = 20260118

# ---------------------------------------------------------------------------
# 1. EvaluationContext valid construction
# ---------------------------------------------------------------------------

@testset "EvaluationContext valid construction" begin
    ctx = _eval_contract_ctx()

    @test ctx.method === :smc
    @test ctx.generation == 3
    @test ctx.particle == 5
    @test ctx.proposal == 2
    @test ctx.attempt == 4
    @test ctx.replicate == 6
    @test ctx.seed == UInt64(0x0000000000000001)
    @test ctx.base_seed == UInt64(0x0000000000000042)

    # Persistent identity indices are fixed-width, not platform-sized.
    @test fieldtype(EvaluationContext, :generation) === Int64
    @test fieldtype(EvaluationContext, :particle) === Int64
    @test fieldtype(EvaluationContext, :proposal) === Int64
    @test fieldtype(EvaluationContext, :attempt) === Int64
    @test fieldtype(EvaluationContext, :replicate) === Int64
    @test fieldtype(EvaluationContext, :seed) === UInt64
    @test fieldtype(EvaluationContext, :base_seed) === UInt64

    # Convenience construction derives the public schema seed.
    derived = EvaluationContext(:smc, 3, 5, 2, 4, 6, UInt64(42))
    @test derived.seed == evaluation_seed(:smc, 3, 5, 2, 4, 6, UInt64(42))
    @test derived.base_seed == UInt64(42)

    # Maximum values are valid, not sentinels.
    max_ctx = EvaluationContext(
        :smc, 1, 1, 1, 1, 1, typemax(UInt64), typemax(UInt64))
    @test max_ctx.seed == typemax(UInt64)
    @test max_ctx.base_seed == typemax(UInt64)
end

# ---------------------------------------------------------------------------
# 2. EvaluationContext invalid identity fields
# ---------------------------------------------------------------------------

@testset "EvaluationContext invalid identity fields" begin
    @test_throws ArgumentError EvaluationContext(
        Symbol(""), 1, 1, 1, 1, 1, UInt64(1), UInt64(42))
    @test_throws MethodError EvaluationContext(
        "smc", 1, 1, 1, 1, 1, UInt64(1), UInt64(42))

    for bad in (0, -1, Int128(typemax(Int64)) + 1)
        @test_throws ArgumentError EvaluationContext(
            :smc, bad, 1, 1, 1, 1, UInt64(1), UInt64(42))
        @test_throws ArgumentError EvaluationContext(
            :smc, 1, bad, 1, 1, 1, UInt64(1), UInt64(42))
        @test_throws ArgumentError EvaluationContext(
            :smc, 1, 1, bad, 1, 1, UInt64(1), UInt64(42))
        @test_throws ArgumentError EvaluationContext(
            :smc, 1, 1, 1, bad, 1, UInt64(1), UInt64(42))
        @test_throws ArgumentError EvaluationContext(
            :smc, 1, 1, 1, 1, bad, UInt64(1), UInt64(42))
    end

    # The convenience constructor validates before deriving a seed.
    @test_throws ArgumentError EvaluationContext(:smc, 0, 1, 1, 1, 1, UInt64(42))
    @test_throws ArgumentError EvaluationContext(
        Symbol(""), 1, 1, 1, 1, 1, UInt64(42))

    # Public seed derivation performs the same validation.
    @test_throws ArgumentError evaluation_seed(
        Symbol(""), 1, 1, 1, 1, 1, UInt64(42))
    for bad in (0, -1, Int128(typemax(Int64)) + 1)
        @test_throws ArgumentError evaluation_seed(
            :smc, bad, 1, 1, 1, 1, UInt64(42))
        @test_throws ArgumentError evaluation_seed(
            :smc, 1, bad, 1, 1, 1, UInt64(42))
        @test_throws ArgumentError evaluation_seed(
            :smc, 1, 1, bad, 1, 1, UInt64(42))
        @test_throws ArgumentError evaluation_seed(
            :smc, 1, 1, 1, bad, 1, UInt64(42))
        @test_throws ArgumentError evaluation_seed(
            :smc, 1, 1, 1, 1, bad, UInt64(42))
    end
end

# ---------------------------------------------------------------------------
# 3. Zero-seed behavior
# ---------------------------------------------------------------------------

@testset "Zero-seed behavior" begin
    # UInt64(0) is a valid finite integer; there is no NaN/Inf and no sentinel.
    zero_ctx = EvaluationContext(:smc, 1, 1, 1, 1, 1, UInt64(0), UInt64(0))
    @test zero_ctx.seed == UInt64(0)
    @test zero_ctx.base_seed == UInt64(0)
    @test zero_ctx.seed isa UInt64
    @test zero_ctx.base_seed isa UInt64

    derived = EvaluationContext(:smc, 1, 1, 1, 1, 1, UInt64(0))
    @test derived.seed isa UInt64
    @test derived.seed == evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(0))

    # Zero base seeds participate in the public schema like any other value.
    @test evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(0)) !=
          evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(1))

    # An explicit zero seed can be reproduced through the primitive form.
    rebuilt = EvaluationContext(context_primitive(zero_ctx))
    @test rebuilt == zero_ctx
end

# ---------------------------------------------------------------------------
# 4. Legacy seed compatibility
# ---------------------------------------------------------------------------

@testset "Legacy seed compatibility" begin
    # make_particle_seed is the frozen legacy schema used by SMC/PMC/MCMC.
    points = (
        (1, 1, 1, 1, 42),
        (1, 1, 1, 0, 42),
        (2, 10, 3, 1, 2026),
        (5, 100, 1, 2, 7),
        (1, 0, 0, -1, 42),
    )
    for (gen, i, attempt, m, base) in points
        @test make_particle_seed(gen, i, attempt, m, base) ==
              hash((gen, i, attempt, m, base))
    end

    @test typeof(make_particle_seed(1, 1, 1, 1, 42)) === UInt64
    @test make_particle_seed(1, 1, 1, 1, 42) ==
          make_particle_seed(1, 1, 1, 1, 42)

    # The legacy stream is named explicitly and is not the public schema.
    @test SequentialABC.LEGACY_PARTICLE_SEED_SCHEMA_NAME ==
          "make_particle_seed_legacy_hash"
    @test SequentialABC.LEGACY_PARTICLE_SEED_SCHEMA_NAME !=
          SequentialABC.EVALUATION_SEED_SCHEMA_NAME
end

# ---------------------------------------------------------------------------
# 5. Public seed schema v1
# ---------------------------------------------------------------------------

@testset "Public seed schema v1" begin
    base = evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(42))

    @test base isa UInt64
    @test base == evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(42))
    @test base == UInt64(0x2d6a3fca9dc51067)
    @test evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(0)) ==
          UInt64(0x05d36f500b137d1f)
    @test evaluation_seed(Symbol("méthod"), 1, 1, 1, 1, 1, UInt64(42)) ==
          UInt64(0xbf19748a817b69f6)
    @test SequentialABC.EVALUATION_SEED_SCHEMA_VERSION == UInt64(1)
    @test SequentialABC.EVALUATION_SEED_SCHEMA_NAME == "evaluation_seed_v1"

    # Every identity field participates in the public derivation.
    variants = (
        (:pmc, 1, 1, 1, 1, 1, UInt64(42)),
        (:smc, 2, 1, 1, 1, 1, UInt64(42)),
        (:smc, 1, 2, 1, 1, 1, UInt64(42)),
        (:smc, 1, 1, 2, 1, 1, UInt64(42)),
        (:smc, 1, 1, 1, 2, 1, UInt64(42)),
        (:smc, 1, 1, 1, 1, 2, UInt64(42)),
        (:smc, 1, 1, 1, 1, 1, UInt64(43)),
    )
    for args in variants
        @test evaluation_seed(args...) != base
    end

    # Method bytes are length-prefixed and encoded as UTF-8, not via hash().
    @test evaluation_seed(Symbol("smc2"), 1, 1, 1, 1, 1, UInt64(42)) != base
    @test evaluation_seed(Symbol("méthod"), 1, 1, 1, 1, 1, UInt64(42)) isa UInt64

    # The convenience constructor agrees with the selected public schema.
    ctx = EvaluationContext(:smc, 2, 3, 1, 4, 2, UInt64(42))
    @test ctx.seed == evaluation_seed(:smc, 2, 3, 1, 4, 2, UInt64(42))
end

# ---------------------------------------------------------------------------
# 6. Identity-field separation
# ---------------------------------------------------------------------------

@testset "Identity-field separation" begin
    base = evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(42))
    seen = Set{UInt64}((base,))

    for method in (:pmc, :mcmc)
        seed = evaluation_seed(method, 1, 1, 1, 1, 1, UInt64(42))
        @test seed != base
        @test !(seed in seen)
        push!(seen, seed)
    end

    for index in 1:5
        args = Any[:smc, 1, 1, 1, 1, 1, UInt64(42)]
        args[index + 1] = 2
        seed = evaluation_seed(args...)
        @test seed != base
        @test !(seed in seen)
        push!(seen, seed)
    end

    seed = evaluation_seed(:smc, 1, 1, 1, 1, 1, UInt64(43))
    @test seed != base
    @test !(seed in seen)
    push!(seen, seed)
end

# ---------------------------------------------------------------------------
# 7. Bounded collision scan
# ---------------------------------------------------------------------------

@testset "Bounded collision scan" begin
    methods = (:smc, :pmc, :mcmc)
    bases = (UInt64(0), UInt64(1), UInt64(42))
    seeds = UInt64[]
    for method in methods,
        gen in 1:4,
        particle in 1:4,
        proposal in 1:2,
        attempt in 1:2,
        replicate in 1:2,
        base_seed in bases
        push!(seeds, evaluation_seed(
            method, gen, particle, proposal, attempt, replicate, base_seed))
    end

    expected = length(methods) * 4 * 4 * 2 * 2 * 2 * length(bases)
    @test expected == 1152
    @test length(seeds) == expected
    @test length(unique(seeds)) == expected
end

# ---------------------------------------------------------------------------
# 8. Primitive representation and round trip
# ---------------------------------------------------------------------------

@testset "Primitive representation and round trip" begin
    ctx = EvaluationContext(:smc, 3, 5, 2, 4, 6, UInt64(42))
    prim = context_primitive(ctx)

    @test prim.method == "smc"
    @test prim.generation == 3
    @test prim.particle == 5
    @test prim.proposal == 2
    @test prim.attempt == 4
    @test prim.replicate == 6
    @test prim.seed == ctx.seed
    @test prim.base_seed == UInt64(42)
    @test prim.seed_schema == SequentialABC.EVALUATION_SEED_SCHEMA_VERSION
    @test prim.seed_schema_name == SequentialABC.EVALUATION_SEED_SCHEMA_NAME

    # Fixed-width primitive fields avoid host word-size ambiguity.
    @test typeof(prim.generation) === Int64
    @test typeof(prim.particle) === Int64
    @test typeof(prim.proposal) === Int64
    @test typeof(prim.attempt) === Int64
    @test typeof(prim.replicate) === Int64
    @test typeof(prim.seed) === UInt64
    @test typeof(prim.base_seed) === UInt64
    @test prim.method isa String

    rebuilt = EvaluationContext(prim)
    @test rebuilt == ctx
    @test rebuilt.seed == ctx.seed

    # Unknown, missing, or mismatched schemas are rejected.
    @test_throws ArgumentError EvaluationContext(
        merge(prim, (seed_schema = UInt64(99),)))
    @test_throws ArgumentError EvaluationContext(
        merge(prim, (seed_schema_name = "other_schema",)))

    no_schema = (
        method = "smc",
        generation = 1,
        particle = 1,
        proposal = 1,
        attempt = 1,
        replicate = 1,
        seed = UInt64(1),
        base_seed = UInt64(42),
    )
    @test_throws ArgumentError EvaluationContext(no_schema)

    # Explicit seeds (e.g. reproducing a legacy stream) round-trip unchanged.
    explicit = EvaluationContext(
        :smc, 1, 1, 1, 1, 1, UInt64(12345), UInt64(42))
    @test explicit.seed == UInt64(12345)
    @test EvaluationContext(context_primitive(explicit)) == explicit
end

# ---------------------------------------------------------------------------
# 9. Equality semantics
# ---------------------------------------------------------------------------

@testset "Equality semantics" begin
    ctx = _eval_contract_ctx()
    @test ctx == _eval_contract_ctx()
    @test ctx != EvaluationContext(:pmc, 3, 5, 2, 4, 6,
        UInt64(1), UInt64(42))
    @test ctx != EvaluationContext(:smc, 4, 5, 2, 4, 6,
        UInt64(1), UInt64(42))
    @test ctx != EvaluationContext(:smc, 3, 6, 2, 4, 6,
        UInt64(1), UInt64(42))
    @test ctx != EvaluationContext(:smc, 3, 5, 3, 4, 6,
        UInt64(1), UInt64(42))
    @test ctx != EvaluationContext(:smc, 3, 5, 2, 5, 6,
        UInt64(1), UInt64(42))
    @test ctx != EvaluationContext(:smc, 3, 5, 2, 4, 7,
        UInt64(1), UInt64(42))
    @test ctx != EvaluationContext(:smc, 3, 5, 2, 4, 6,
        UInt64(2), UInt64(42))
    @test ctx != EvaluationContext(:smc, 3, 5, 2, 4, 6,
        UInt64(1), UInt64(43))

    failure = EvaluationFailure(:simulator_error, "E", "m", nothing)
    a = EvaluationResult(ctx, :failed, nothing, (tag = :a,);
        failure=failure, wall_time_s=0.5)
    b = EvaluationResult(ctx, :failed, nothing, (tag = :a,);
        failure=failure, wall_time_s=0.5)
    c = EvaluationResult(ctx, :failed, nothing, (tag = :b,);
        failure=failure, wall_time_s=0.5)
    @test a == b
    @test a != c
    @test a != EvaluationResult(ctx, :failed, nothing, (tag = :a,);
        failure=failure, wall_time_s=0.6)
end

# ---------------------------------------------------------------------------
# 10. EvaluationFailure validation
# ---------------------------------------------------------------------------

@testset "EvaluationFailure validation" begin
    for category in SequentialABC.EVALUATION_FAILURE_CATEGORIES
        failure = EvaluationFailure(category, "SomeError", "message", nothing)
        @test failure.category === category
        @test failure.exception_type == "SomeError"
        @test failure.message == "message"
        @test failure.stacktrace === nothing
    end

    @test_throws ArgumentError EvaluationFailure(
        :unknown_category, "SomeError", "message", nothing)

    symbol_failure = EvaluationFailure(
        :timeout, :TimeoutError, "timed out"; stacktrace = "stack")
    @test symbol_failure.exception_type isa String
    @test symbol_failure.exception_type == "TimeoutError"
    @test symbol_failure.stacktrace == "stack"

    string_failure = EvaluationFailure(
        :user_defined, "CustomError", "custom message";
        stacktrace = "trace text")
    @test string_failure.exception_type == "CustomError"
    @test string_failure.stacktrace == "trace text"

    # No arbitrary Exception object can be persisted.
    @test_throws MethodError EvaluationFailure(
        :simulator_error, ErrorException("boom"), "message")
    @test !(:exception in fieldnames(EvaluationFailure))
    @test fieldtype(EvaluationFailure, :exception_type) === String
    @test fieldtype(EvaluationFailure, :message) === String
    @test fieldtype(EvaluationFailure, :stacktrace) === Union{Nothing,String}

    # Equality is structural.
    a = EvaluationFailure(:simulator_error, "E", "m", nothing)
    b = EvaluationFailure(:simulator_error, "E", "m", nothing)
    @test a == b
    @test a != EvaluationFailure(:timeout, "E", "m", nothing)
end

# ---------------------------------------------------------------------------
# 11. EvaluationResult success invariants
# ---------------------------------------------------------------------------

@testset "EvaluationResult success invariants" begin
    ctx = _eval_contract_ctx()
    result = EvaluationResult(ctx, :success, 1.5, NamedTuple(); wall_time_s=0.5)
    @test result.status === :success
    @test result.summary == 1.5
    @test result.failure === nothing
    @test isfinite(result.wall_time_s)
    @test result.wall_time_s >= 0.0
    @test result.context === ctx

    failure = EvaluationFailure(:simulator_error, "E", "m", nothing)
    @test_throws ArgumentError EvaluationResult(
        ctx, :success, nothing, NamedTuple(); wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :success, [1.0], NamedTuple();
        failure=failure, wall_time_s=0.1)

    @test_throws ArgumentError EvaluationResult(
        ctx, :success, NaN, NamedTuple(); wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :success, [1.0, Inf], NamedTuple(); wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :success, (x = [1.0, NaN],), NamedTuple(); wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :success, missing, NamedTuple(); wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :success, [1.0, missing], NamedTuple(); wall_time_s=0.1)

    @test_throws ArgumentError EvaluationResult(
        ctx, :bogus, [1.0], NamedTuple();
        failure=failure, wall_time_s=0.1)
end

# ---------------------------------------------------------------------------
# 12. EvaluationResult failure invariants
# ---------------------------------------------------------------------------

@testset "EvaluationResult failure invariants" begin
    ctx = _eval_contract_ctx()
    failure = EvaluationFailure(:simulator_error, "E", "m", nothing)

    failed = EvaluationResult(
        ctx, :failed, nothing, NamedTuple(); failure=failure, wall_time_s=0.5)
    @test failed.status === :failed
    @test failed.summary === nothing
    @test failed.failure === failure

    nonfinite = EvaluationResult(
        ctx, :nonfinite, nothing, NamedTuple();
        failure=EvaluationFailure(:nonfinite_summary, "N", "n", nothing),
        wall_time_s=0.1)
    @test nonfinite.status === :nonfinite
    @test nonfinite.summary === nothing
    @test nonfinite.failure !== nothing

    timeout = EvaluationResult(
        ctx, :timeout, nothing, NamedTuple();
        failure=EvaluationFailure(:timeout, "T", "t", nothing),
        wall_time_s=30.0)
    @test timeout.status === :timeout
    @test timeout.summary === nothing
    @test timeout.failure !== nothing

    @test_throws ArgumentError EvaluationResult(
        ctx, :failed, [1.0], NamedTuple();
        failure=failure, wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :timeout, 1.0, NamedTuple();
        failure=failure, wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :failed, nothing, NamedTuple(); wall_time_s=0.1)
    @test_throws ArgumentError EvaluationResult(
        ctx, :nonfinite, nothing, NamedTuple(); wall_time_s=0.1)
end

# ---------------------------------------------------------------------------
# 13. Wall-time validation
# ---------------------------------------------------------------------------

@testset "Wall-time validation" begin
    ctx = _eval_contract_ctx()
    failure = EvaluationFailure(:simulator_error, "E", "m", nothing)

    for bad_time in (-1.0, NaN, Inf, -Inf)
        @test_throws ArgumentError EvaluationResult(
            ctx, :success, 1.0, NamedTuple(); wall_time_s=bad_time)
        @test_throws ArgumentError EvaluationResult(
            ctx, :failed, nothing, NamedTuple();
            failure=failure, wall_time_s=bad_time)
    end

    zero = EvaluationResult(ctx, :success, 1.0, NamedTuple(); wall_time_s=0.0)
    @test zero.wall_time_s == 0.0
    positive = EvaluationResult(
        ctx, :success, 1.0, NamedTuple(); wall_time_s=1.25)
    @test isfinite(positive.wall_time_s)
    @test positive.wall_time_s == 1.25
end

# ---------------------------------------------------------------------------
# 14. Recursive finiteness
# ---------------------------------------------------------------------------

@testset "Recursive finiteness" begin
    finite = SequentialABC._is_finite_summary
    @test finite(1.0)
    @test finite(0)
    @test finite(1 // 2)
    @test !finite(NaN)
    @test !finite(Inf)
    @test !finite(-Inf)

    @test finite([1.0, 2.0])
    @test finite([1.0, [2.0, 3.0]])
    @test !finite([1.0, [2.0, NaN]])
    @test finite(zeros(2, 2))
    @test !finite([1.0 NaN; 2.0 3.0])
    @test finite(Float64[])
    @test finite((1.0, 2.0))
    @test finite((1.0, (2.0, [3.0])))
    @test !finite((1.0, (2.0, [Inf])))
    @test finite((mean = 1.0, nested = (sd = 2.0,)))
    @test !finite((mean = 1.0, nested = (sd = NaN,)))
    @test !finite(missing)
    @test !finite([1.0, missing])
    @test finite(nothing)
    @test finite((1.0, nothing))
end

# ---------------------------------------------------------------------------
# 15. Scalar, vector, tuple, and NamedTuple summaries
# ---------------------------------------------------------------------------

@testset "Scalar, vector, tuple, and NamedTuple summaries" begin
    ctx = _eval_contract_ctx()
    scalar = EvaluationResult(ctx, :success, 1.5, NamedTuple(); wall_time_s=0.5)
    vector = EvaluationResult(
        ctx, :success, [1.0, 2.0, 3.0], NamedTuple(); wall_time_s=0.5)
    tuple_summary = EvaluationResult(
        ctx, :success, (1.0, [2.0, 3.0]), NamedTuple(); wall_time_s=0.5)
    named = EvaluationResult(
        ctx, :success, (mean = 1.0, sd = 2.0), NamedTuple(); wall_time_s=0.5)

    @test scalar.summary == 1.5
    @test vector.summary == [1.0, 2.0, 3.0]
    @test tuple_summary.summary == (1.0, [2.0, 3.0])
    @test named.summary == (mean = 1.0, sd = 2.0)
    @test named.summary.mean == 1.0
    @test named.summary.sd == 2.0

    # The same flexibility holds through the adapter.
    for summary in (1.5, [1.0, 2.0], (1.0, 2.0), (mean = 1.0, sd = 2.0))
        sim = EvalContractSimulator((θ, rng) -> summary)
        result = evaluate(sim, [0.0], ctx)
        @test result.status === :success
        @test result.summary == summary
    end
end

# ---------------------------------------------------------------------------
# 16. Opaque-summary documented behavior
# ---------------------------------------------------------------------------

@testset "Opaque-summary documented behavior" begin
    ctx = _eval_contract_ctx()
    opaque = EvalContractOpaqueSummary(NaN)
    result = EvaluationResult(
        ctx, :success, opaque, NamedTuple(); wall_time_s=0.5)
    @test result.status === :success
    @test result.summary === opaque

    finite = SequentialABC._is_finite_summary
    @test finite(opaque)
    @test finite("not a number")
    @test finite(Dict(:a => NaN))

    # The limitation is explicit: arbitrary opaque objects are not introspected.
    sim = EvalContractSimulator((θ, rng) -> opaque)
    evaluated = evaluate(sim, [0.0], ctx)
    @test evaluated.status === :success
    @test evaluated.summary === opaque
end

# ---------------------------------------------------------------------------
# 17. RejectEvaluation
# ---------------------------------------------------------------------------

@testset "RejectEvaluation" begin
    ctx = _eval_contract_ctx()
    failed = _eval_contract_failed(ctx)
    policy = RejectEvaluation()

    @test policy(ctx, failed) === :reject
    action, next = retry_with_limit(policy, ctx, failed, 3)
    @test action === :reject
    @test next == ctx
    @test failed.summary === nothing
    @test failed.failure !== nothing
end

# ---------------------------------------------------------------------------
# 18. RetryEvaluation
# ---------------------------------------------------------------------------

@testset "RetryEvaluation" begin
    ctx = EvaluationContext(:smc, 3, 5, 2, 1, 6, UInt64(42))
    failed = _eval_contract_failed(ctx)
    policy = RetryEvaluation()

    @test policy(ctx, failed) === :retry
    action, next = retry_with_limit(policy, ctx, failed, 1)
    @test action === :retry
    @test next.attempt == ctx.attempt + 1
    @test next.seed == evaluation_seed(
        ctx.method, ctx.generation, ctx.particle, ctx.proposal,
        ctx.attempt + 1, ctx.replicate, ctx.base_seed)
    @test next.seed != ctx.seed
    @test next.method == ctx.method
    @test next.generation == ctx.generation
    @test next.particle == ctx.particle
    @test next.proposal == ctx.proposal
    @test next.replicate == ctx.replicate
    @test next.base_seed == ctx.base_seed
end

# ---------------------------------------------------------------------------
# 19. AbortInference
# ---------------------------------------------------------------------------

@testset "AbortInference" begin
    ctx = _eval_contract_ctx()
    failed = _eval_contract_failed(ctx)
    policy = AbortInference()

    @test policy(ctx, failed) === :abort
    action, next = retry_with_limit(policy, ctx, failed, 3)
    @test action === :abort
    @test next == ctx
    @test failed.failure === failed.failure
end

# ---------------------------------------------------------------------------
# 20. FunctionFailurePolicy
# ---------------------------------------------------------------------------

@testset "FunctionFailurePolicy" begin
    ctx = _eval_contract_ctx()
    failed = _eval_contract_failed(ctx)

    seen = Ref{Any}(nothing)
    policy = FunctionFailurePolicy((c, r) -> (seen[] = (c, r); :abort))
    @test policy(ctx, failed) === :abort
    @test seen[][1] === ctx
    @test seen[][2] === failed

    retry = FunctionFailurePolicy((c, r) -> :retry)
    @test retry(ctx, failed) === :retry
    reject = FunctionFailurePolicy((c, r) -> :reject)
    @test reject(ctx, failed) === :reject
end

# ---------------------------------------------------------------------------
# 21. Invalid policy action
# ---------------------------------------------------------------------------

@testset "Invalid policy action" begin
    ctx = _eval_contract_ctx()
    failed = _eval_contract_failed(ctx)

    invalid_policy = FunctionFailurePolicy((c, r) -> :bogus)
    @test_throws ArgumentError invalid_policy(ctx, failed)
    @test_throws ArgumentError retry_with_limit(invalid_policy, ctx, failed, 3)

    scripted = EvalContractScriptedPolicy(:bogus)
    @test_throws ArgumentError retry_with_limit(scripted, ctx, failed, 3)
    @test_throws ArgumentError retry_with_limit(
        RetryEvaluation(), ctx, failed, -1)
end

# ---------------------------------------------------------------------------
# 22. Retry exhaustion and seed progression
# ---------------------------------------------------------------------------

@testset "Retry exhaustion and seed progression" begin
    ctx = EvaluationContext(:smc, 3, 5, 2, 1, 6, UInt64(42))
    failure = EvaluationFailure(:simulator_error, "TestError", "boom", nothing)
    failed = EvaluationResult(
        ctx, :failed, nothing, NamedTuple(); failure=failure, wall_time_s=0.5)

    # max_retries = 0: no retry budget, context unchanged.
    action, next = retry_with_limit(RetryEvaluation(), ctx, failed, 0)
    @test action === :abort
    @test next == ctx
    @test next.seed == ctx.seed
    @test next.attempt == ctx.attempt

    # One retry moves attempt forward and re-derives the seed.
    action, next = retry_with_limit(RetryEvaluation(), ctx, failed, 1)
    @test action === :retry
    @test next.attempt == ctx.attempt + 1
    @test next.seed == evaluation_seed(
        :smc, 3, 5, 2, next.attempt, 6, UInt64(42))
    @test next.seed != ctx.seed

    # Budget is total, not per call: at attempt 5 with max_retries = 1 the
    # budget is exhausted and the context is returned unchanged.
    action, exhausted = retry_with_limit(RetryEvaluation(), next, failed, 1)
    @test action === :abort
    @test exhausted == next
    @test exhausted.attempt == next.attempt
    @test exhausted.seed == next.seed

    # Full progression for max_retries = 3: attempts 4 -> 7, then abort.
    current = ctx
    for expected_attempt in (ctx.attempt + 1):(ctx.attempt + 3)
        action, retried = retry_with_limit(
            RetryEvaluation(), current, failed, 3)
        @test action === :retry
        @test retried.attempt == expected_attempt
        @test retried.seed == evaluation_seed(
            :smc, 3, 5, 2, expected_attempt, 6, UInt64(42))
        @test retried.seed != current.seed
        current = retried
    end
    action, final_ctx = retry_with_limit(
        RetryEvaluation(), current, failed, 3)
    @test action === :abort
    @test final_ctx == current
    @test final_ctx.attempt == ctx.attempt + 3

    # Exactly one policy call per invocation; no internal simulation loop.
    calls = Ref(0)
    counting_policy = FunctionFailurePolicy((c, r) -> (calls[] += 1; :retry))
    action, _ = retry_with_limit(counting_policy, ctx, failed, 3)
    @test action === :retry
    @test calls[] == 1

    # The original structured failure is owned by the caller and unchanged.
    @test failed.failure === failure
    @test failed.failure.category === :simulator_error
    @test failed.failure.message == "boom"
    @test failed.summary === nothing
end

# ---------------------------------------------------------------------------
# 23. evaluate success
# ---------------------------------------------------------------------------

@testset "evaluate success" begin
    sim = EvalContractSimulator(
        (θ, rng) -> θ .+ randn(rng, length(θ)))
    ctx = EvaluationContext(:smc, 1, 1, 1, 1, 1, UInt64(42))

    result = evaluate(sim, [1.0, 2.0], ctx)
    @test result.status === :success
    @test result.context === ctx
    @test result.summary !== nothing
    @test length(result.summary) == 2
    @test all(isfinite, result.summary)
    @test result.failure === nothing
    @test isfinite(result.wall_time_s)
    @test result.wall_time_s >= 0.0

    replay = evaluate(sim, [1.0, 2.0], ctx)
    @test replay.summary == result.summary
    @test replay.status === :success
    @test replay.context == result.context

    named_sim = EvalContractSimulator(
        (θ, rng) -> (mean = θ[1], sd = abs(θ[2])))
    named = evaluate(named_sim, [-1.0, 2.0], ctx)
    @test named.status === :success
    @test named.summary == (mean = -1.0, sd = 2.0)

    # The result type has no distance field; discrepancy calculation remains
    # separate from simulator evaluation.
    @test !(:distance in fieldnames(EvaluationResult))
end

# ---------------------------------------------------------------------------
# 24. Provenance preservation and default provenance
# ---------------------------------------------------------------------------

@testset "Provenance preservation and default provenance" begin
    ctx = _eval_contract_ctx()
    sim = EvalContractSimulator((θ, rng) -> θ .* 2.0)

    provenance = (run_id = 1, trial = "A")
    result = evaluate(sim, [0.5], ctx; provenance = provenance)
    @test result.status === :success
    @test result.provenance == provenance
    @test result.provenance.run_id == 1
    @test result.provenance.trial == "A"

    default_result = evaluate(sim, [0.5], ctx)
    @test length(default_result.provenance) == 0
    @test default_result.provenance == NamedTuple()

    failing_provenance = (run_id = 2, reason = "exception")
    failed = evaluate(
        EvalContractSimulator((θ, rng) -> error("boom")), [0.5], ctx;
        provenance = failing_provenance)
    @test failed.status === :failed
    @test failed.provenance == failing_provenance
end

# ---------------------------------------------------------------------------
# 25. Simulator exception
# ---------------------------------------------------------------------------

@testset "Simulator exception" begin
    ctx = _eval_contract_ctx()

    error_sim = EvalContractSimulator(
        (θ, rng) -> error("simulator crashed"))
    result = evaluate(error_sim, [1.0], ctx)
    @test result.status === :failed
    @test result.summary === nothing
    @test result.failure !== nothing
    @test result.failure.category === :simulator_error
    @test result.failure.exception_type == "ErrorException"
    @test result.failure.exception_type != "typename(ErrorException)"
    @test occursin("simulator crashed", result.failure.message)
    @test result.failure.stacktrace isa String
    @test occursin("Stacktrace", result.failure.stacktrace)

    custom_sim = EvalContractSimulator(
        (θ, rng) -> throw(EvalContractCustomError("custom boom")))
    custom = evaluate(custom_sim, [1.0], ctx)
    @test custom.status === :failed
    @test custom.summary === nothing
    @test custom.failure.category === :simulator_error
    @test custom.failure.exception_type == "EvalContractCustomError"
    @test custom.failure.message == "custom boom"

    # Stack traces are optional and can be disabled for privacy.
    no_trace = evaluate(error_sim, [1.0], ctx; capture_stacktrace=false)
    @test no_trace.status === :failed
    @test no_trace.failure !== nothing
    @test no_trace.failure.stacktrace === nothing

    # A malformed `nothing` summary is structured failure, not success.
    nothing_sim = EvalContractSimulator((θ, rng) -> nothing)
    malformed = evaluate(nothing_sim, [1.0], ctx)
    @test malformed.status === :failed
    @test malformed.summary === nothing
    @test malformed.failure !== nothing
    @test malformed.failure.category === :simulator_error
    @test malformed.failure.exception_type == "MalformedSummary"

    # Interrupts are process control, not scientific outcomes.
    interrupt_sim = EvalContractSimulator((θ, rng) -> throw(InterruptException()))
    @test_throws InterruptException evaluate(interrupt_sim, [1.0], ctx)
end

# ---------------------------------------------------------------------------
# 26. Nonfinite simulator output
# ---------------------------------------------------------------------------

@testset "Nonfinite simulator output" begin
    ctx = _eval_contract_ctx()
    bad_summaries = (
        NaN,
        Inf,
        -Inf,
        [1.0, NaN],
        (1.0, Inf),
        (x = [NaN],),
        missing,
        [1.0, missing],
    )

    for bad in bad_summaries
        sim = EvalContractSimulator((θ, rng) -> bad)
        result = evaluate(sim, [0.0], ctx)
        @test result.status === :nonfinite
        @test result.summary === nothing
        @test result.failure !== nothing
        @test result.failure.category === :nonfinite_summary
        @test result.failure.exception_type == "NonfiniteSummary"
        @test isfinite(result.wall_time_s)
        @test result.wall_time_s >= 0.0
    end
end

# ---------------------------------------------------------------------------
# 27. Timeout representation where applicable
# ---------------------------------------------------------------------------

@testset "Timeout representation where applicable" begin
    ctx = _eval_contract_ctx()
    timeout_failure = EvaluationFailure(
        :timeout, "TimeoutError", "simulation exceeded deadline", nothing)
    timeout = EvaluationResult(
        ctx, :timeout, nothing, (source = :transport,);
        failure=timeout_failure, wall_time_s=30.0)

    @test timeout.status === :timeout
    @test timeout.summary === nothing
    @test timeout.failure === timeout_failure
    @test timeout.failure.category === :timeout
    @test timeout.provenance.source === :transport
    @test timeout.wall_time_s == 30.0

    # The adapter has no wall-clock interruption mechanism; the timeout is
    # represented by the result type and supplied by an outer transport layer.
    @test !(:timeout_s in fieldnames(EvaluationResult))
end

# ---------------------------------------------------------------------------
# 28. Failure never becomes a numeric discrepancy
# ---------------------------------------------------------------------------

@testset "Failure never becomes a numeric discrepancy" begin
    ctx = _eval_contract_ctx()
    distance_calls = Ref(0)
    distance = (s, s_obs) -> (distance_calls[] += 1; abs(s - s_obs))

    error_result = evaluate(
        EvalContractSimulator((θ, rng) -> error("boom")), [0.0], ctx)
    nonfinite_result = evaluate(
        EvalContractSimulator((θ, rng) -> NaN), [0.0], ctx)

    @test error_result.status === :failed
    @test error_result.summary === nothing
    @test error_result.failure !== nothing
    @test nonfinite_result.status === :nonfinite
    @test nonfinite_result.summary === nothing
    @test nonfinite_result.failure !== nothing
    @test distance_calls[] == 0
    @test !(:distance in fieldnames(EvaluationResult))

    # A large scientific discrepancy is different from a software failure.
    far = evaluate(
        EvalContractSimulator((θ, rng) -> fill(1.0e9, length(θ))),
        [0.0], ctx)
    @test far.status === :success
    @test far.failure === nothing
    @test far.summary == [1.0e9]
    @test distance(far.summary[1], 0.0) == 1.0e9
    @test distance_calls[] == 1
end

# ---------------------------------------------------------------------------
# 29. Legacy run_sim_once compatibility
# ---------------------------------------------------------------------------

@testset "Legacy run_sim_once compatibility" begin
    sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ randn(rng, length(θ)),
        summary = identity,
    )
    cfg = ABCSMCConfig(
        θ -> -sum(abs2, θ),
        rng -> randn(rng, 1),
        identity,
        (s, s_obs) -> sqrt(sum(abs2, s .- s_obs)),
        GaussianKernel(),
        Matrix{Float64}(I, 1, 1),
        sim,
    )

    legacy = SequentialABC.run_sim_once(
        cfg, 1, 1, 1, [0.5]; M = 1, base_seed = 42)
    @test legacy isa Vector
    @test length(legacy) == 1
    @test all(isfinite, legacy[1])

    expected_seed = make_particle_seed(1, 1, 1, 1, 42)
    expected = SequentialABC.run_sim_once(
        sim, cfg.transform([0.5]), MersenneTwister(expected_seed))
    @test legacy[1] == expected

    multi = SequentialABC.run_sim_once(
        cfg, 1, 1, 1, [0.5]; M = 3, base_seed = 42)
    @test length(multi) == 3
    for m in 1:3
        expected_m = SequentialABC.run_sim_once(
            sim, cfg.transform([0.5]),
            MersenneTwister(make_particle_seed(1, 1, 1, m, 42)))
        @test multi[m] == expected_m
    end
end

# ---------------------------------------------------------------------------
# 30. Direct legacy-versus-adapter deterministic equivalence
# ---------------------------------------------------------------------------

@testset "Direct legacy-versus-adapter deterministic equivalence" begin
    sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ randn(rng, length(θ)),
        summary = identity,
    )
    cfg = ABCSMCConfig(
        θ -> -sum(abs2, θ),
        rng -> randn(rng, 1),
        identity,
        (s, s_obs) -> sqrt(sum(abs2, s .- s_obs)),
        GaussianKernel(),
        Matrix{Float64}(I, 1, 1),
        sim,
    )

    θ = cfg.transform([0.5])
    legacy_seed = make_particle_seed(1, 1, 1, 1, 42)
    legacy_summary = SequentialABC.run_sim_once(
        cfg, 1, 1, 1, [0.5]; M = 1, base_seed = 42)[1]

    legacy_context = EvaluationContext(
        :smc, 1, 1, 1, 1, 1, legacy_seed, UInt64(42))
    @test legacy_context.seed == legacy_seed
    @test context_primitive(legacy_context).seed == legacy_seed

    adapter = evaluate(sim, θ, legacy_context)
    @test adapter.status === :success
    @test adapter.summary == legacy_summary
    @test adapter.failure === nothing

    public_context = EvaluationContext(:smc, 1, 1, 1, 1, 1, UInt64(42))
    @test public_context.seed == evaluation_seed(
        :smc, 1, 1, 1, 1, 1, UInt64(42))
    @test context_primitive(public_context).seed_schema ==
          SequentialABC.EVALUATION_SEED_SCHEMA_VERSION
    @test public_context.seed != legacy_seed
end

# ---------------------------------------------------------------------------
# 31. Existing inference behavior remains unchanged
# ---------------------------------------------------------------------------

@testset "Existing inference behavior remains unchanged" begin
    cfg = _eval_contract_cfg()

    smc_a = run_smc(cfg, _EVAL_CONTRACT_OBS;
        N=10, T=2, ε0=6.0, base_seed=_EVAL_CONTRACT_SEED)
    smc_b = run_smc(cfg, _EVAL_CONTRACT_OBS;
        N=10, T=2, ε0=6.0, base_seed=_EVAL_CONTRACT_SEED)
    smc_other = run_smc(cfg, _EVAL_CONTRACT_OBS;
        N=10, T=2, ε0=6.0, base_seed=_EVAL_CONTRACT_SEED + 1)

    @test smc_a.θ == smc_b.θ
    @test smc_a.w == smc_b.w
    @test smc_a.θ != smc_other.θ

    via_dispatch = run_inference(:smc, cfg, _EVAL_CONTRACT_OBS;
        N=10, T=2, ε0=6.0, base_seed=_EVAL_CONTRACT_SEED)
    @test via_dispatch.θ == smc_a.θ
    @test via_dispatch.w == smc_a.w

    mcmc_a = run_mcmc(cfg, _EVAL_CONTRACT_OBS;
        n_samples=20, ε=6.0, base_seed=_EVAL_CONTRACT_SEED)
    mcmc_b = run_mcmc(cfg, _EVAL_CONTRACT_OBS;
        n_samples=20, ε=6.0, base_seed=_EVAL_CONTRACT_SEED)
    @test mcmc_a.state.θ == mcmc_b.state.θ
    @test mcmc_a.accepted == mcmc_b.accepted

    pmc_a = run_pmc(cfg, _EVAL_CONTRACT_OBS;
        N=10, T=2, εs=[6.0, 4.0], base_seed=_EVAL_CONTRACT_SEED)
    pmc_b = run_pmc(cfg, _EVAL_CONTRACT_OBS;
        N=10, T=2, εs=[6.0, 4.0], base_seed=_EVAL_CONTRACT_SEED)
    @test pmc_a.θ == pmc_b.θ
    @test pmc_a.w == pmc_b.w
end
