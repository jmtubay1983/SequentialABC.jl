"""
    EvaluationContext

Fixed-width, serializable identity of one simulator evaluation.

Fields:
  - `method::Symbol`          — nonempty engine symbol (e.g. `:smc`, `:pmc`,
    `:mcmc`)
  - `generation::Int64`       — ≥ 1; generation index of the evaluation
  - `particle::Int64`         — ≥ 1; particle index within the generation
  - `proposal::Int64`         — ≥ 1; proposal index within the particle
  - `attempt::Int64`          — ≥ 1; attempt index (redraw-until-accepted)
  - `replicate::Int64`        — ≥ 1; simulator replicate index
  - `seed::UInt64`            — per-evaluation seed (normally derived with
    [`evaluation_seed`](@ref); see below)
  - `base_seed::UInt64`       — run-level base seed

Identity indices are stored as fixed-width `Int64`, so the primitive
representation does not depend on the host word size. No model, observed-data,
or inference-configuration object is stored here. Equality operates on all
eight fields.

# Seed values

`UInt64(0)` is a valid `seed` and `base_seed`; there is no reserved sentinel.
The convenience constructor derives `seed` from the other fields with
[`evaluation_seed`](@ref). The explicit-seed constructor exists for
deserialization and for callers that intentionally reproduce a legacy stream.

# Schema

`EVALUATION_SEED_SCHEMA_VERSION` documents the encoding used by
[`evaluation_seed`](@ref), and [`context_primitive`](@ref) records it. This is a
persistent encoding; do not change it without a major version bump and a
documented migration.

# Generation 0 / non-particle evaluations

Current engines always evaluate particles at generation ≥ 1. Generation 0 or
non-particle evaluations are not part of this contract. If a future engine
needs such evaluations, it should define its own documented sentinel semantics
rather than loosening validation here.
"""
struct EvaluationContext
    method::Symbol
    generation::Int64
    particle::Int64
    proposal::Int64
    attempt::Int64
    replicate::Int64
    seed::UInt64
    base_seed::UInt64

    function EvaluationContext(
        method::Symbol,
        generation::Integer,
        particle::Integer,
        proposal::Integer,
        attempt::Integer,
        replicate::Integer,
        seed::UInt64,
        base_seed::UInt64,
    )
        isempty(string(method)) && throw(ArgumentError(
            "EvaluationContext method must be a nonempty Symbol, got $(repr(method))"))
        g = _evaluation_index(generation, "EvaluationContext generation")
        p = _evaluation_index(particle, "EvaluationContext particle")
        q = _evaluation_index(proposal, "EvaluationContext proposal")
        a = _evaluation_index(attempt, "EvaluationContext attempt")
        r = _evaluation_index(replicate, "EvaluationContext replicate")
        return new(method, g, p, q, a, r, seed, base_seed)
    end
end

"""
    EvaluationContext(
        method::Symbol,
        generation::Integer,
        particle::Integer,
        proposal::Integer,
        attempt::Integer,
        replicate::Integer,
        base_seed::UInt64,
    ) -> EvaluationContext

Convenience constructor: derives `seed` from the remaining fields via
[`evaluation_seed`](@ref). All invariants are validated.
"""
function EvaluationContext(
    method::Symbol,
    generation::Integer,
    particle::Integer,
    proposal::Integer,
    attempt::Integer,
    replicate::Integer,
    base_seed::UInt64,
)
    seed = evaluation_seed(
        method, generation, particle, proposal, attempt, replicate, base_seed,
    )
    return EvaluationContext(
        method, generation, particle, proposal, attempt, replicate,
        seed, base_seed,
    )
end

"""
    context_primitive(context::EvaluationContext) -> NamedTuple

Stable, NamedTuple-based primitive representation of an evaluation context.
Suitable for serialization, logging, or debugging. The field order and names
are part of this contract and will not change without a major version bump.
"""
function context_primitive(context::EvaluationContext)::NamedTuple
    return (
        method = string(context.method),
        generation = context.generation,
        particle = context.particle,
        proposal = context.proposal,
        attempt = context.attempt,
        replicate = context.replicate,
        seed = context.seed,
        base_seed = context.base_seed,
        seed_schema = EVALUATION_SEED_SCHEMA_VERSION,
        seed_schema_name = EVALUATION_SEED_SCHEMA_NAME,
    )
end

"""
    EvaluationContext(prim::NamedTuple) -> EvaluationContext

Rebuild an [`EvaluationContext`](@ref) from its primitive representation
(see [`context_primitive`](@ref)). Validates all invariants on rebuild,
including that the recorded seed schema is the supported schema.
"""
function EvaluationContext(prim::NamedTuple)
    hasproperty(prim, :seed_schema) || throw(ArgumentError(
        "EvaluationContext primitive is missing the seed_schema field"))
    prim.seed_schema == EVALUATION_SEED_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported EvaluationContext seed schema $(repr(prim.seed_schema)); " *
        "expected $(EVALUATION_SEED_SCHEMA_VERSION)"))
    if hasproperty(prim, :seed_schema_name)
        prim.seed_schema_name == EVALUATION_SEED_SCHEMA_NAME || throw(ArgumentError(
            "unsupported EvaluationContext seed schema name " *
            "$(repr(prim.seed_schema_name)); expected " *
            "$(repr(EVALUATION_SEED_SCHEMA_NAME))"))
    end
    return EvaluationContext(
        Symbol(prim.method),
        prim.generation,
        prim.particle,
        prim.proposal,
        prim.attempt,
        prim.replicate,
        prim.seed,
        prim.base_seed,
    )
end

Base.:(==)(a::EvaluationContext, b::EvaluationContext) =
    (a.method == b.method) &&
    (a.generation == b.generation) &&
    (a.particle == b.particle) &&
    (a.proposal == b.proposal) &&
    (a.attempt == b.attempt) &&
    (a.replicate == b.replicate) &&
    (a.seed == b.seed) &&
    (a.base_seed == b.base_seed)

function Base.show(io::IO, ctx::EvaluationContext)
    return print(io,
        "EvaluationContext($(ctx.method), gen=$(ctx.generation), " *
        "particle=$(ctx.particle), proposal=$(ctx.proposal), " *
        "attempt=$(ctx.attempt), replicate=$(ctx.replicate), " *
        "seed=$(ctx.seed), base_seed=$(ctx.base_seed))")
end
