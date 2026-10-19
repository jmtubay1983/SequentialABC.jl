# Simulator evaluation contracts

This page documents the additive evaluation contract introduced in
SequentialABC.jl 0.4.0. It is deliberately **model- and engine-agnostic**: it
describes one simulator evaluation, its result, and the handling of software
failures. It does not contain observed data, model configuration, posterior
types, or a discrepancy calculation.

The existing `run_sim_once` signatures and the SMC, PMC, and ABC-MCMC
engines are unchanged. The engines do **not** yet consume these types; engine
adoption is a later, separate step.

## Evaluation identity

[`EvaluationContext`](@ref) is the immutable identity of one evaluation:

```julia
ctx = EvaluationContext(
    :demo,          # method / engine label (nonempty Symbol)
    1,              # generation
    2,              # particle
    1,              # proposal
    1,              # attempt
    1,              # simulator replicate
    UInt64(7),      # run-level base seed
)
```

The identity indices are stored as fixed-width `Int64`; `seed` and `base_seed`
are `UInt64`. The convenience constructor derives `seed` with
[`evaluation_seed`](@ref). An explicit-seed constructor is available when a
caller intentionally needs to reproduce a particular stream:

```julia
ctx = EvaluationContext(:demo, 1, 2, 1, 1, 1, UInt64(12345), UInt64(7))
```

[`context_primitive`](@ref) returns a named-tuple representation with string
method, fixed-width integer fields, and the seed schema version and name. It can
be used for logging or serialization; rebuilding through
`EvaluationContext(primitive)` validates the schema and all identity
invariants.

### Seed schemas

Two schemas are intentionally distinct.

**Legacy engine schema.** [`make_particle_seed`](@ref) is the pre-existing seed
stream used by SMC, PMC, and ABC-MCMC. It is frozen and remains the source of
those engines' streams. Its schema name is
`"make_particle_seed_legacy_hash"`. Existing inference behavior and replay
streams must not change.

**Public evaluation schema v1.** [`evaluation_seed`](@ref) derives a `UInt64`
from the complete public identity:

- schema version `1`;
- method UTF-8 byte length and bytes;
- generation, particle, proposal, attempt, and replicate;
- base seed.

Each fixed-width field is encoded as eight little-endian bytes, folded with
FNV-1a 64-bit, and finalized with the SplitMix64 avalanche. The encoding is
explicit and deterministic across Julia sessions and operating systems. It does
not use Julia's `hash()`. The version is exported as
[`EVALUATION_SEED_SCHEMA_VERSION`](@ref) and recorded in
[`context_primitive`](@ref).

`UInt64(0)` is a valid `seed` and `base_seed`. There is no reserved zero
sentinel; `isfinite` is not used to justify rejecting it because `UInt64` has no
NaN or infinity.

The public schema is not a drop-in replacement for the legacy engine stream.
It exists for the new evaluation contract. A caller comparing the two schemas
must request the legacy seed explicitly and place it in an explicit-seed
context.

## Results and failures

[`EvaluationResult`](@ref) has a status, an optional summary, caller-supplied
provenance, an optional structured failure, and a finite nonnegative wall time.
The supported statuses are:

| Status | Meaning | Required fields |
|---|---|---|
| `:success` | Valid finite summary | `summary !== nothing`, `failure === nothing` |
| `:failed` | Simulator or infrastructure failure | `summary === nothing`, structured `failure` |
| `:nonfinite` | Output was numerically invalid | `summary === nothing`, structured `failure` |
| `:timeout` | Outer execution layer timed out | `summary === nothing`, structured `failure` |

A successful result accepts scalar, vector, tuple, named-tuple, and custom
summaries. Recursive finiteness validation covers `Real` values, arrays, tuples,
and named tuples. `missing` is not accepted as a successful summary; `nothing`
is not a successful summary either. Opaque custom objects are **not**
introspected: a custom wrapper can contain a NaN or other invalid value and
still be accepted. Callers that use opaque summaries are responsible for their
own validation. Complex numbers and dictionaries, for example, are opaque under
this contract.

[`EvaluationFailure`](@ref) contains only serializable structured information:
a category, an exception type name, a message, and an optional textual stack
trace. It never stores an arbitrary `Exception` object. Permitted categories are
`:simulator_error`, `:nonfinite_summary`, `:timeout`, and `:user_defined`.

## Failure policies

Failure policies make handling explicit:

- [`RejectEvaluation`](@ref) returns `:reject`.
- [`RetryEvaluation`](@ref) returns `:retry`.
- [`AbortInference`](@ref) returns `:abort`.
- [`FunctionFailurePolicy`](@ref) wraps a user function returning one of those
  three actions. Any other action is rejected with an `ArgumentError`.

[`retry_with_limit`](@ref) applies a policy **once** and enforces a bounded
retry budget. It is a policy-planning helper: it never runs a simulator and
never modifies the supplied failure result.

`max_retries` counts retries after the original evaluation. `attempt` starts at
1, so a context with `attempt == 1` and `max_retries == 3` can advance through
attempts 2, 3, and 4; another retry request returns `:abort` with the context
unchanged. Every successful retry increments `attempt` and re-derives the seed
with [`evaluation_seed`](@ref), so a retry never reuses the failed evaluation's
stream. Non-retry actions return the context unchanged, and the caller owns the
original structured failure.

## The evaluate adapter

[`evaluate`](@ref) is an additive adapter around a simulator:

```julia
result = evaluate(simulator, parameters, ctx; provenance = (run_id = 1,))
```

It invokes the existing three-argument `run_sim_once(simulator, parameters, rng)`
with a private `MersenneTwister(ctx.seed)`, measures monotonic wall time,
preserves the context and provenance, and returns:

- `:success` with the summary for a valid finite result;
- `:nonfinite` with a structured failure for invalid numerical output;
- `:failed` with a structured failure when the simulator throws, or when it
  returns `nothing` as a malformed successful summary.

Stack traces are captured by default and can be omitted with
`capture_stacktrace = false` for privacy-sensitive environments. Interrupts and
out-of-memory conditions are rethrown rather than converted into scientific
results. The adapter has no wall-clock timeout implementation; an outer
execution or transport layer can construct a `:timeout` result explicitly.

The adapter never computes a discrepancy. Distance calculation remains a
separate step:

```julia
result = evaluate(simulator, parameters, ctx)
if result.status === :success
    discrepancy = my_distance(result.summary, observed_summary)
    # Use discrepancy for ABC acceptance or diagnostics.
end
```

A large scientific discrepancy is not a failure: a finite summary far from the
observed summary remains `:success`. A simulator exception, nonfinite output,
timeout, or malformed output is a software/execution failure and must not be
turned into a numeric distance.

## A domain-neutral example

The following simulator is a stochastic shift. It has no domain-specific
meaning; it only demonstrates the adapter contract.

```julia
using SequentialABC
using Random

struct ShiftSimulator <: ABCSimulator
    scale::Float64
end

function SequentialABC.run_sim_once(sim::ShiftSimulator, θ, rng)
    return θ .+ sim.scale .* randn(rng, length(θ))
end

ctx = EvaluationContext(:demo, 1, 1, 1, 1, 1, UInt64(7))
result = evaluate(ShiftSimulator(0.5), [1.0, -1.0], ctx)

result.status          # :success
result.summary         # finite vector
result.failure         # nothing
```

The same context always produces the same stream and therefore the same
successful summary for a deterministic simulator. Changing any identity field
(or the base seed) changes the public evaluation seed.

## Compatibility

The contract is additive:

- existing `run_sim_once` methods and signatures are unchanged;
- SMC, PMC, and ABC-MCMC still use their legacy seed stream and remain
  unmigrated;
- no inference engine computes, stores, or consumes these evaluation results;
- `EvaluationResult` has no distance field, and the adapter has no distance
  argument;
- no RunStore, posterior type, or new inference algorithm is introduced.

The types are intended as a stable foundation for a later, separately reviewed
engine-adoption step.
