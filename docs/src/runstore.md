# Run history

A *run store* holds the queryable history of a run: the accepted populations
of every generation, the evaluations that were executed, their statuses and
failures, their wall times, and the provenance that describes where the data
came from.

This page documents the run-history contract that ships with the in-memory
reference backend. It is deliberately independent of any storage technology.

## What a run store is — and is not

| | |
|---|---|
| A run store **is** | a queryable, append-only record of populations, evaluations, statuses, failures, timings, and provenance |
| A run store **is not** | a checkpoint, a snapshot of sampler state, or a way to resume inference |

A run store holds no configuration, no RNG state, and no executable sampler
state. There is no API that reconstructs a sampler from a store, and none is
planned: exact restart remains the job of
[`save_checkpoint`](@ref) and [`load_latest_state`](@ref).

The two are complementary and independent. A run may have both, and neither
substitutes for the other.

## The store boundary

[`AbstractRunStore`](@ref) is the boundary. The backend in this build is
[`MemoryRunStore`](@ref), the **normative reference implementation**: it fixes
what a run history means without committing to files, serialization,
checksums, locking, or a portable export format. A persistent backend must
reproduce the same semantics.

```julia
metadata = current_run_metadata(run_id = "demo-run", method = :smc, base_seed = 42)
store = initialize_run!(MemoryRunStore; metadata = metadata)
```

A store holds one immutable [`RunMetadata`](@ref), the pending generations
that were started but not completed, and the complete generations. Complete
generations are immutable.

## Canonical values

Summaries and provenance are converted into a closed, canonical value tree
([`RecordedValue`](@ref)) so that a store never holds a reference to an
arbitrary user object.

* Supported: `nothing`, `Bool`, `Int64`, `UInt64`, `Float64`, `String`,
  `Symbol`, vectors, tuples, and named tuples.
* Nonfinite numbers are rejected. A nonfinite value is represented
  explicitly with a tagged [`RecordedSentinel`](@ref) (`:nan`,
  `:positive_infinity`, `:negative_infinity`).
* Unsupported values are rejected with
  [`UnsupportedRecordedValue`](@ref). An explicit omission is possible and
  stays visible as a [`RecordedOmitted`](@ref) that carries the type name and
  the reason.
* Conversion is deterministic and recursively validated.

```julia
recorded_value((a = 1.0, b = [2.0]))     # a canonical named tuple
recorded_value(Dict(:a => 1))            # error: unsupported
recorded_value(Dict(:a => 1); omit = true)  # an explicit, visible omission
```

Provenance is canonical in the same way: a `NamedTuple` becomes one entry
per field, and any other provenance object is rejected
([`recorded_provenance`](@ref)).

## Records

### PopulationRecord

[`PopulationRecord`](@ref) describes one accepted population: the particles,
the normalized weights, the acceptance distances, the tolerance, the
effective sample size, optional per-particle attempt counts, optional
ancestry, the local resampling count, optional
[`AdaptiveDiagnostics`](@ref), the start and end timestamps, and the
provenance.

The constructor is method-agnostic. It checks the structural invariants —
aligned vector lengths, finite and normalized weights, finite and nonnegative
distances, a finite effective sample size in `[1, N]`, valid attempt counts
and ancestry indices, ordered timestamps — but it does **not** require
`distance <= epsilon`, because acceptance consistency is a property of a
particular inference method. Use
[`check_acceptance_consistency`](@ref) for that check.

The only nonfinite value the contract allows is the documented
generation-one positive-infinity epsilon sentinel, which represents an
unconstrained initial tolerance.

### EvaluationRecord

[`EvaluationRecord`](@ref) is built from a production
[`EvaluationResult`](@ref) and keeps the production invariants: a successful
evaluation has no failure, a non-successful evaluation has a structured
[`EvaluationFailure`](@ref) and no summary, and the wall time is finite and
nonnegative.

```julia
result = evaluate(simulator, parameters, context)
record = EvaluationRecord(result)                    # a recorded history entry
record = EvaluationRecord(result; discrepancy = 0.3) # a scientific discrepancy
```

A discrepancy is a value the caller calculated scientifically from a
successful summary. **The store never computes one**, and a failure never
receives one: a software failure is never turned into a number.

### Evaluation identity

[`EvaluationKey`](@ref) is the identity of an evaluation inside a run:
`(method, generation, particle, proposal, attempt, replicate)`. The seed is
deliberately not part of the key, so a repeated identity is a duplicate even
when the seed differs. The key defines the deterministic total order that
every query uses.

## Store semantics

| operation | meaning |
|---|---|
| `initialize_run!` | create a store, or reuse one under identical metadata |
| `open_run` | reuse an already initialized store under identical metadata |
| `write_population!` | record the population of the next writable generation |
| `write_evaluation!` | append an evaluation to the next writable generation |
| `finalize_generation!` | complete a generation; it becomes immutable |
| `read_population` | read the population of a complete generation |
| `read_evaluations` | query evaluations with filters, in a deterministic order |
| `latest_generation`, `generations` | the complete generations of the store |
| `run_metadata`, `schema_version` | the identity of the store |
| `validate_store` | check every invariant and report violations |

The rules that a persistent backend must reproduce:

* **Append-only.** Records are never silently replaced. A generation holds
  exactly one population, and a population is never overwritten.
* **Contiguous generations.** Generations start at 1 and follow each other.
  Writing generation 3 before generation 2 is a
  [`GenerationGap`](@ref).
* **Complete generations are immutable.** Writing to or re-finalizing a
  complete generation is a [`GenerationAlreadyComplete`](@ref).
* **Unique evaluation identities.** A repeated identity inside a generation
  is a [`DuplicateEvaluation`](@ref).
* **One method per run.** An evaluation whose method differs from the run
  method is a [`MethodMismatch`](@ref).
* **Metadata conflicts are errors.** A store has exactly one identity; a
  different metadata object is a [`ConflictingRunMetadata`](@ref).
* **Expected plans are checked when supplied.** `finalize_generation!`
  compares the recorded identities with the expected ones and reports
  [`MissingExpectedEvaluation`](@ref) or [`UnexpectedEvaluation`](@ref).
* **Reads return copies.** A caller cannot mutate stored history.

```julia
write_population!(store, population)
write_evaluation!(store, EvaluationRecord(result))
completion = finalize_generation!(store, 1;
                                  expected_evaluations = [EvaluationKey(result.context)])
```

## Queries

[`read_evaluations`](@ref) filters by generation, particle, proposal, attempt,
replicate, status, failure category, and an inclusive wall-time range. The
result is ordered by generation and then by [`EvaluationKey`](@ref), so the
order never depends on the order in which records were written.

```julia
read_evaluations(store; generation = 3, status = :failed)
read_evaluations(store; failure_category = :timeout,
                 min_wall_time_s = 1.0, max_wall_time_s = 60.0)
```

## Validation

[`validate_store`](@ref) returns a [`StoreValidationReport`](@ref) with the
errors, the warnings (such as a generation that was started but never
completed), the complete and pending generations, and every visible
omission. Validation never mutates the store.

## What is not here yet

The following are deliberately out of scope of the run-history core and are
the subject of a later backend:

* filesystem layout, atomic publication, and generation commit markers;
* serialization, checksums, and integrity manifests;
* writer locking;
* portable export (JSONL, CSV, Arrow, …) and re-import;
* engine instrumentation (the SMC, PMC, and ABC-MCMC engines are untouched;
  they still use their own seed streams).

The schemas are independent concepts and stay that way: the storage schema
([`RUNSTORE_SCHEMA_NAME`](@ref) /
[`RUNSTORE_SCHEMA_VERSION`](@ref)), the seed schema
([`EVALUATION_SEED_SCHEMA_VERSION`](@ref)), the checkpoint schema, and a
future export schema are separate. [`RunMetadata`](@ref) records the storage
schema and the *declared* seed schema; it never interprets or re-derives a
seed.
