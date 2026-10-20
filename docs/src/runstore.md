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

[`AbstractRunStore`](@ref) is the boundary. Two backends implement it:
[`MemoryRunStore`](@ref), the **normative reference implementation**, which
fixes what a run history means without committing to files, serialization,
checksums, locking, or a portable export format; and
[`NativeRunStore`](@ref), the file-backed backend, which stores the same
history on disk with digests, manifests, and commit markers.

Both backends are held to one shared conformance suite, so their semantics
cannot drift apart.

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

## The native store

[`NativeRunStore`](@ref) is the file-backed backend. It implements exactly the
same semantics as the in-memory backend, so both are held to one shared
conformance suite.

| | [`MemoryRunStore`](@ref) | [`NativeRunStore`](@ref) |
|---|---|---|
| history lives in | the process | the run directory |
| survives a restart | no | yes |
| writer ownership | none | a writer lease |
| integrity | none | SHA-256 digests, a manifest, and a `COMPLETE` marker |
| use | tests, and a reference for the semantics | real runs |

### Layout

```text
run_dir/
  metadata/
    schema              # storage schema name and version
    run.jls             # the run metadata, written once
    run.sha256          # SHA-256 of the exact bytes of run.jls
    writer.lease/       # the writer lease, held while a writer is active
      holder            # who holds the lease
  generations/
    generation_000001/
      population.jls    # the accepted population
      evaluations.jls   # the evaluations, in canonical order
      manifest          # sizes, digests, counts, and identities
      COMPLETE          # written last; binds the manifest
  tmp/
    generation_000001.<nonce>/   # staging; never visible
```

Every complete generation is one directory with exactly four canonical
artifacts. There is one file per evaluation: the evaluations of a generation
live in a single partition.

### Opening, reading, and writing

```julia
store = initialize_run!(NativeRunStore, "runs/demo"; metadata = metadata)  # write mode
read  = open_run(NativeRunStore, "runs/demo"; mode = :read)
close_run!(store)
```

A store in **write mode** owns the writer lease for the whole session and
rejects a second writer. A store in **read mode** takes no lease: it can
read the committed generations while another writer is active, and it never
sees a staging directory.

### Pending records are not durable

The records of the generation that is being written live in the store
session. They are **not durable** until
[`finalize_generation!`](@ref) publishes the generation directory.
Finalization is the durability boundary of the run store.

This is deliberate, and it matches the checkpoint contract: a checkpoint is
written only after a completed generation, and the deterministic seeds make
an interrupted generation reproducible. A pending generation that is never
published is discarded — with a warning when the store is closed — and it
never leaves a partial generation behind.

### Publication

Publication writes the whole generation into a staging directory, verifies
it, and then moves the staging directory to its canonical name with a single
rename:

1. the population partition is streamed to the staging directory;
2. the evaluation partitions are streamed in the canonical order;
3. both partitions are reopened and revalidated;
4. the SHA-256 digests and the byte sizes are computed;
5. the manifest is written and validated;
6. the `COMPLETE` marker is written **last**;
7. the staging directory is validated as a whole;
8. the staging directory is renamed to `generations/generation_000001/`;
9. the published directory is validated again.

An interruption before the rename leaves only a staging directory, which no
reader ever sees. A generation is visible only when its directory exists and
its `COMPLETE` marker, its manifest, and its partitions verify.

### Integrity, and what it does not buy

The manifest records the digest, the size, and the record count of every
partition, the digest of the run metadata, a digest of the evaluation
identities, and (when a plan was supplied) the digest of the expected plan.
The `COMPLETE` marker records the digest of the manifest, so the marker
binds the manifest and the manifest binds the data.

SHA-256 digests detect **accidental corruption and truncation**. They are
not authentication: a writer that can rewrite the data, the manifest, and
the marker consistently defeats them. The native store is a tool for
trusted local files.

**Checksums versus semantic validation.** The two layers answer different
questions:

* a digest proves that the bytes of a partition are the bytes that were
  published;
* the *constructor revalidation* proves that the decoded scientific record
  still satisfies the run-history invariants.

A partition whose digests are consistent but whose records are invalid — for
instance a population whose weights are not normalized — is rejected by the
revalidation, not by a checksum. The store never hands out a record it would
itself refuse to construct.

### Failure behaviour and recovery

**Staging versus committed.** A generation under `tmp/` is *staged*: it is
invisible to every reader, it is not part of the generation inventory, and
it is not a generation of the run. A generation under `generations/` with a
`COMPLETE` marker is *committed*: it is listed, it is queryable, and it is
verified on every read.

**A crash before the final rename.** The directory rename is the commit
point. If the writer stops before it, the staging directory stays behind and
*no* new generation becomes visible:

* the previously committed generations are byte-identical and remain
  queryable;
* `latest_generation` still reports the last committed generation;
* the staging directory is reported by [`validate_store`](@ref) as a
  warning;
* nothing is deleted and nothing is repaired: removing the staging
  directory is an operator decision, and the store may keep it for
  inspection;
* the writer lease of the interrupted session is *not* released
  automatically — see below.

**A crash after the final rename.** The publication already completed: the
generation is committed, complete, and visible. Only the acknowledgement was
interrupted. The store therefore reports the generation as published, the
pending records of that generation are stale and are dropped, and the
generation cannot be published a second time — a retry is refused with
[`GenerationAlreadyComplete`](@ref). This is *not* a failed publication.

**Operator responsibilities.** The store never repairs itself. An operator
may have to

* remove a leftover staging directory after an interrupted publication;
* remove a stale writer lease after a crashed writer (below);
* replace or archive a generation whose files were damaged.

**Corruption is detected, never repaired.** A committed generation whose
files were damaged fails closed: the read raises a contextual
[`RunStoreError`](@ref) — [`ChecksumMismatch`](@ref),
[`TruncatedPartition`](@ref), [`CorruptStore`](@ref),
[`MissingManifest`](@ref), [`MalformedManifest`](@ref) or
[`MalformedMarker`](@ref) — and there is never a fallback to an older
generation. [`validate_store`](@ref) reports the damage and leaves every
artifact untouched.

**No protection against a malicious writer.** The digests detect accidental
damage, not a writer that rewrites the data, the manifest, and the marker
consistently. The store does not attempt to defend against that, and it does
not authenticate the run.

**No restart claim.** A damaged or missing generation costs history, not
the run: the sampler is restarted from a checkpoint, never from the store.

**Pending records remain nondurable.** The records of a generation that is
being written live in the store session until
[`finalize_generation!`](@ref) publishes it. A crash loses them, exactly as
it loses the generation, and the generation is recomputed from its
deterministic seeds.

### Native format, trusted input only

The `.jls` partitions use Julia `Serialization`. They are native to Julia and
to the types of this package, they are **trusted-input only** (deserialization
reconstructs arbitrary types and can execute code), and they are **not** a
portable export format. Summaries and provenance are canonical
[`RecordedValue`](@ref) trees and are therefore always readable; particles
are native, so a population whose particle types no longer exist can become
unreadable.

Because `Serialization` is not safe against a corrupted *type* header — a
damaged header can crash the process instead of raising a catchable error —
the store verifies every digest **before** it deserializes anything.
Accidental damage is therefore rejected without ever reaching the
deserializer; only a writer that rewrites the digests consistently can get
there, which is why the format is documented as trusted-input only.

Portable, untrusted interchange belongs to a later export layer. There is
none yet.

### A stale writer lease

The writer lease is never broken automatically. If a writer died, the lease
remains and every new writer is rejected with
[`WriterAlreadyActive`](@ref) (or [`StaleWriterLease`](@ref) when the lease
cannot be interpreted). The procedure is manual:

1. read `metadata/writer.lease/holder` for the host, the process id, and the
   acquisition time;
2. confirm that the recorded writer is not running any more — process-id
   liveness is advisory only, and a process id can be reused;
3. remove `metadata/writer.lease/` by hand.

The lease is not removed by the store, not by a timeout, and not by a
validator: [`validate_store`](@ref) only reports it.

### Filesystem assumptions

The native store assumes a local filesystem: the run directory is staged and
published inside itself, so the publication is a same-volume rename, and
the digests and the `COMPLETE` marker detect an interrupted or damaged
publication. Network filesystems do not provide the same guarantees, and the
store does not claim them.

### The native store does not resume inference

The native store records history. It holds no configuration, no RNG state,
and no executable sampler state, and it exposes no operation that
reconstructs a sampler. Exact restart remains the job of
[`save_checkpoint`](@ref) and [`load_latest_state`](@ref).

The two mechanisms have different guarantees, and neither replaces the other:
a checkpoint is the exact sampler-restart authority, while the store adds
digests, manifests, and completion markers to the *history*. The store does
not make a checkpoint verifiable, and a checkpoint does not record the run's
history. See [Checkpoint guarantees](design.md#6-checkpoint-guarantees) for the
side-by-side comparison.
