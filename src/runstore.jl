# src/runstore.jl
#
# Run-history records and the in-memory reference run store.
#
# This PR fixes the *semantics* of a run history independently of any
# storage technology:
#   * RecordedValue        — closed canonical value tree
#   * RunMetadata          — immutable identity of a run
#   * EvaluationKey        — identity of one evaluation
#   * EvaluationRecord     — one recorded evaluation
#   * PopulationRecord     — one recorded population
#   * AbstractRunStore     — store boundary
#   * MemoryRunStore       — normative reference backend
#
# Design principles:
#   * Model- and domain-agnostic: no model, observed data, or inference
#     configuration is stored.
#   * A store records history; it never resumes inference. No API reconstructs
#     sampler state from a store.
#   * Records are immutable and validated at construction; a failed operation
#     never leaves a partially visible result.
#   * Arbitrary user objects are never stored implicitly: summaries and
#     provenance are converted into a closed, canonical value tree, and
#     omissions stay visible.
#   * Software failures never become numeric discrepancies.
#
# Filesystem storage, serialization, checksums, locking, atomic publication,
# and portable export are deliberately out of scope here; a later backend
# implements the same semantics on top of these types.

include("runstore/errors.jl")
include("runstore/native_errors.jl")
include("runstore/values.jl")
include("runstore/metadata.jl")
include("runstore/records.jl")
include("runstore/filters.jl")
include("runstore/canonical.jl")
include("runstore/partition.jl")
include("runstore/memory.jl")
include("runstore/native.jl")
