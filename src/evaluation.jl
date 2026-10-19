# src/evaluation.jl
#
# Structured evaluation contracts for deterministic simulator evaluation.
#
# This module defines a small, model-agnostic production contract:
#   * EvaluationContext — the immutable identity of one simulator evaluation
#   * EvaluationResult  — the structured outcome (success, failure, nonfinite, timeout)
#   * EvaluationFailure — a serializable description of a software failure
#   * Failure policies  — explicit, user-controllable handling of failures
#   * evaluate          — an additive adapter around existing simulator execution
#
# Design principles:
#   * Model-agnostic: no domain-specific terminology, no ABCSMCConfig, no
#     observed summaries, no model configuration inside EvaluationContext.
#   * Additive and backward-compatible: existing run_sim_once signatures and
#     inference behavior remain unchanged.
#   * Software failures never silently become numeric discrepancies.
#   * Fixed-width, serializable types with a versioned seed encoding.
#
# Nothing in these files is threaded into the SMC/PMC/MCMC engines. Engine
# adoption is left to a later focused PR.

using Random

include("evaluation/seed.jl")
include("evaluation/context.jl")
include("evaluation/failure.jl")
include("evaluation/result.jl")
include("evaluation/policies.jl")
include("evaluation/adapter.jl")
