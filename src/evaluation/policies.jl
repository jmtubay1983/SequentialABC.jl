
# ============================================================================
# Failure policy
# ============================================================================

"""
    FailurePolicy

Abstract type for explicit failure policies. A policy receives an
[`EvaluationContext`](@ref) and an [`EvaluationResult`](@ref) and returns a
documented action symbol: `:retry`, `:reject`, or `:abort`.

Custom policies must return one of these three symbols; any other return
value is rejected as invalid.
"""
abstract type FailurePolicy end

"""
    RejectEvaluation <: FailurePolicy

Reject the particle: do not create a positive-weight accepted particle.
The caller should treat the evaluation as if it never produced a usable
summary.
"""
struct RejectEvaluation <: FailurePolicy end

(policy::RejectEvaluation)(ctx::EvaluationContext, result::EvaluationResult) = :reject

"""
    RetryEvaluation <: FailurePolicy

Retry the evaluation: increment `attempt` and obtain a new deterministic seed.
Exhausted retries fail clearly rather than looping indefinitely (see
[`retry_with_limit`](@ref)).
"""
struct RetryEvaluation <: FailurePolicy end

(policy::RetryEvaluation)(ctx::EvaluationContext, result::EvaluationResult) = :retry

"""
    AbortInference <: FailurePolicy

Abort the inference run, preserving the original structured failure
information.
"""
struct AbortInference <: FailurePolicy end

(policy::AbortInference)(ctx::EvaluationContext, result::EvaluationResult) = :abort

"""
    FunctionFailurePolicy(f::Function) <: FailurePolicy

User-defined failure policy adapter. `f` is called as
`f(context, result)` and must return one of `:retry`, `:reject`, or `:abort`.
Any other return value is rejected as invalid.
"""
struct FunctionFailurePolicy <: FailurePolicy
    f::Function
end

function (policy::FunctionFailurePolicy)(
    ctx::EvaluationContext, result::EvaluationResult,
)
    action = policy.f(ctx, result)
    action in (:retry, :reject, :abort) || throw(ArgumentError(
        "user-defined failure policy returned $(repr(action)); " *
        "must be one of :retry, :reject, or :abort"))
    return action
end

"""
    retry_with_limit(
        policy::FailurePolicy,
        ctx::EvaluationContext,
        result::EvaluationResult,
        max_retries::Int,
    ) -> Tuple{Symbol,EvaluationContext}

Apply a failure policy once and enforce a bounded retry budget.

This helper computes *policy actions only*. It never runs a simulator and never
creates or modifies an [`EvaluationResult`](@ref); the structured failure in
`result` remains owned by the caller.

Semantics:

  - `max_retries` is the maximum number of retries allowed after the original
    attempt. Attempts are counted by `ctx.attempt`, which starts at 1, so the
    helper aborts when `ctx.attempt - 1 >= max_retries`.
  - If the policy returns `:retry` and budget remains, the returned context has
    `attempt + 1` and a seed re-derived by [`evaluation_seed`](@ref) for the new
    attempt. The caller performs the next simulation with that context.
  - If the policy returns `:retry` and the budget is exhausted, `:abort` is
    returned with the current context unchanged.
  - `:reject` and `:abort` are returned with the current context unchanged.
  - Any other policy action throws `ArgumentError`.

The function performs exactly one policy call per invocation. It cannot loop
indefinitely, regardless of the policy's return value.
"""
function retry_with_limit(
    policy::FailurePolicy,
    ctx::EvaluationContext,
    result::EvaluationResult,
    max_retries::Int,
)
    max_retries >= 0 || throw(ArgumentError(
        "retry_with_limit max_retries must be ≥ 0, got $(max_retries)"))

    action = policy(ctx, result)
    action in (:retry, :reject, :abort) || throw(ArgumentError(
        "failure policy returned $(repr(action)); " *
        "must be one of :retry, :reject, or :abort"))

    action === :retry || return (action, ctx)

    retries_used = ctx.attempt - 1
    retries_used >= max_retries && return (:abort, ctx)

    next_ctx = EvaluationContext(
        ctx.method,
        ctx.generation,
        ctx.particle,
        ctx.proposal,
        ctx.attempt + 1,
        ctx.replicate,
        ctx.base_seed,
    )
    return (:retry, next_ctx)
end
