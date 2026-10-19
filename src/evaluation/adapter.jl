# ============================================================================
# Compatibility adapter: evaluate
# ============================================================================

"""
    evaluate(
        simulator,
        parameters,
        context::EvaluationContext;
        provenance::NamedTuple=NamedTuple(),
        capture_stacktrace::Bool=true,
    ) -> EvaluationResult

Additive adapter around existing simulator execution. Invokes the simulator
using `context.seed`, measures wall time, captures successful summaries,
classifies nonfinite summaries, and converts simulator exceptions into
structured [`EvaluationFailure`](@ref) values.

`provenance` is caller-supplied metadata preserved verbatim in the result.
`capture_stacktrace` controls whether exception failures include a textual
stack trace (captured by default). Disable it if stack traces may contain
sensitive local paths or environment details.

A simulator returning `nothing` is malformed output: the result is `:failed`
with a structured failure, not a successful evaluation. `InterruptException`
and `OutOfMemoryError` are process/control conditions and are rethrown rather
than converted into scientific results.

The adapter never calculates or assigns a numeric discrepancy for a software
failure. Distance calculation remains separate: extract the summary from a
successful result, compare it with the observed summary, and calculate the
discrepancy.

Existing `run_sim_once` signatures are unchanged; this adapter is additive.
Timeout status is part of [`EvaluationResult`](@ref), but this adapter does not
implement wall-clock interruption; an outer execution/transport layer may
construct a `:timeout` result explicitly.
"""
function evaluate(
    simulator,
    parameters,
    context::EvaluationContext;
    provenance::NamedTuple=NamedTuple(),
    capture_stacktrace::Bool=true,
)
    start_ns = time_ns()
    rng = MersenneTwister(context.seed)

    summary::Any = nothing
    failure::Union{Nothing,EvaluationFailure} = nothing
    status::Symbol = EVALUATION_STATUS_SUCCESS

    try
        raw_summary = run_sim_once(simulator, parameters, rng)

        if raw_summary === nothing
            status = EVALUATION_STATUS_FAILED
            failure = EvaluationFailure(
                :simulator_error,
                "MalformedSummary",
                "simulator returned nothing; a successful evaluation must " *
                "return a non-nothing summary",
                nothing,
            )
        else
            summary = raw_summary
            if !_is_finite_summary(summary)
                status = EVALUATION_STATUS_NONFINITE
                failure = EvaluationFailure(
                    :nonfinite_summary,
                    "NonfiniteSummary",
                    "simulator returned a nonfinite summary",
                    nothing,
                )
                summary = nothing
            end
        end
    catch err
        # Interrupts and out-of-memory conditions are not simulator outcomes;
        # propagating them avoids turning process control into scientific data.
        (err isa InterruptException || err isa OutOfMemoryError) && rethrow()

        status = EVALUATION_STATUS_FAILED
        summary = nothing
        bt = capture_stacktrace ? catch_backtrace() : nothing
        failure = EvaluationFailure(
            :simulator_error,
            string(nameof(typeof(err))),
            sprint(showerror, err),
            bt === nothing ? nothing : sprint(io -> Base.show_backtrace(io, bt)),
        )
    end

    wall_time_s = Float64(time_ns() - start_ns) / 1.0e9

    return EvaluationResult(
        context, status, summary, provenance;
        failure=failure, wall_time_s=wall_time_s,
    )
end
