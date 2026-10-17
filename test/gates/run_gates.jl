# test/gates/run_gates.jl
# Validation-gate orchestrator.
#
# Runs Gates 1–4 and prints one canonical verdict:
#   ENGINE_VALIDATED
#   CAPABILITY_GAPS(<list>)
#   TEST_FAILURE
#   ABM_INTEGRATION_NOT_RUN
#
# These gates are intentionally NOT part of `Pkg.test()` (test/runtests.jl is
# untouched). They are run by a separate, non-required CI job that is allowed to
# fail, so that missing production capabilities are surfaced honestly without
# turning the main suite red.
#
# Run: julia --project=. test/gates/run_gates.jl

include("gate_common.jl")
include("gate1_api.jl")
include("gate2_statistics.jl")
include("gate3_reliability.jl")
include("gate4_external_simulator.jl")

function main()
    results = GateResult[]
    for gate_fn in (run_gate1, run_gate2, run_gate3, run_gate4)
        r = gate_fn()
        push!(results, r)
        summarize(r)
    end

    verdict = compute_verdict(results)

    println()
    println("═" ^ 60)
    println("VALIDATION VERDICT: $(verdict)")
    println("═" ^ 60)

    # Compact gap list for the verdict line
    gaps = String[]
    for r in results
        append!(gaps, r.capability_gaps)
    end
    if !isempty(gaps)
        println("Missing production capabilities:")
        for g in gaps
            println("  - $(g)")
        end
    end
    for r in results
        if r.not_run_reason !== nothing
            println("Not run: $(r.not_run_reason)")
        end
    end

    # Exit code: 0 only on ENGINE_VALIDATED; non-zero otherwise (CI job allowed
    # to fail, so non-zero is fine and honest).
    exit(verdict == "ENGINE_VALIDATED" ? 0 : 1)
end

main()
