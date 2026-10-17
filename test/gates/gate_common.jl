# test/gates/gate_common.jl
# Shared result types and check helpers for the validation gates.
#
# Gates run OUTSIDE Pkg.test() (see test/gates/run_gates.jl) and must never
# throw on a missing capability: they record CAPABILITY_NOT_IMPLEMENTED and
# continue, so every gate reports before the harness finishes.

using SequentialABC
using Distributions
using Random
using Statistics
using LinearAlgebra
using Serialization

"""
    GateCheck(name, passed, detail)

A single check within one gate.
"""
struct GateCheck
    name::String
    passed::Bool
    detail::String
end

"""
    GateResult(name; checks, capability_gaps, test_failures, not_run_reason)

Result of one gate.

`capability_gaps` holds `"CAPABILITY_NOT_IMPLEMENTED: <name>"` strings for
production capabilities the engine does not provide. `test_failures` holds
details for assertions that *could* be evaluated against production code but
came back wrong. `not_run_reason` is set when a gate deliberately does not run
(e.g. `"ABM_INTEGRATION_NOT_RUN: no downstream adapter supplied"`).
"""
mutable struct GateResult
    name::String
    checks::Vector{GateCheck}
    capability_gaps::Vector{String}
    test_failures::Vector{String}
    not_run_reason::Union{Nothing,String}
end

GateResult(name::String) = GateResult(name, GateCheck[], String[], String[], nothing)

"""
    check!(r, name, passed; detail="")

Record one check against production code. Returns `passed`.
"""
function check!(r::GateResult, name::String, passed::Bool; detail::String="")
    push!(r.checks, GateCheck(name, passed, detail))
    passed || push!(r.test_failures, detail)
    return passed
end

"""
    capability_gap!(r, name)

Record that production capability `name` is missing. Does NOT throw and does
not install any test-local stand-in.
"""
function capability_gap!(r::GateResult, name::String)
    gap = "CAPABILITY_NOT_IMPLEMENTED: $name"
    push!(r.capability_gaps, gap)
    push!(r.checks, GateCheck(name, false, gap))
    return false
end

"""
    summarize(r)

Print a human-readable summary of one gate result.
"""
function summarize(r::GateResult)
    println("  $(r.name)")
    npass = count(c -> c.passed, r.checks)
    nreal_fail = length(r.test_failures)
    ngap = length(r.capability_gaps)
    println("    checks: $npass passed, $nreal_fail failed, $ngap capability gaps")
    for g in r.capability_gaps
        println("    GAP: $g")
    end
    for f in r.test_failures
        println("    FAIL: $f")
    end
    if r.not_run_reason !== nothing
        println("    NOT_RUN: $(r.not_run_reason)")
    end
    return nothing
end

"""
    compute_verdict(results)

Combine gate results into one verdict string.

Precedence: TEST_FAILURE > CAPABILITY_GAPS > ABM_INTEGRATION_NOT_RUN > ENGINE_VALIDATED.
"""
function compute_verdict(results::Vector{GateResult})
    for r in results
        if !isempty(r.test_failures)
            return "TEST_FAILURE"
        end
    end
    gaps = String[]
    for r in results
        append!(gaps, r.capability_gaps)
    end
    if !isempty(gaps)
        return "CAPABILITY_GAPS($(join(gaps, ", ")))"
    end
    for r in results
        if r.not_run_reason !== nothing
            return r.not_run_reason
        end
    end
    return "ENGINE_VALIDATED"
end

"""
    _std_config(; dim, noise, prior_sd, transform)

Build a standard ABCSMCConfig for gate testing.

  - prior:      N(0, prior_sd^2) on the raw parameter space
  - sample_prior: draws from the same N(0, prior_sd^2)
  - transform:  raw → model space (applied only at simulation time)
  - likelihood: summary = transform(θ_raw) + N(0, noise^2)
  - kernel:     GaussianKernel with Σ = I
"""
function _std_config(; dim::Int=1, noise::Float64=0.1, prior_sd::Float64=5.0,
                     transform=identity)
    prior = θ -> sum(logpdf(Normal(0.0, prior_sd), x) for x in θ)
    sample_prior = rng -> prior_sd .* randn(rng, dim)
    distance = (s, s_obs) -> sqrt(sum(abs2, s .- s_obs))
    Σ = Matrix{Float64}(I, dim, dim)
    sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> transform(θ) .+ noise .* randn(rng, dim),
        summary = identity,
    )
    return ABCSMCConfig(prior, sample_prior, transform, distance, GaussianKernel(), Σ, sim)
end
