# Example: the unified run_inference entrypoint.
#
#   julia --project=. examples/run_inference_example.jl
#
# One toy model, one observation, one front door: the engine is just a Symbol,
# so the same script can be pointed at ABC-SMC, ABC-MCMC or ABC-PMC without
# changing anything else. The dispatcher forwards the keywords unchanged and
# returns each engine's own result type, so everything downstream keeps working.
#
# The whole script is deterministic: pinned seeds replay identical runs.

using SequentialABC
using Distributions
using Random

# --- toy model:  θ ~ N(0, 4²I),  s(θ) = θ + N(0, I)  ---------------------------
prior(θ) = sum(logpdf(Normal(0.0, 4.0), x) for x in θ)
sample_prior(rng) = 4.0 .* randn(rng, 2)

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = identity,
)

cfg = ABCSMCConfig(
    prior,
    sample_prior,
    identity,
    (s, so) -> sqrt(sum(abs2, s .- so)),
    GaussianKernel(),
    [1.0 0.0; 0.0 1.0],
    simulator,
)

s_obs = [1.0, -1.0]

const SEED   = 20260118
const LADDER = [6.0, 4.0, 2.5, 1.5]      # fixed tolerance ramp: ε₁ > … > ε₄

# --- one front door, three engines --------------------------------------------
smc = run_inference(:smc, cfg, s_obs; N = 100, T = 4, ε0 = LADDER[1],
                    schedule = FixedSchedule(LADDER), base_seed = SEED)
println("smc:  final ε = ", round(smc.ε; digits = 4), ", N = ", length(smc.θ),
        ", ESS ≈ ", round(smc.ess; digits = 1), ", resamples = ", smc.n_resamples)

chain = run_inference(:mcmc, cfg, s_obs; n_samples = 500, ε = 4.0, base_seed = SEED)
println("mcmc: samples = ", length(chain), ", ε = ", chain.ε,
        ", accept rate = ", round(acceptance_rate(chain); digits = 3))

pmc = run_inference(:pmc, cfg, s_obs; N = 100, T = 4, εs = LADDER, base_seed = SEED)
println("pmc:  final ε = ", round(pmc.ε; digits = 4), ", N = ", length(pmc.θ),
        ", resamples = ", pmc.n_resamples)

# --- the results are the engines' own types, unwrapped ------------------------
@assert smc isa ABCSMCState   "expected run_smc's result type"
@assert chain isa ABCMCMCChain "expected run_mcmc's result type"
@assert pmc isa ABCSMCState   "expected run_pmc's result type"
@assert pmc.n_resamples == 0  "PMC must never resample"

# --- deterministic: the same call replays bit-identically ---------------------
smc2 = run_inference(:smc, cfg, s_obs; N = 100, T = 4, ε0 = LADDER[1],
                     schedule = FixedSchedule(LADDER), base_seed = SEED)
@assert smc2.θ == smc.θ && smc2.w == smc.w "run_inference is not reproducible"

# --- an unsupported engine is rejected before anything is simulated -----------
err = try
    run_inference(:bsl, cfg, s_obs; N = 100, T = 4)
    nothing
catch e
    e
end
@assert err isa ArgumentError "an unknown engine must be rejected"
println("unknown engine -> ", sprint(showerror, err))

println("\n✓ run_inference example completed successfully")
