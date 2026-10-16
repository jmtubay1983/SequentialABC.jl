# Example: pyABC-style plots for an ABC-SMC run.
#
# A tiny, deterministic ABC-SMC fit of a Normal(μ, σ) to synthetic summary
# statistics, followed by every function of the plotting API. Figures are saved
# as PNGs to repro/plots/ (untracked).
#
# Run from the package root:
#   julia --project=examples/plots_env examples/plot_examples.jl
#
# The plotting backends live only in this example environment; they are loaded
# lazily by the package (Requires.jl) once `using Plots` has been executed.

import Pkg
Pkg.activate(joinpath(@__DIR__, "plots_env"))   # examples/plots_env
Pkg.instantiate()

ENV["GKSwstype"] = "100"        # headless GR: no display required

using SequentialABC
using Plots
using StatsPlots
using Distributions
using Random
using Statistics

# --- tiny deterministic ABC-SMC problem -------------------------------------
# Infer (μ, σ) of a Normal from two summaries: the sample mean and sample sd.
const N_SAMPLE = 20
summ(x) = [mean(x), std(x)]

μ_true, σ_true = 0.7, 0.5
s_obs = summ(rand(MersenneTwister(2024), Normal(μ_true, σ_true), N_SAMPLE))

prior_μ  = Uniform(-2.0, 2.0)
prior_σ  = Uniform(0.2, 2.0)

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ[2] > 0 ?
        summ(rand(rng, Normal(θ[1], θ[2]), N_SAMPLE)) : [Inf, Inf],
    summary = identity,
)

cfg = ABCSMCConfig(
    θ -> logpdf(prior_μ, θ[1]) + logpdf(prior_σ, θ[2]),
    rng -> [rand(rng, prior_μ), rand(rng, prior_σ)],
    identity,
    (s, so) -> sqrt(sum(abs2, s .- so)),
    GaussianKernel(),
    [0.1 0.0; 0.0 0.1],
    simulator,
)

# --- run ABC-SMC, keeping the per-generation history ------------------------
const N       = 80
const N_GEN   = 3
ckpt = mktempdir()
state = make_state(cfg, Vector{Float64}, N, 3.0, 42)
run_smc!(state, s_obs; T = N_GEN, schedule = QuantileSchedule(0.5), checkpoint_dir = ckpt)

history = [load_checkpoint(path, cfg) for (_, path) in list_checkpoints(ckpt)]
println("ran ABC-SMC: ", length(history), " generations, N = ", N,
        ", final ε = ", round(state.ε; digits = 4))

# --- produce every plot -----------------------------------------------------
outdir = normpath(joinpath(@__DIR__, "..", "repro", "plots"))
mkpath(outdir)

figures = [
    "marginals"   => plot_marginals(history; labels = ["μ", "σ"]),
    "pairs"       => plot_pairs(history; labels = ["μ", "σ"]),
    "trace"       => plot_trace(history; labels = ["μ", "σ"]),
    "diagnostics" => plot_diagnostics(history),
    "ppc"         => plot_ppc(history, s_obs),
]

for (name, fig) in figures
    path = joinpath(outdir, "abc_" * name * ".png")
    savefig(fig, path)
    println("saved ", path)
end

println("✓ plotting example completed successfully")
