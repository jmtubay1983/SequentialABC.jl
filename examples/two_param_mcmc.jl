# Example: two-parameter ABC-MCMC.
#
#   julia --project=. examples/two_param_mcmc.jl
#
# Linear regression  y = a + b·x + noise  with a Gaussian prior on (a, b).
# The summary statistic is the ordinary-least-squares fit (â, b̂), and the
# ABC-MCMC chain targets the ABC posterior of (a, b) — the same result
# container as an ABC-SMC run, so the plotting and postprocessing tools apply
# to the chain unchanged.
#
# The whole script is deterministic: the same seed replays an identical chain.

using SequentialABC
using Distributions
using Random
using Statistics

# --- toy model:  y = a + b·x + N(0, σ²) --------------------------------------
const N_OBS  = 50
const X      = collect(range(-1.0, 1.0; length = N_OBS))
const A_TRUE = 1.0
const B_TRUE = 0.5
const NOISE  = 0.3

# OLS fit as the summary statistic
function ols_summary(y)
    xm = X .- mean(X)
    b = sum(xm .* (y .- mean(y))) / sum(abs2, xm)
    a = mean(y) - b * mean(X)
    return [a, b]
end

prior(θ) = logpdf(Normal(0.0, 2.0), θ[1]) + logpdf(Normal(0.0, 1.0), θ[2])
sample_prior(rng) = [2.0 * randn(rng), randn(rng)]

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ[1] .+ θ[2] .* X .+ NOISE .* randn(rng, N_OBS),
    summary = ols_summary,
)

s_obs = ols_summary(A_TRUE .+ B_TRUE .* X .+ NOISE .* randn(MersenneTwister(1234), N_OBS))

cfg = ABCSMCConfig(
    prior,
    sample_prior,
    identity,
    (s, so) -> sqrt(sum(abs2, s .- so)),
    GaussianKernel(),
    [0.05 0.0; 0.0 0.05],
    simulator,
)

# --- the chain ---------------------------------------------------------------
const N_SAMPLES = 2000
const EPS       = 0.35

chain = run_mcmc(cfg, s_obs; n_samples = N_SAMPLES, ε = EPS, base_seed = 2024)

θ = chain.state.θ
w = chain.state.w
wmean(f) = sum(w .* [f(p) for p in θ])
wsd(f) = (m = wmean(f); sqrt(sum(w .* ([f(p) for p in θ] .- m) .^ 2)))

println("observed summary s_obs = ", round.(s_obs; digits = 3))
println("ε = ", EPS, ",  stored samples = ", length(chain),
        ",  simulations = ", chain.n_simulations)
println("acceptance rate = ", round(acceptance_rate(chain); digits = 3))
println()
println("parameter      truth    ABC-MCMC posterior mean ± sd")
println("──────────────────────────────────────────────────────")
println("a", "        ", rpad(A_TRUE, 7), "  ", round(wmean(p -> p[1]); digits = 3),
        " ± ", round(wsd(p -> p[1]); digits = 3))
println("b", "        ", rpad(B_TRUE, 7), "  ", round(wmean(p -> p[2]); digits = 3),
        " ± ", round(wsd(p -> p[2]); digits = 3))

# --- reproducibility ---------------------------------------------------------
# Same configuration and seed ⇒ bit-identical chain.
chain2 = run_mcmc(cfg, s_obs; n_samples = N_SAMPLES, ε = EPS, base_seed = 2024)
@assert chain.state.θ == chain2.state.θ "ABC-MCMC chain is not reproducible"
@assert chain.accepted == chain2.accepted
println("\n✓ reproducible: the chain replayed exactly from base_seed = ",
        chain.base_seed)

# --- the chain is an ABCSMCState, so downstream tools just work --------------
adjusted = adjusted_posterior(chain.state; observed_summary = s_obs, method = :ols)
adj_mean(f) = sum(adjusted.w .* [f(p) for p in adjusted.θ])
println("regression-adjusted posterior mean (OLS): a = ",
        round(adj_mean(p -> p[1]); digits = 3),
        ", b = ", round(adj_mean(p -> p[2]); digits = 3))

# --- optional marginal-posterior figure (plotting API + Plots.jl) ------------
const PLOTDIR = normpath(joinpath(@__DIR__, "..", "repro", "plots"))

have_plots = try
    Base.eval(Main, :(using Plots))
    true
catch
    false
end

if isdefined(SequentialABC, :plot_marginals) && have_plots
    mkpath(PLOTDIR)
    fig = SequentialABC.plot_marginals(chain.state; labels = ["a", "b"], backend = :plots)
    path = joinpath(PLOTDIR, "mcmc_marginals.png")
    Plots.savefig(fig, path)
    println("saved ", path)
elseif isdefined(SequentialABC, :plot_marginals)
    println("(Plots.jl missing — skipping the marginal figure; ",
            "run with an environment that provides Plots)")
else
    println("(plotting API not present — skipping the marginal figure)")
end

@assert length(chain) == N_SAMPLES + 1
println("\n✓ two-parameter ABC-MCMC example completed successfully")
