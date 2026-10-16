# Example: regression adjustment postprocessing for ABC posteriors.
#
#   julia --project=. examples/regression_adjustment.jl
#
# A toy linear-Gaussian problem whose ABC posterior is biased by the finite
# tolerance ε. Regression adjustment (OLS and local linear) removes the leading
# part of that bias. When the plotting API and Plots.jl are available the script
# also writes a before/after marginal-posterior figure to repro/plots/.

using SequentialABC
using Distributions
using Random
using Statistics

# --- toy model:  s = θ + N(0, σ²) --------------------------------------------
const THETA_TRUE = 2.0
const NOISE_SD   = 0.5
const N          = 200

prior = Uniform(-1.0, 5.0)

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ[1] + NOISE_SD * randn(rng),
    summary = identity,
)

s_obs = THETA_TRUE + NOISE_SD * randn(MersenneTwister(2024))

cfg = ABCSMCConfig(
    θ -> logpdf(prior, θ[1]),
    rng -> [rand(rng, prior)],
    identity,
    (s, so) -> abs(s - so),
    GaussianKernel(),
    [0.2;;],
    simulator,
)

# --- a tiny ABC-SMC run ------------------------------------------------------
state = make_state(cfg, Vector{Float64}, N, 4.0, 42)
run_smc!(state, s_obs; T = 3, schedule = QuantileSchedule(0.4))

θ = [p[1] for p in state.θ]
w = state.w
wmean(x) = sum(w .* x)
wsd(x) = (m = wmean(x); sqrt(sum(w .* (x .- m) .^ 2)))

# --- regression adjustment ---------------------------------------------------
# `summaries_fn` maps one stored summary to its vector of regressors.
summaries_fn = s -> [s]

adj_ols = adjusted_posterior(state; summaries_fn = summaries_fn,
                             observed_summary = s_obs, method = :ols)
adj_ll  = adjusted_posterior(state; summaries_fn = summaries_fn,
                             observed_summary = s_obs, method = :local_linear)

θ_ols = [p[1] for p in adj_ols.θ]
θ_ll  = [p[1] for p in adj_ll.θ]

# low-level entry points (plain matrices in, plain matrix out) must agree
S = reshape(Float64.(state.summaries), :, 1)
Θ = reshape(Float64.(θ), :, 1)
@assert vec(ols_adjust(Θ, S, [s_obs]; w = w)) ≈ θ_ols
@assert vec(local_linear_adjust(Θ, S, [s_obs]; w = w, kernel = :epanechnikov)) ≈ θ_ll

println("observed summary s_obs = ", round(s_obs; digits = 3),
        "   (true θ = ", THETA_TRUE, ")")
println("final ε = ", round(state.ε; digits = 3), ",  N = ", N)
println()
println("method                 mean      sd")
println("──────────────────────────────────────")
println("unadjusted         ", rpad(round(wmean(θ); digits = 3), 7), "  ",
        round(wsd(θ); digits = 3))
println("OLS                ", rpad(round(wmean(θ_ols); digits = 3), 7), "  ",
        round(wsd(θ_ols); digits = 3))
println("local linear       ", rpad(round(wmean(θ_ll); digits = 3), 7), "  ",
        round(wsd(θ_ll); digits = 3))

# --- optional before/after figure (plotting API + Plots.jl) ------------------
const PLOTDIR = normpath(joinpath(@__DIR__, "..", "repro", "plots"))

have_plots = try
    Base.eval(Main, :(using Plots))
    true
catch
    false
end

if isdefined(SequentialABC, :plot_marginals) && have_plots
    mkpath(PLOTDIR)
    for (name, res) in (("ra_before", state), ("ra_after", adj_ll))
        fig = SequentialABC.plot_marginals(res; labels = ["θ"], backend = :plots)
        path = joinpath(PLOTDIR, name * ".png")
        Plots.savefig(fig, path)
        println("saved ", path)
    end
elseif isdefined(SequentialABC, :plot_marginals)
    println("(Plots.jl missing — skipping the before/after figures; ",
            "run with an environment that provides Plots)")
else
    println("(plotting API not present — skipping the before/after figures)")
end

@assert length(adj_ols.θ) == N && length(adj_ll.θ) == N
println("\n✓ regression-adjustment example completed successfully")
