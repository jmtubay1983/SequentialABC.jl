# Example: adaptive multivariate kernels for ABC-SMC and ABC-MCMC.
#
#   julia --project=. examples/adaptive_kernel_example.jl
#
# Both engines move particles with a Gaussian random walk θ' ~ 𝒩(θ, Σ). Picking
# Σ by hand is awkward: too small and the population cannot move, too large and
# almost every proposal misses the acceptance region. This script starts both
# engines from a deliberately badly scaled Σ, adapts it from the particle
# population with `AdaptiveKernel` / `update!`, and reports what changes.
#
# Everything is deterministic (fixed seeds) and writes no files.

using SequentialABC
using Distributions
using Random
using Statistics
using LinearAlgebra

# --- toy model:  s = θ + N(0, σ²),  θ ~ N(0, 2²)² ----------------------------
# The summary is the parameter plus noise, so the ABC posterior is close to a
# Gaussian centred on s_obs with a variance of roughly σ²; this keeps the
# comparison easy to interpret.
const NOISE = 0.5
const Σ0    = [4.0 0.0; 0.0 4.0]        # deliberately ~20× too large
const Q     = 0.6                       # tolerance quantile per generation
const N_POP = 400
const T_GEN = 6

prior(θ) = logpdf(Normal(0.0, 2.0), θ[1]) + logpdf(Normal(0.0, 2.0), θ[2])
sample_prior(rng) = 2.0 .* randn(rng, 2)

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ NOISE .* randn(rng, 2),
    summary = identity,
)

distance(s, so) = sqrt(sum(abs2, s .- so))

s_obs = [0.8, -0.4]                                     # observed summary
θ_true = [0.8, -0.4]                                    # s_obs is an unbiased estimate of these

function make_cfg(Σ)
    ABCSMCConfig(prior, sample_prior, identity, distance, GaussianKernel(), Σ, simulator)
end

# --- one ABC-SMC driver, optionally adapting the kernel every generation -----
function run_smc(cfg, s_obs; N, T, q, base_seed, adaptive::Bool, kernel = nothing)
    state = make_state(cfg, Vector{Float64}, N, 4.0, base_seed)
    initialize!(state, s_obs)
    cumulative = Int[state.n_attempts]
    for _ in 2:T
        if adaptive
            update!(kernel, state.θ, state.w)          # Σ ← 2·Cov(population)
            state.cfg = with_covariance(state.cfg, kernel)
        end
        run_generation!(state, s_obs; ε = epsilon_quantile(state, q))
        push!(cumulative, state.n_attempts)
    end
    return (state = state, cumulative = cumulative)
end

wmean(st, f) = sum(st.w .* [f(p) for p in st.θ])
wsd(st, f) = (m = wmean(st, f); sqrt(sum(st.w .* ([f(p) for p in st.θ] .- m) .^ 2)))

fixed = run_smc(make_cfg(Σ0), s_obs; N = N_POP, T = T_GEN, q = Q,
                base_seed = 42, adaptive = false)

kernel = AdaptiveKernel(Σ0; rule = :beaumont)
adaptive = run_smc(make_cfg(Σ0), s_obs; N = N_POP, T = T_GEN, q = Q,
                   base_seed = 42, adaptive = true, kernel = kernel)

println("ABC-SMC on the same model, same seeds, same initial Σ0 = ", Σ0[1, 1], "·I")
println("observed summary s_obs = ", s_obs, ",  T = ", T_GEN, " generations, N = ", N_POP)
println()
println("                         fixed Σ0      adaptive Σ")
println("─────────────────────────────────────────────────")
println("simulations             ", rpad(fixed.cumulative[end], 12),
        adaptive.cumulative[end])
println("final ε                 ", rpad(round(fixed.state.ε; digits = 3), 12),
        round(adaptive.state.ε; digits = 3))
for (i, name) in enumerate(("θ₁", "θ₂"))
    println("posterior mean ", name, "        ",
            rpad(round(wmean(fixed.state, p -> p[i]); digits = 3), 12),
            round(wmean(adaptive.state, p -> p[i]); digits = 3))
    println("posterior sd   ", name, "        ",
            rpad(round(wsd(fixed.state, p -> p[i]); digits = 3), 12),
            round(wsd(adaptive.state, p -> p[i]); digits = 3))
end
gap(st) = hypot(wmean(st, p -> p[1]) - s_obs[1], wmean(st, p -> p[2]) - s_obs[2])
println("|mean − s_obs|          ", rpad(round(gap(fixed.state); digits = 3), 12),
        round(gap(adaptive.state); digits = 3))
println("kernel updates          ", rpad(0, 12), kernel.n_updates)
println("(θ_true = ", θ_true, "; the ABC posterior targets s_obs, so the two")
println(" runs should agree on where they land, not on how much they cost)")

Σhat = kernel.Σ
println()
println("adapted covariance after ", kernel.n_updates, " updates:")
for i in 1:2
    println("   ", round.(Σhat[i, :]; digits = 4))
end
println("population covariance at the end of the adaptive run:")
println("   ", round.(estimate_covariance(adaptive.state.θ, adaptive.state.w); digits = 4))

# --- the same covariance, used as the ABC-MCMC proposal ----------------------
function mcmc_acceptance(cfg, ε, seed)
    chain = run_mcmc(cfg, s_obs; n_samples = 1000, ε = ε, base_seed = seed)
    return chain, acceptance_rate(chain)
end

ε_final = adaptive.state.ε
_, acc0 = mcmc_acceptance(make_cfg(Σ0), ε_final, 7)
_, accA = mcmc_acceptance(with_covariance(make_cfg(Σ0), kernel), ε_final, 7)

println()
println("ABC-MCMC at ε = ", round(ε_final; digits = 3), " (1000 iterations each)")
println("─────────────────────────────────────────────────")
println("proposal Σ0              acceptance rate ", round(acc0; digits = 3))
println("proposal adaptive Σ      acceptance rate ", round(accA; digits = 3))

@assert adaptive.cumulative[end] < fixed.cumulative[end]
@assert accA > acc0
@assert kernel.n_updates == T_GEN - 1
println("\n✓ adaptive-kernel example completed successfully")
