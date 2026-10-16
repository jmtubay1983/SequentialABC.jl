# Example: the streaming (online) covariance estimator feeding AdaptiveKernel.
#
#   julia --project=. examples/online_kernel_example.jl
#
# `AdaptiveKernel` can be adapted in two ways:
#
#   * batch      — hand it the whole population:      update!(kernel, state.θ, state.w)
#   * streaming  — feed particles one at a time:      update!(kernel, θ_i, w_i) … commit!(kernel)
#
# The streaming form needs O(d²) memory instead of a copy of the population and
# produces the *same* estimate, because `OnlineCovEstimator` uses the same
# reliability-weighted normalisation as `estimate_covariance`. This script shows
# the parity first, then runs the same ABC-SMC problem with both forms.
#
# Everything is deterministic (fixed seeds) and writes no files.

using SequentialABC
using Distributions
using Random
using Statistics
using LinearAlgebra

# --- toy model:  s = θ + N(0, σ²),  θ ~ N(0, 2²)² ----------------------------
const NOISE = 0.5
const Σ0    = [4.0 0.0; 0.0 4.0]        # deliberately badly scaled start
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
θ_true = [0.8, -0.4]                                    # s_obs estimates these

make_cfg(Σ) =
    ABCSMCConfig(prior, sample_prior, identity, distance, GaussianKernel(), Σ, simulator)

# --- 1. parity: streaming vs batch estimate ---------------------------------
rng = MersenneTwister(2027)
Σtrue = [1.0 0.6; 0.6 2.0]
X = permutedims(rand(rng, MvNormal([1.0, -2.0], Σtrue), 20_000))

est = OnlineCovEstimator(2; shrink = 0.0)
for i in 1:size(X, 1)
    update!(est, X[i, :])                               # one particle at a time
end
Σonline = cov(est)
Σbatch = estimate_covariance(X; shrink = 0.0)

println("streaming vs batch covariance estimator")
println("─────────────────────────────────────────────────")
println("particles                ", est.n)
println("sample matrix memory     ", size(X, 1), " × ", size(X, 2), " Float64  (",
        round(sizeof(X) / 1024; digits = 1), " KiB kept alive)")
println("estimator memory         d + d² + 3 Float64 = ",
        round(sizeof(est) / 1024; digits = 3), " KiB")
println("max |Σ_streaming − Σ_batch| ", maximum(abs.(Σonline .- Σbatch)))
println()

# --- 2. the same ABC-SMC run, adapted batch-wise and streamed ---------------
function run_smc(cfg, s_obs; N, T, q, base_seed, streaming::Bool)
    kernel = AdaptiveKernel(cfg.Σ; rule = :beaumont)
    state = make_state(cfg, Vector{Float64}, N, 4.0, base_seed)
    initialize!(state, s_obs)

    trace = NamedTuple[]
    for _ in 2:T
        if streaming
            for i in 1:N
                update!(kernel, state.θ[i], state.w[i])   # O(d²) memory, one pass
            end
            commit!(kernel)                               # adopt Σ, reset accumulator
        else
            update!(kernel, state.θ, state.w)             # whole population at once
        end
        state.cfg = with_covariance(state.cfg, kernel)

        run_generation!(state, s_obs; ε = epsilon_quantile(state, q))
        push!(trace, (gen = state.gen, ε = state.ε, sims = state.n_attempts,
                      σ2 = copy(diag(cov(kernel))), updates = kernel.n_updates))
    end
    return (state = state, kernel = kernel, trace = trace)
end

wmean(st, f) = sum(st.w .* [f(p) for p in st.θ])

batch = run_smc(make_cfg(Σ0), s_obs;
                N = N_POP, T = T_GEN, q = Q, base_seed = 42, streaming = false)
streamed = run_smc(make_cfg(Σ0), s_obs;
                   N = N_POP, T = T_GEN, q = Q, base_seed = 42, streaming = true)

println("ABC-SMC, ", T_GEN, " generations, N = ", N_POP,
        ", initial Σ0 = ", Σ0[1, 1], "·I, adaptive rule :beaumont")
println("per-generation trace of the streaming run")
println("─────────────────────────────────────────────────")
println("gen    ε        diag Σ (streaming)      simulations   updates")
for t in streamed.trace
    println(rpad(t.gen, 7), rpad(round(t.ε; digits = 3), 9),
            rpad(string(round.(t.σ2; digits = 4)), 24),
            rpad(t.sims, 14), t.updates)
end
println()

println("streaming vs batch adaptation, same seeds")
println("─────────────────────────────────────────────────")
println("                        batch          streaming")
println("simulations             ", rpad(batch.trace[end].sims, 15), streamed.trace[end].sims)
println("final ε                 ", rpad(round(batch.state.ε; digits = 3), 15),
        round(streamed.state.ε; digits = 3))
for (i, name) in enumerate(("θ₁", "θ₂"))
    println("posterior mean ", name, "        ",
            rpad(round(wmean(batch.state, p -> p[i]); digits = 4), 15),
            round(wmean(streamed.state, p -> p[i]); digits = 4))
end
println("Σ agrees to             ",
        maximum(abs.(cov(batch.kernel) .- cov(streamed.kernel))))
println("(s_obs = ", s_obs, ", θ_true = ", θ_true,
        "; ABC targets the observed summary)")
println()

# --- 3. the streamed covariance is a valid ABC-MCMC proposal ----------------
ε_final = streamed.state.ε
chain = run_mcmc(with_covariance(make_cfg(Σ0), streamed.kernel), s_obs;
                 n_samples = 1000, ε = ε_final, base_seed = 7)
println("ABC-MCMC with the streamed covariance at ε = ", round(ε_final; digits = 3))
println("acceptance rate         ", round(acceptance_rate(chain); digits = 3))

# --- assertions ---------------------------------------------------------------
@assert est.n == 20_000
@assert maximum(abs.(Σonline .- Σbatch)) < 1e-10
@assert streamed.kernel.n_updates == T_GEN - 1
@assert isposdef(Symmetric(cov(streamed.kernel)))
@assert streamed.trace[end].sims == batch.trace[end].sims
@assert maximum(abs.(cov(streamed.kernel) .- cov(batch.kernel))) < 1e-8
@assert all(abs(wmean(streamed.state, p -> p[i]) - wmean(batch.state, p -> p[i])) < 1e-9
            for i in 1:2)
@assert acceptance_rate(chain) > 0.0
println("\n✓ online-kernel example completed successfully")
