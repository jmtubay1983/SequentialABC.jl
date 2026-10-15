# Example: continuous (ODE) backend
#
# Infer the growth rate r of logistic growth from a single noisy endpoint.
#
#   dN/dt = r · N · (1 − N/K),   N(0) = N0 = 10,   K = 100,   t ∈ [0, 3]
#
# Synthetic observation: N(3) under true r = 0.8 with multiplicative noise.
# (We use the early, unsaturated phase where N(3) identifies r well; at t ≈ 10
# the trajectory has saturated at K and the endpoint barely constrains r.)
# The ABC-SMC posterior over r should concentrate near 0.8.
#
# Run from the package root:
#   julia --project=. examples/continuous_logistic.jl

using SequentialABC
using Distributions
using Random
using Printf

# --- true model -------------------------------------------------------------
const K        = 100.0
const N0       = 10.0
const T_FINAL  = 3.0
const R_TRUE   = 0.8
const NOISE_SD = 0.05

# RK4 integration of the logistic ODE to T_FINAL (scalar form)
function logistic_endpoint(r)
    dt = 0.01
    f(u) = r * u * (1.0 - u / K)
    u = N0
    for _ in 1:round(Int, T_FINAL / dt)
        k1 = f(u)
        k2 = f(u + dt / 2 * k1)
        k3 = f(u + dt / 2 * k2)
        k4 = f(u + dt * k3)
        u += dt / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return u
end

# --- ABC components ---------------------------------------------------------
prior      = Uniform(0.1, 1.5)
transform  = θ -> θ
distance   = (s, s_obs) -> abs(s - s_obs)

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> logistic_endpoint(θ[1]) * exp(NOISE_SD * randn(rng)),
    summary = u -> u,
)

# Synthetic "observed" endpoint (its own seed — independent of sampler seeds)
y_obs = simulator.evolve([R_TRUE], MersenneTwister(2024))
@printf("Observed endpoint: N(3) = %.3f  (true r = %.2f)\n\n", y_obs, R_TRUE)

cfg = ABCSMCConfig(
    θ -> (0.1 <= θ[1] <= 1.5 ? logpdf(prior, θ[1]) : -Inf),
    rng -> [rand(rng, prior)],
    transform, distance,
    GaussianKernel(), [0.05;;], simulator,
)

# --- run ABC-SMC with adaptive ε and checkpointing --------------------------
N = 150
ckpt = mktempdir()
state = make_state(cfg, Vector{Float64}, N, 5.0, 42)
run_smc!(state, y_obs; T=6, schedule=QuantileSchedule(0.5), checkpoint_dir=ckpt)

@printf("Generation  ε         ESS     attempts\n")
@printf("───────────────────────────────────────\n")
# ε/ESS history is in the checkpoints
for (g, path) in list_checkpoints(ckpt)
    st = load_checkpoint(path, cfg)
    @printf("    %d      %.4f   %6.1f   %d\n", g, st.ε, st.ess, st.n_attempts)
end

r_mean = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
r_var  = sum(state.w[i] * (state.θ[i][1] - r_mean)^2 for i in eachindex(state.w))
@printf("\nPosterior r: mean = %.3f, sd = %.3f   (true = %.2f)\n", r_mean, sqrt(r_var), R_TRUE)
@printf("Final ε = %.4f, ESS = %.1f, resamplings = %d\n", state.ε, state.ess, state.n_resamples)

@assert abs(r_mean - R_TRUE) < 0.3 "posterior mean too far from truth: $r_mean"
println("\n✓ Continuous-backend example completed successfully")
