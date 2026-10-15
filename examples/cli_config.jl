# Config file for bin/sequentialabc-run.jl — logistic-growth example (small/fast).
#
# Must evaluate to a NamedTuple with keys:
#   cfg, summary_obs, N, ε0, base_seed, T, schedule
# optional: Tθ, M, max_attempts
#
# Local test (no Slurm needed):
#   julia --project=. bin/sequentialabc-run.jl \
#       --config examples/cli_config.jl --checkpoint-dir /tmp/abc_ckpt --target-gen 2

using SequentialABC
using Distributions
using Random

const K_     = 100.0
const N0_    = 10.0
const TF_    = 3.0
const NOISE_ = 0.05

# RK4 integration of dN/dt = rN(1−N/K) to TF_ (scalar form)
function logistic_endpoint_(r)
    dt = 0.01
    f(u) = r * u * (1.0 - u / K_)
    u = N0_
    for _ in 1:round(Int, TF_ / dt)
        k1 = f(u)
        k2 = f(u + dt / 2 * k1)
        k3 = f(u + dt / 2 * k2)
        k4 = f(u + dt * k3)
        u += dt / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return u
end

prior_ = Uniform(0.1, 1.5)

simulator_ = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> logistic_endpoint_(θ[1]) * exp(NOISE_ * randn(rng)),
    summary = u -> u,
)

# Synthetic observation: true r = 0.8 with noise, fixed seed (reproducible)
y_obs_ = logistic_endpoint_(0.8) * exp(NOISE_ * randn(MersenneTwister(2024)))

(
    cfg = ABCSMCConfig(
        θ -> (0.1 <= θ[1] <= 1.5 ? logpdf(prior_, θ[1]) : -Inf),
        rng -> [rand(rng, prior_)],
        identity,
        (s, s_obs) -> abs(s - s_obs),
        GaussianKernel(), [0.05;;], simulator_,
    ),
    summary_obs = y_obs_,
    N = 60,
    ε0 = 5.0,
    base_seed = 42,
    T = 4,
    schedule = QuantileSchedule(0.5),
)
