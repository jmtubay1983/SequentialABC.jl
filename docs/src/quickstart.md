# Quickstart

Run your first ABC‑SMC inference in a few lines of Julia. This guide covers:

1. Defining the model and prior
2. Building the config
3. Running the sampler
4. Checkpointing and restarting

## A continuous (logistic growth) example

Here we infer the growth rate \(r\) of a logistic model from a noisy observation of the population at time \(t=3\).

```julia
using SequentialABC
using Distributions
using Random
using Statistics
using LinearAlgebra

# --- 1. Prior and transform -------------------------------------------------
# Prior: r ~ Uniform(0, 1)
prior(θ_raw) = logpdf(Uniform(0.0, 1.0), θ_raw[1])

# Sample from prior
sample_prior(rng) = [rand(rng, Uniform(0.0, 1.0))]

# Transform: raw space = model space (identity here)
transform(θ_raw) = θ_raw

# --- 2. Distance ------------------------------------------------------------
function distance(sim_summaries, obs_summaries)
    return abs(sim_summaries[1] - obs_summaries[1])
end

# --- 3. Simulator -----------------------------------------------------------
# Logistic growth: N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))
function logistic_model(θ)
    r = θ[1]
    K = 100.0
    N0 = 1.0
    t = 3.0
    return K / (1 + ((K - N0) / N0) * exp(-r * t))
end

sim = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> logistic_model(θ),
    summary = x -> [x],
)

# --- 4. Configuration -------------------------------------------------------
Σ = [0.01;;]  # perturbation covariance (1×1 matrix)
cfg = ABCSMCConfig(
    prior, sample_prior, transform, distance,
    GaussianKernel(), Σ, sim,
)

# --- 5. Run ABC‑SMC ---------------------------------------------------------
N = 100        # population size
T = 5          # generations, including generation 1
base_seed = 42 # deterministic base seed

# Observed summary (seeded — the global RNG is never used by this package)
s_obs = [logistic_model([0.8]) + 0.1 * randn(MersenneTwister(2024))]  # noisy observation

schedule = QuantileSchedule(0.5)  # adaptive ε: median of accepted distances

state = run_smc(cfg, s_obs; N=N, T=T, schedule=schedule, base_seed=base_seed)

println("Final ε    = ", state.ε)
println("Final ESS  = ", state.ess)
println("Generation = ", state.gen)
```

`run_smc` builds the state, samples generation 1 and advances to generation `T`
for you; it mirrors `run_mcmc(cfg, s_obs; n_samples, ε)`. By default generation 1
is a plain prior sample (`ε0 = Inf`), and the schedule fixes the tolerance from
generation 2 onwards. Pass a finite `ε0` to start from a tighter, pre‑filtered
population instead, and `Tθ` to override the inferred parameter type (it is
otherwise obtained from one discarded prior draw).

## Checkpointing and restart

For long runs on an HPC cluster, enable checkpointing:

```julia
state = run_smc(cfg, s_obs; N=N, T=20, schedule=schedule,
                checkpoint_dir="checkpoints/")
```

If the job is preempted, resume from the latest checkpoint — note this uses the
state‑based [`run_smc!`](@ref), which continues from `state.gen + 1` rather than
starting over:

```julia
state = load_latest_state("checkpoints/", cfg)
run_smc!(state, s_obs; T=20, schedule=schedule, checkpoint_dir="checkpoints/")
```

The resumed run is **bit-identical** to an uninterrupted one — same particles, same weights, same ε trajectory.

## Agents.jl example (optional)

SequentialABC.jl also supports agent-based models via [`Agents.jl`](https://juliaagents.github.io/Agents.jl/stable/):

```julia
using Agents

# Build an Agents.jl model from θ
model_builder(θ, rng) = ...  # your ABM code

# Advance one step
step!(model, rng) = ...  # your step rule

# Extract summary statistic
summary(model) = ...  # e.g. segregation index

sim = AgentsSimulator(
    model_builder = model_builder,
    step! = step!,
    summary = summary,
    steps = 100,
)
```

The full Schelling segregation example is in `examples/agents_schelling.jl`. It runs only if `Agents.jl` is installed; otherwise it is skipped gracefully.

## Key design principles

- **No global RNG**: Every random draw uses an explicitly constructed `MersenneTwister`.
- **Deterministic seeds**: `make_particle_seed(gen, i, attempt, role, base_seed)` ensures full replay.
- **Raw parameter space**: Particles are stored where the prior is defined; the transform is applied only at simulation time.
- **Checkpoints**: Written to `.tmp` then renamed, so a kill before the rename leaves the last checkpoint intact (a stale `.tmp` is ignored by the loader). The file is native to the Julia version that wrote it and carries no checksum — a damaged checkpoint is detected when it fails to load.