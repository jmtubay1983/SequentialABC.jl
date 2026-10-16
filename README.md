# SequentialABC.jl

A small, project-owned **ABC-SMC** (Approximate Bayesian Computation with
Sequential Monte Carlo) package for Julia, implementing the algorithm of
[Toni et al. (2009)](https://royalsocietypublishing.org/doi/10.1098/rsif.2008.0172)
faithfully, with the practical features real simulation studies need:

- **Deterministic RNG architecture** — every stochastic step is a pure
  function of `(gen, particle, attempt, replicate, base_seed)`; bit-identical
  replay across runs, restarts, and Slurm nodes.
- **Redraw-until-accepted** populations (no zero-weight particles; ESS and
  population size are always well-defined).
- **Systematic resampling** when ESS < N/2.
- **Atomic, generation-indexed checkpointing** — restart-safe under job
  preemption; checkpoints store *data only*, so you can revise your simulator
  code between restarts.
- **Adaptive (quantile) and fixed ε schedules**.
- **Pluggable simulators** — one abstraction for agent-based (Agents.jl) and
  continuous (ODE/PDE/SSA) models.

Requires **Julia ≥ 1.11** (developed and tested on 1.13.2).

## Installation

```julia
using Pkg
Pkg.develop(path = "~/projects/SequentialABC")   # or your fork's path
```

## Quickstart — noisy Gaussian mean

Toy problem with an analytic posterior, showing the full API:

```julia
using SequentialABC, Distributions, Random

# Generative model: y | θ ~ N(θ, 1), prior θ ~ N(0, 5²), observed y = 1.0
const PRIOR  = Normal(0.0, 5.0)
const Y_OBS  = 1.0

sample_prior = rng -> [rand(rng, PRIOR)]              # particles are vectors
prior_logpdf = θ  -> logpdf(PRIOR, θ[1])
distance     = (s, s_obs) -> abs(s - s_obs)

simulator = ContinuousSimulator(
    model_builder = θ -> θ[1],
    evolve        = (μ, rng) -> μ + randn(rng),       # simulate one datum
    summary       = y -> y,
)

cfg = ABCSMCConfig(prior_logpdf, sample_prior, identity, distance,
                   GaussianKernel(), [0.5;;], simulator)

state = make_state(cfg, Vector{Float64}, 200, 3.0, 1234)
run_smc!(state, Y_OBS; T = 3, schedule = FixedSchedule([3.0, 1.0, 0.3]),
         checkpoint_dir = "checkpoints/")

mean_θ = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
@show mean_θ, state.ε, state.ess     # analytic posterior mean ≈ 0.96
```

Restart after a crash or preemption — the trajectory continues bit-identically:

```julia
state = load_latest_state("checkpoints/", cfg)
run_smc!(state, Y_OBS; T = 10, schedule = QuantileSchedule(0.5),
         checkpoint_dir = "checkpoints/")
```

## Core API

| Function | Purpose |
|---|---|
| `ABCSMCConfig(prior, sample_prior, transform, distance, kernel, Σ, simulator)` | Bundle the algorithm components. `prior` is a **log-density on the raw (population) space**; `transform` is applied only at simulation time; `Σ_inv`/`logdetΣ` are precomputed. |
| `make_state(cfg, Tθ, N, ε0, base_seed)` | Empty generation-0 state; `Tθ` is the particle element type (e.g. `Vector{Float64}`). |
| `initialize!(state, s_obs; max_attempts)` | Generation 1 from the prior, redraw-until-accepted, uniform weights. |
| `run_generation!(state, s_obs; ε, M, max_attempts)` | One ABC-SMC generation (ancestor resampling → perturb → simulate → accept/reweight → maybe resample). |
| `run_smc!(state, s_obs; T, schedule, M, max_attempts, checkpoint_dir)` | Full sequential run; checkpoints after every generation when `checkpoint_dir` is set. |
| `save_checkpoint(state, dir)` / `load_checkpoint(path, cfg)` / `load_latest_state(dir, cfg)` / `list_checkpoints(dir)` | Atomic (`tmp` → `rename`) generation-indexed checkpointing. Data-only: reattach `cfg` on load. |
| `FixedSchedule(εs)` / `QuantileSchedule(q)` / `epsilon_quantile(state, q)` | Callable `(state, t) -> ε_t` schedules. Toni-style fixed, or adaptive q-quantile of the previous population's distances (monotone by construction). |
| `systematic_resample(w, rng)` / `ess(w)` / `importance_weight(cfg, θ, θ_prev, w_prev)` | Low-variance resampling, ESS, and Toni et al. importance weights (log-space). |

### Simulator backends

```julia
# Continuous (ODE / PDE / SSA): model_builder(θ), evolve(problem, rng) -> final_state
ContinuousSimulator(model_builder, evolve, summary)

# Agents.jl: model_builder(θ, rng) builds the ABM ON the given rng
# (e.g. StandardABM(...; rng = rng)), step!(model, rng) advances one step
AgentsSimulator(model_builder, step!, summary, steps)
```

The Agents.jl backend keeps the whole particle simulation on one deterministic
rng stream: construction (`add_agent!`, initial placement) and every step

## Determinism / seed scheme

All randomness flows through seeded `MersenneTwister`s. The seed is a pure
function `make_particle_seed(gen, i, attempt, m, base_seed)` with distinct
roles for `m`, so no two roles ever collide:

| Role | `(gen, i, attempt, m)` |
|---|---|
| Generation-1 prior draws | `(1, i≥1, attempt≥1, 0)` |
| Ancestor resampling (gen ≥ 2) | `(gen, i≥1, attempt≥1, 0)` |
| Perturbation kernel | `(gen, i≥1, attempt≥1, −1)` |
| Simulation replicates | `(gen, i≥1, attempt≥1, m ≥ 1)` |
| Population resampling | `(gen, 0, 0, −2)` |

The `attempt` counter is what makes redraw-until-accepted replayable: each
rejected proposal gets a fresh, reproducible stream.

## Examples

Two validated end-to-end examples, each recovering its true parameter within
1 posterior sd:

```bash
# Continuous backend: logistic growth dN/dt = rN(1−N/K), infer r from a noisy
# early-phase endpoint.  → r = 0.771 ± 0.031 (true 0.80)
julia --project=. examples/continuous_logistic.jl

# Agents.jl backend: Schelling segregation, infer agents' tolerance from the
# segregation index.  → tol = 0.577 ± 0.071 (true 0.60)
julia --project=examples/agents_env -e 'using Pkg; Pkg.instantiate()'  # once
julia --project=examples/agents_env examples/agents_schelling.jl
```

Both print per-generation ε / ESS / attempt diagnostics from the checkpoints.

**Summary-statistic warning** (learned the hard way, twice): verify your
summary varies non-degenerately in θ before trusting an ABC posterior. A
saturated summary (e.g. "fraction happy" hitting 1.0, or an ODE endpoint at
carrying capacity) yields ε → 0 and a confident-looking posterior that is
actually prior-driven.

## Testing

```bash
julia --project=. test_load.jl
```

13 checks: kernel log-density vs `logpdf(MvNormal)` (1e-12), toy-model
inference vs analytic posterior, determinism (bit-identical reruns),
restartability (interrupted == uninterrupted), gen-1 uniform weights,
systematic-resampling unit tests, forced ESS-collapse resampling, checkpoint
round-trips and file hygiene, adaptive-ε monotonicity and restart equivalence.

Note: docs/Manifest.toml is ignored.

## Slurm

The deterministic seed scheme plus generation-indexed checkpoints give you a
trivially safe pattern: one job per generation-block, re-running the same
`run_smc!(…; checkpoint_dir=…)` call — a preempted or duplicated job either
no-ops (checkpoint already exists at that generation) or reproduces the
identical result. A dedicated CLI wrapper is planned (Step 5).

## References

Toni, Welch, Strelkowa, Ipsen & Stumpf (2009). *Approximate Bayesian
computation scheme for parameter inference and model selection in dynamical
systems.* J. R. Soc. Interface 6, 187–202.

## Roadmap

- [x] Phase 1 — core Toni ABC-SMC, deterministic seeds, redraw-until-accepted
- [x] Phase 2 Steps 1–4 — systematic resampling, checkpointing, adaptive ε, example models
- [ ] Step 5 — Slurm CLI
- [ ] Step 6 — formal `test/` suite (`Pkg.test`)
- [ ] Step 7 — documentation site / API docs

consume it in order. Agents.jl itself is **not** a dependency of
SequentialABC — see `examples/agents_env/` for a self-contained environment.
