# API Reference

All types and functions exported by `SequentialABC.jl`.

## Sampler configuration

### `ABCSMCConfig`

```julia
ABCSMCConfig(prior, sample_prior, transform, distance, kernel, Σ, simulator)
```

Configuration struct for the Toni et al. (2009) ABC‑SMC sampler.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `prior` | `Function` | Log-density \(\pi(\theta_{\text{raw}})\) on the raw parameter space |
| `sample_prior` | `Function` | Sampler `(rng) → θ_raw` from the prior (generation‑1 init) |
| `transform` | `Function` | \(\theta_{\text{raw}} \to \theta_{\text{model}}\) parameter map |
| `distance` | `Function` | `(summary_sim, summary_obs) → Real` |
| `kernel` | `GaussianKernel` | Perturbation kernel: log \(K(\theta_{\text{new}} \mid \theta_{\text{old}}, \Sigma)\) |
| `Σ` | `Matrix{Float64}` | Perturbation covariance |
| `Σ_inv` | `Matrix{Float64}` | Precomputed inverse covariance (computed automatically) |
| `logdetΣ` | `Float64` | Log-determinant of Σ (computed automatically) |
| `simulator` | `ABCSimulator` | Simulator backend |

The constructor automatically computes `Σ_inv = inv(Σ)` and `logdetΣ = logdet(Σ)`.

## Sampler state

### `ABCSMCState`

```julia
ABCSMCState(cfg, θ, w, summaries, dists, ε, gen, ess, base_seed, n_attempts, n_resamples)
```

Mutable sampler state.

**Fields:**

| Field | Description |
|-------|-------------|
| `cfg` | `ABCSMCConfig` |
| `θ` | Particles at current generation (raw parameter space) |
| `w` | Normalized importance weights |
| `summaries` | Cached simulation summaries |
| `dists` | Cached distances \(d(s_i, s_{\text{obs}})\) |
| `ε` | Current tolerance |
| `gen` | Current generation index |
| `ess` | Effective sample size |
| `base_seed` | Base RNG seed for deterministic replication |
| `n_attempts` | Accepted-particle attempts this generation |
| `n_resamples` | Cumulative population resamplings |

### `make_state`

```julia
make_state(cfg::ABCSMCConfig, ::Type{Tθ}, N::Int, ε0::Float64, base_seed::Int)
```


## Simulator backends

### `ABCSimulator`

Abstract type for simulator backends.

### `ContinuousSimulator`

```julia
ContinuousSimulator(model_builder, evolve, summary)
```

Continuous (ODE/PDE/SSA) backend.

- `model_builder(θ)` — build the initial state/problem from parameters
- `evolve(problem, rng)` — run to completion, return final state
- `summary(final_state)` — extract summary statistic

### `AgentsSimulator`

```julia
AgentsSimulator(model_builder, step!, summary, steps)
```

Agents.jl backend for agent-based models.

- `model_builder(θ, rng)` — build the model on the given RNG
- `step!(model, rng)` — advance one step
- `summary(model)` — extract summary after `steps` steps
- `steps` — number of steps per simulation

Keyword constructors are also provided:
```julia
ContinuousSimulator(; model_builder, evolve, summary)
AgentsSimulator(; model_builder, step!, summary, steps)
```

## Perturbation kernel

### `GaussianKernel`

```julia
GaussianKernel()
```

Gaussian perturbation kernel: \(\theta_{\text{new}} = \theta_{\text{old}} + \epsilon\), \(\epsilon \sim \mathcal{N}(0, \Sigma)\).

**Call:**

```julia
(k::GaussianKernel)(θ_new, θ_old, Σ_inv, logdetΣ) -> Float64
```

Returns the log-density of \(\mathcal{N}(\theta_{\text{old}}, \Sigma)\) at \(\theta_{\text{new}}\), numerically identical to `Distributions.logpdf(MvNormal(θ_old, Σ), θ_new)`.

**Perturb:**

```julia
perturb(k::GaussianKernel, θ_old::AbstractVector, Σ::Matrix{Float64}, rng::AbstractRNG) -> Vector{Float64}
```

Draw \(\theta_{\text{new}} \sim \mathcal{N}(\theta_{\text{old}}, \Sigma)\) using Cholesky factorization.

## Importance weights and ESS

### `importance_weight`

```julia
importance_weight(cfg::ABCSMCConfig, θ_prop, θ_prev, w_prev) -> Float64
```

Toni et al. (2009) importance weight:

\[
w \propto \frac{\pi(\theta_{\text{prop}})}{\sum_j w_{\text{prev}}[j] \cdot K(\theta_{\text{prop}} \mid \theta_{\text{prev}}[j], \Sigma)}
\]

Implemented in log-space for numerical stability.

### `ess`

```julia
ess(w::AbstractVector{<:Real}) -> Float64
```

Effective sample size: \( \text{ESS} = 1 / \sum_i w_i^2 \).

For \(N=1\), ESS is defined as 1.0.

### `systematic_resample`

```julia
systematic_resample(w::AbstractVector{<:Real}, rng::AbstractRNG) -> Vector{Int}
```

Low-variance systematic resampling. Returns ancestor indices. The RNG is passed explicitly — never the global RNG.

## Tolerance schedules

Schedules are callables `(state, t) → ε_t` used by `run_smc!`.

### `FixedSchedule`

```julia
FixedSchedule(εs::AbstractVector{<:Real})
```

Hand-picked, non-increasing tolerance schedule. Generation \(t\) uses `εs[t]`. Generation 1 uses `state.ε` (the initial tolerance), so index \(t \ge 2\) entries are consumed for a \(T\)-generation run started from gen 0.

Throws `ArgumentError` if the schedule is not non-increasing.

### `QuantileSchedule`

```julia
QuantileSchedule(q::Real)
```

Adaptive tolerance: each generation uses the \(q\)-quantile of the previous population's accepted distances. \(q \in (0, 1]\), typically \(q = 0.5\) (median).

Monotone non-increasing by construction.

### `epsilon_quantile`

```julia
epsilon_quantile(state::ABCSMCState, q::Real) -> Float64
```

The \(q\)-quantile of `state.dists`, guarded by `min(ε_new, state.ε)`.

## Sampler drivers

### `initialize!`

```julia
initialize!(state::ABCSMCState, summary_obs; max_attempts::Int=10_000)
```

Generate generation‑1 from the prior via redraw-until-accepted. Sets uniform weights, `gen = 1`, `ess = N`.

### `run_generation!`

```julia
run_generation!(state::ABCSMCState, summary_obs; ε::Float64, M::Int=1, max_attempts::Int=10_000)
```

Run a single generation: resample ancestor, perturb, simulate, accept/reject, compute importance weight, normalize, optionally resample if ESS < N/2.

### `run_smc!`

```julia
run_smc!(state::ABCSMCState, summary_obs; T::Int, schedule, M::Int=1, max_attempts::Int=10_000, checkpoint_dir::Union{Nothing,AbstractString}=nothing)
```

Run ABC‑SMC for \(T\) generations. If `checkpoint_dir` is given, atomic checkpoints are written after initialization and after every generation.

## Deterministic RNG

### `make_particle_seed`

```julia
make_particle_seed(gen::Int, i::Int, attempt::Int, m::Int, base::Int) -> Int
```

Deterministic seed for particle simulation. A pure function of `(gen, i, attempt, m, base)` — full replay across restarts and nodes.

The `attempt` counter ensures redraw-until-accepted never replays an identical rejected proposal.

## Checkpointing

### `save_checkpoint`

```julia
save_checkpoint(state::ABCSMCState, dir::AbstractString) -> String
```

Atomically checkpoint the sampler to `dir/state_gen_XXX.jls`. Only data is serialized (never `cfg` and its closures), making checkpoints portable across Julia versions and code revisions.

Atomicity: written to `.tmp` then renamed — POSIX rename is atomic on the same filesystem.

### `list_checkpoints`

```julia
list_checkpoints(dir::AbstractString) -> Vector{Tuple{Int, String}}
```

Sorted `(gen, path)` pairs of all valid checkpoints in `dir`. Stale `.tmp` files are ignored.

### `load_checkpoint`

```julia
load_checkpoint(path::AbstractString, cfg) -> ABCSMCState
```

Load a single checkpoint file and reattach `cfg`.

### `load_latest_state`

```julia
load_latest_state(dir::AbstractString, cfg) -> ABCSMCState
```

Load the highest-generation checkpoint from `dir`. Resume with `run_smc!`, which continues from `state.gen + 1` using stored deterministic seeds — bit-identical to an uninterrupted run.

Throws an error if no checkpoints are found.
