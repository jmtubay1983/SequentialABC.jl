# API Reference

All types and functions exported by `SequentialABC.jl`.

Each `@autodocs` block uses `Private = false` so that only the exported (public)
API is rendered; internal helpers such as `perturb` are excluded.

## Sampler configuration

```@autodocs
Modules = [SequentialABC]
Order = [:type, :function]
Private = false
Filter = x -> occursin("ABCSMCConfig", string(x)) || occursin("make_state", string(x))
```

## Sampler state

```@autodocs
Modules = [SequentialABC]
Order = [:type]
Private = false
Filter = x -> occursin("ABCSMCState", string(x))
```

## Simulator backends

```@autodocs
Modules = [SequentialABC]
Order = [:type, :function]
Private = false
Filter = x -> occursin("ABCSimulator", string(x)) || occursin("ContinuousSimulator", string(x)) || occursin("AgentsSimulator", string(x))
```

## Perturbation kernel

```@autodocs
Modules = [SequentialABC]
Order = [:type, :function]
Private = false
Filter = x -> occursin("GaussianKernel", string(x))
```

## Importance weights and ESS

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("importance_weight", string(x)) || string(x) == "ess" || occursin("systematic_resample", string(x))
```

## Tolerance schedules

```@autodocs
Modules = [SequentialABC]
Order = [:type, :function]
Private = false
Filter = x -> occursin("FixedSchedule", string(x)) || occursin("QuantileSchedule", string(x)) || occursin("epsilon_quantile", string(x))
```

## Sampler drivers

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("initialize!", string(x)) || occursin("run_generation!", string(x)) || occursin("run_smc", string(x)) || occursin("run_pmc", string(x))
```

## Unified inference entrypoint

When the engine is chosen at run time — a config file, a job-array variable, a
`--engine` flag — [`run_inference`](@ref) is a named forwarder to the three
entrypoints above:

```julia
engine = Symbol(get(ENV, "ABC_ENGINE", "smc"))     # :smc | :mcmc | :pmc

result = run_inference(engine, cfg, s_obs; N = 1000, T = 5, base_seed = 2026)
```

| symbol | entrypoint | result type | keywords it accepts |
|---|---|---|---|
| `:smc`  | `run_smc`  | `ABCSMCState`  | `N`, `T`, `schedule`, `ε0`, `M`, `base_seed`, `checkpoint_dir` |
| `:mcmc` | `run_mcmc` | `ABCMCMCChain` | `n_samples`, `ε`, `thin`, `M`, `base_seed` |
| `:pmc`  | `run_pmc`  | `ABCSMCState`  | `N`, `T`, `εs`, `schedule`, `kernel_rule`, `M`, `base_seed` |

Keywords are **not** inspected by the dispatcher — it forwards them and each
engine validates its own, so the failure you get for a wrong keyword is the one
the engine itself raises. Note in particular that ABC-MCMC takes `n_samples`
rather than `N`: `run_inference(:mcmc, cfg, s_obs; N = 1000)` fails the same way
`run_mcmc(cfg, s_obs; N = 1000)` does.

Every call starts a fresh run; returning an `ABCSMCState` means the usual
downstream tools work on the result, and resuming a checkpointed population goes
through `run_smc!` / `run_pmc!` on a state from `load_latest_state`, not through
this dispatcher.

```@docs
run_inference
```

## Deterministic RNG

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("make_particle_seed", string(x))
```

```@docs
SEED_SCHEMA_VERSION
```

## Checkpointing

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("save_checkpoint", string(x)) || occursin("list_checkpoints", string(x)) || occursin("load_checkpoint", string(x)) || occursin("load_latest_state", string(x))
```
