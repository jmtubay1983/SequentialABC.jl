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
Filter = x -> occursin("initialize!", string(x)) || occursin("run_generation!", string(x)) || occursin("run_smc!", string(x))
```

## Deterministic RNG

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("make_particle_seed", string(x))
```

## Checkpointing

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("save_checkpoint", string(x)) || occursin("list_checkpoints", string(x)) || occursin("load_checkpoint", string(x)) || occursin("load_latest_state", string(x))
```
