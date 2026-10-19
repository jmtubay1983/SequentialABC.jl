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

## Adaptive distance scaling

Per-key inverse-distance scaling for ABC-SMC summaries, enabled with the
`adaptive` keyword of [`run_smc`](@ref) / [`run_smc!`](@ref). With the ordinary
Euclidean base distance (p = 2), both summaries are pre-scaled by the inverse
scale weights before their weighted Euclidean distance is evaluated. Pre-scaling
both sides by adaptive inverse-scale weights gives the same shape to the base
distance; if the base distance is already weighted (e.g. a `WeightedL2` or a
Mahalanobis distance), the two weightings compose (product/matrix composition)
— see the internal `adaptive_distance` docstring for details.

```julia
adaptive = AdaptiveDistanceConfig(
    StandardDeviationScale(),
    NumericalPolicy(constant_policy = :clamp),
)
state = run_smc(cfg, s_obs; N = 1000, T = 10, adaptive = adaptive)
```

Numerical policy (`floor`, `constant_policy`, `max_weight`) lives only in
[`NumericalPolicy`](@ref) — strategy types identify the estimator alone. The
runtime state is internal: inspect it read-only through `state.adaptive`.

**Notes on edge cases:** For `N = 1`, the unweighted `std` is `NaN` (path 2
of the policy) — under default `constant_policy = :error` the first adaptive
boundary throws `ArgumentError`. A nominally constant key with floating-point
rounding can have `scale < floor` → weight `1/floor` (possibly > `max_weight`;
`max_weight` is the `:clamp` value, not a global cap). See
[`NumericalPolicy`](@ref) for details.

This release is ABC-SMC-only. Estimated weights use the previous accepted
population's unweighted standard deviation (so `N ≥ 2` is required to obtain
finite positive scales without invoking path 2 for regular keys). Estimated
strategies initialize with unit weights; `FixedScale` uses its explicit
weights from generation 1.
Immediately before every generation actually executed after the first, the
implementation estimates that generation's weights, recomputes the preceding
population's distances, derives epsilon, and freezes both weights and epsilon.
It performs no update after the final requested generation. The recomputed
`epsilon_selection_distances` are not bounded by the previous tolerance (see
[`epsilon_quantile`](@ref) for monotonicity details and the role of the
`ε_cap`/`min` guard). Decompiled tolerances are frozen in
`state.adaptive.epsilon` for `gen_applied` and never change mid-generation.

MAD scaling, importance-weighted scale estimation, rejected/all-particle scale
estimation, selected update generations, prior-predictive initialization,
weight-ratio bounds, and adaptive-distance integration with ABC-MCMC or ABC-PMC
are deferred. **Performance note (deferred):** for a large population per
generation, the per-particle recomputation of `epsilon_selection_distances` under
new weights is cheap in general, but can be optimized later to scale the
observed summary once per generation and re-use a lookup — see the associated
post-merge tracker for details. This is not a claim of full pyABC parity.

```@autodocs
Modules = [SequentialABC]
Order = [:type, :function]
Private = false
Filter = x -> any(n -> occursin(n, string(x)),
                  ("AbstractScaleStrategy", "NoAdaptation", "FixedScale",
                   "StandardDeviationScale", "PreviousGenAccepted", "UnitInit",
                   "NumericalPolicy", "AdaptiveDistanceConfig"))
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
| `:smc`  | `run_smc`  | `ABCSMCState`  | `N`, `T`, `schedule`, `ε0`, `M`, `base_seed`, `adaptive`, `checkpoint_dir` |
| `:mcmc` | `run_mcmc` | `ABCMCMCChain` | `n_samples`, `ε`, `thin`, `M`, `base_seed` |
| `:pmc`  | `run_pmc`  | `ABCSMCState`  | `N`, `T`, `εs`, `schedule`, `kernel_rule`, `M`, `base_seed` |

`adaptive` (an `AdaptiveDistanceConfig`) is forwarded by `:smc` only. Each engine
validates its own keywords — forwarding does not add engine support.

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

## Simulator evaluation contracts

The additive evaluation contract is described in the
[Evaluation contracts](evaluation.md) page. It does not replace
`make_particle_seed` and is not yet consumed by the SMC, PMC, or ABC-MCMC
engines.

```@docs
EvaluationContext
EvaluationResult
EvaluationFailure
FailurePolicy
RejectEvaluation
RetryEvaluation
AbortInference
FunctionFailurePolicy
evaluation_seed
EVALUATION_SEED_SCHEMA_VERSION
context_primitive
retry_with_limit
evaluate
```

## Checkpointing

```@autodocs
Modules = [SequentialABC]
Order = [:function]
Private = false
Filter = x -> occursin("save_checkpoint", string(x)) || occursin("list_checkpoints", string(x)) || occursin("load_checkpoint", string(x)) || occursin("load_latest_state", string(x))
```
