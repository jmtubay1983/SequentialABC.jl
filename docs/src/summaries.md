# Summary statistics

ABC compares simulations to data through a *summary statistic*, and everything
downstream — the distance, the tolerance ladder, the ESS, the posterior — is
only as informative as that summary. The framework here makes summaries
**named**, **composable** and **checkable**:

| You write | You get |
|---|---|
| `FunctionSummary(fn, :count)` | one value, one name |
| `CompositeSummary([...], :abm)` | a flat NamedTuple with stable key order |
| `aggregate(reps, s)` | replicate mean **and** sd per key |
| `pca_reduce(s, X, k)` | a fitted projection, with explained variance |
| `summary_keys(s)` | the declared key order — the layout of the distance vector |

Keys are declared, not discovered: `summary_keys` works *before* any simulation
runs, so a composite can be validated when it is built, a data matrix can be
checked against it, and the vector a distance function compares has a documented
order. Duplicate keys are an error at construction time; two parts never
silently overwrite each other.

Nothing about the engines changes. A summary is still the `summary` field of the
simulator and the shape of the observed summary — see
[Plugging a summary into a sampler](#plugging-a-summary-into-a-sampler) for the
exact wiring.

## A composite summary for an agent-based model

The Schelling model in `examples/agents_schelling.jl` reduces a finished model to
one number — the segregation index. A composite replaces that single number with
a *named* set of statistics whose individual behaviour can be inspected:

```julia
using SequentialABC, Agents, Distributions, Statistics

# The summaries receive exactly what the simulator's `summary` field receives:
# a built model, after `steps` steps.
age_summary   = FunctionSummary(model -> mean(a.age for a in allagents(model)), :mean_age)
count_summary = FunctionSummary(model -> nagents(model),                      :count)
# spatial variance of the agents' x positions — a cheap clustering diagnostic
spread_summary = FunctionSummary(model -> var([a.pos[1] for a in allagents(model)]),
                                 :spatial_var)

abm = CompositeSummary([count_summary, age_summary, spread_summary], :abm)

summary_keys(abm)          # [:count, :mean_age, :spatial_var]  — fixed forever
```

Each part is a plain function plus a name, so anything you already compute about
a model drops in. The composite is then the simulator's summary — one line, and
the named values become the vector the distance sees:

```julia
simulator = AgentsSimulator(
    model_builder = (θ, rng) -> build_model(θ[1], rng),
    step!         = (model, _rng) -> step!(model, 1),
    summary       = model -> namedtuple_to_vector(compute_summary(abm, model)),
    steps         = 100,
)
```

`compute_summary(abm, model)` returns
`(count = …, mean_age = …, spatial_var = …)`; `namedtuple_to_vector` lays it out
in `summary_keys` order, which is all `distance` and `run_inference` need. The
observed side is built the same way, from the same summaries — never by hand:

```julia
y_obs = namedtuple_to_vector(compute_summary(abm, observed_model))
```

Two details worth knowing:

- A part that returns a NamedTuple is **flattened with its name prefixed**, so
  `FunctionSummary(x -> (mean = …, sd = …), :age)` contributes `:age_mean` and
  `:age_sd`. That is how per-key diagnostics enter a composite without extra
  naming work.
- `compute_summary(s, output)` maps **one** simulator output. A `Vector` is a
  legitimate single output (a time series, a field, a set of replicates) — use
  `aggregate` when you want replicate statistics instead.

## Aggregating replicates

The samplers have their own replicate knob, `M`: each particle simulates `M`
times and the summaries are averaged. That is the right default, but it hides
the variability. `aggregate` is the summary-level version and reports both:

```julia
pop_summary = CompositeSummary([FunctionSummary(length, :count),
                                FunctionSummary(mean,   :mean_age),
                                FunctionSummary(var,    :spatial_var)], :population)

pop  = [1.0, 2.0, 4.0, 8.0]
reps = [pop, 2 .* pop, 3 .* pop]           # replicate outputs of one θ

aggregate(reps, pop_summary)
# (mean_count = 4.0, mean_mean_age = 7.5, mean_spatial_var ≈ 44.72,
#  sd_count = 0.0,  sd_mean_age = 3.75, sd_spatial_var ≈ 38.73)
```

Keys are aggregator-major — all means, then all standard deviations, in
`summary_keys` order — so the resulting NamedTuple has a fixed, documented
layout like any other summary. `aggs` names the aggregators and can be
replaced: `aggs = (mean = mean, q90 = v -> quantile(v, 0.9))`.

Two honest use cases:

- **Observed data from repeated runs.** `s_obs = namedtuple_to_vector(aggregate(obs_runs, pop_summary))`
  gives the distance a target for *every* key, mean and sd alike.
- **Inside the sampler, when replicas are the science.** Let the simulator hand
  the summary a vector of replicate outputs, and aggregate there — then the
  replicate sd takes part in the acceptance decision, not just the mean:

  ```julia
  simulator = ContinuousSimulator(
      model_builder = θ -> θ,
      evolve        = (θ, rng) -> [replicate(θ, rng) for _ in 1:3],
      summary       = reps -> namedtuple_to_vector(aggregate(reps, pop_summary)),
  )
  ```

  `M = 1` is correct here: the replicates are the simulator's own output, and
  the summary — not the engine — decides how to summarise them.

## Compressing a high-dimensional summary

A large summary is not automatically a good one: each extra key adds noise to
the distance, and correlated keys count several times over. `pca_reduce` fits a
projection on a pilot ensemble and keeps the `k` directions that carry the
variance. PCA needs data, so the data matrix is explicit — one row per
observation, one column per key of the summary:

```julia
# The summary being reduced: one value per key of the raw output
base = CompositeSummary([FunctionSummary(raw -> raw[1], :a),
                         FunctionSummary(raw -> raw[2], :b)], :two_keys)

# Pilot ensemble on the prior predictive distribution: one simulator output per
# prior draw, one summary vector per row (fixed seed — this is a data choice)
rng   = MersenneTwister(2026)
pilot = [sim.evolve(sim.model_builder(θ), rng) for θ in prior_draws]   # ContinuousSimulator
X     = reduce(vcat, permutedims.(namedtuple_to_vector.(compute_summary.(base, pilot))))

proj = pca_reduce(base, X, 3)      # 3 components
proj.explained                     # fraction of total variance per component
summary_keys(proj)                 # [:pc1, :pc2, :pc3]
```

The projection is itself a summary, so it composes and plugs in like any other —
the sampler never knows PCA happened:

```julia
simulator = ContinuousSimulator(model_builder = θ -> θ,
                                evolve        = evolve,
                                summary       = raw -> namedtuple_to_vector(compute_summary(proj, raw)))
s_obs     = namedtuple_to_vector(compute_summary(proj, raw_observed))
```

`proj(X)` applies the fitted map to a whole matrix at once, and
`proj.explained` is the diagnostic to check before shrinking a summary: if the
first component carries 0.99 of the variance, the other keys were mostly noise,
and `k = 1` says so. Eigenvector *signs* are arbitrary, so each component is
flipped to a canonical convention (its largest-|value| entry is positive) —
refitting on the same data is bit-identical, on any machine, with any BLAS.

## Plugging a summary into a sampler

There is no `summaries_fn` keyword, and no new per-engine plumbing: summaries
enter where they always have.

| Piece | Role |
|---|---|
| `simulator.summary` | raw simulator output → summary (a number, or a vector via `namedtuple_to_vector`) |
| `s_obs` | the observed summary, produced by *the same* summary functions |
| `distance` | compares the two — for named summaries, Euclidean on `namedtuple_to_vector` |
| `run_inference` / `run_smc` / `run_mcmc` / `run_pmc` | unchanged; they only see the distance |

So the complete wiring is three lines plus the call you already make:

```julia
base2 = CompositeSummary([FunctionSummary(raw -> raw[1], :a),
                          FunctionSummary(raw -> raw[2], :b)], :two_keys)

simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve        = (θ, rng) -> θ .+ 0.3 .* randn(rng, 2),
    summary       = raw -> namedtuple_to_vector(compute_summary(base2, raw)),
)

cfg = ABCSMCConfig(prior, sample_prior, identity,
                   (s, so) -> sqrt(sum(abs2, s .- so)),
                   GaussianKernel(), [1.0 0.0; 0.0 1.0], simulator)

state = run_inference(:smc, cfg, s_obs; N = 1000, T = 5, base_seed = 2026)
```

The summary's key order fixes the layout of both vectors, so `s_obs` and every
simulated summary are compared element-by-element: `:count` against `:count`,
`:mean_age` against `:mean_age`. Renaming a key renames the comparison — silently
adding or dropping one is caught by `validate_summary_keys` at the first call,
not by a mysteriously worse posterior.

A runnable end-to-end script — composite, replicate aggregation, PCA compression,
and all three engines — lives in `examples/summaries_example.jl`:

```
julia --project=. examples/summaries_example.jl
```

## Units, scaling and name resolution

Keys with wildly different magnitudes make a Euclidean distance a one-key
distance: a count in the thousands drowns a variance in the hundredths. `scale`
multiplies selected keys by fixed factors and *validates them against the
summary*, so a typo cannot silently scale nothing:

```julia
scaled = scale(abm, Dict(:count => 1e-3))    # :mean_age, :spatial_var untouched
```

`scale` is a method of `Distributions.scale` — the scale-parameter accessor for
distributions, here meaning the rescaled summary — so `using SequentialABC` and
`using Distributions` (the usual pairing, since priors are distributions)
resolve it without qualification.

`transform` — the PCA map — is deliberately **not** exported. In this package
`transform` already means the prior→model-parameter map (`cfg.transform`,
written `transform(θ_raw) = …` in the [Quickstart](@ref)); PCA's map is reached
as `proj(X)` or `SequentialABC.transform(proj, X)`, so it can never collide with
your own `transform`.

## Determinism

The framework contains no randomness: summaries are pure functions of one
simulator output, `aggregate` applies whatever aggregator you passed, and PCA is
a deterministic eigendecomposition with the canonical sign convention above.
Two consequences:

- The distance vector for a given output is bit-identical across runs, sessions
  and machines, because it is a function of a fixed key order.
- Fitting a projection twice on the same pilot matrix gives the same components,
  so a saved `proj` and a refitted one cannot silently differ.

`state.θ`, `state.w`, `state.ε` and the checkpoint format are untouched by
everything on this page — the engines' determinism guarantees are the ones you
already rely on (see the Design Notes page).

## API

```@docs
AbstractSummary
FunctionSummary
CompositeSummary
ScaledSummary
PCAProjection
summary_keys
compute_summary
compose
scale(::AbstractSummary, ::AbstractDict{Symbol, <:Real})
pca_reduce
SequentialABC.transform
aggregate
flatten_namedtuple
namedtuple_to_vector
vector_to_namedtuple
validate_summary_keys
```
