# Distances

Every acceptance decision in every engine goes through one call:

```
cfg.distance(summary_sim, summary_obs) -> Real
```

That is the whole interface. The engines — [`run_smc`](@ref), [`run_mcmc`](@ref),
[`run_pmc`](@ref), and the [`run_inference`](@ref) front door — only ever call
that expression and compare the result against the current tolerance; nothing
else about a run changes when the distance does. This page provides the three
metrics that cover most ABC work, ready for that slot:

| You write | You get |
|---|---|
| `l2_distance` | Euclidean — the named form of the ad-hoc `sqrt(sum(abs2, …))` |
| `weighted_l2(w)` | per-key weights: units, importance, muting |
| `mahalanobis_distance(s, s_obs, Σ)` | covariance-aware distance for one explicit `Σ` |
| `make_mahalanobis(X)` | the same, estimated from a pilot **with shrinkage**, factorised once |

Nothing else changes: a distance is a value in the config, the engines call it
exactly as before, and the results stay deterministic (these are pure
functions — no RNG anywhere).

## The one-function contract

A distance is *any* callable of two summaries. A plain function, a closure, or
one of the callables built here all work; the only rule is the signature, and
the two summaries arrive in the same representation — a vector, a scalar, or a
NamedTuple for named summaries:

```julia
using SequentialABC, Distributions

cfg = ABCSMCConfig(prior, sample_prior, transform, l2_distance,
                   GaussianKernel(), Σ, simulator)
```

For named summaries the observed side must be built by the same summary that
produces the simulated side, so key order is shared — see
[Summary statistics](@ref) for the wiring. The distance functions on this page
accept those NamedTuples directly and compare them **by key**: keys must match
as a set, but declaration order does not matter.

## Euclidean: `l2_distance`

`l2_distance` is the named form of the line every ABC project grows first:

```julia
distance(s, so) = sqrt(sum(abs2, s .- so))    # your cfg's distance today
l2_distance                                    # the same function, named
```

It is deliberately the same expression, so switching an existing run to
`l2_distance` cannot change a single rounding bit — and therefore not a single
acceptance decision:

```julia
l2_distance([1.0, 2.0], [4.0, 6.0])            # 5.0

# scalar summaries (a single number per simulation)
l2_distance(3.0, 1.5)                          # 1.5

# named summaries — compared by key, order-free
l2_distance((count = 4.0, mean_age = 3.75),
            (count = 2.0, mean_age = 4.75))    # 2.236…
```

Vectors must have the same length and matrices the same shape (the Frobenius
norm of the difference); a summary key holding an array is an error telling you
to reduce it to a scalar first — exactly the rule
[`namedtuple_to_vector`](@ref) enforces.

## Weighted Euclidean: `weighted_l2`

A count in the thousands and a variance in the hundredths do not belong in the
same Euclidean sum. `weighted_l2` rescales each key's contribution:

```julia
weighted_l2([1.0, 1.0], [0.0, 0.0], [4.0, 9.0])      # √13
weighted_l2([1.0, 100.0], [0.0, 0.0], [1.0, 0.0])    # 1.0 — the second key is muted
weighted_l2(3.0, 1.0, 4.0)                           # 4.0 — scalar form
```

For named summaries, weights can be named too. Unlisted keys keep weight 1 and
keys the summary does not produce throw — the same rule [`scale`](@ref) applies
to summaries, so a typo cannot silently weigh nothing:

```julia
# only :count contributes; :mean_age keeps weight 1
weighted_l2((count = 2000.0, mean_age = 30.0),
            (count = 1900.0, mean_age = 31.0), (count = 1e-3,))
```

Passed with one argument, `weighted_l2` validates the weights once and returns
a callable — the form the config wants:

```julia
cfg = ABCSMCConfig(prior, sample_prior, transform,
                   weighted_l2((count = 1e-3,)),
                   GaussianKernel(), Σ, simulator)
```

`weighted_l2` and [`scale`](@ref) are two places to put the same factor: `scale`
rescales the **summary values** (so the rescaled values are also what a PCA
sees, what gets stored in `state.summaries`, …), while the weight affects only
the distance. They commute — scaling key `i` by `c` is weighing it by `c²`. In
particular, `w = 1 ./ σ²` makes the weighted metric the **diagonal** Mahalanobis
distance.

## Mahalanobis with shrinkage

Summary keys are rarely independent: two proportions of the same population, a
mean and its variance, neighbouring time points. Euclidean distances count each
shared signal once per key, over and over. The Mahalanobis distance divides by
the covariance instead:

```julia
Σ = [4.0 1.0; 1.0 2.0]
mahalanobis_distance([1.0, 1.0], [0.0, 0.0], Σ)      # 0.7559…
```

`Σ` must be symmetric positive definite. Use it directly for a one-off — and
note that a `Cholesky` factor is accepted in place of `Σ`, skipping
re-factorisation when the same covariance is reused.

The usual problem in ABC is that `Σ` must be **estimated**, and the sample
covariance `cov(X; dims = 1)` of a pilot ensemble is singular whenever
observations are few relative to keys — small pilots with many summaries are
the normal case, not the exception. `make_mahalanobis` shrinks the estimate
toward its diagonal,

    Σ̂ = (1 - λ) · cov(X) + λ · Diagonal(cov(X)),     λ = shrinkage ∈ [0, 1],

and factorises it once. One row of `X` is one observation, one column one
summary key, in [`summary_keys`](@ref) order — the same layout as
[`pca_reduce`](@ref):

```julia
# Pilot ensemble on the prior predictive distribution: one summary vector per
# draw. The seed is fixed because the pilot is a data choice, like s_obs itself.
rng   = MersenneTwister(2026)
pilot = [sim.evolve(sim.model_builder(θ), rng) for θ in prior_draws]
X     = reduce(vcat, permutedims.(namedtuple_to_vector.(compute_summary.(base, pilot))))

d   = make_mahalanobis(X)                     # λ = 0.1 by default
cfg = ABCSMCConfig(prior, sample_prior, transform, d, GaussianKernel(), Σ, simulator)

state = run_inference(:smc, cfg, s_obs; N = 1000, T = 5, base_seed = 2026)
```

The shrinkage dial:

- `shrinkage = 0.0` — the raw sample covariance: exact, and singular exactly
  when the pilot is too small or keys are collinear. Collinear keys make this
  either fail the factorisation or produce distances amplified by rounding
  noise — either way, not a metric to run.
- `shrinkage = 0.1` — the default; the diagonal contribution that makes the
  metric well defined at a small bias.
- `shrinkage = 1.0` — ignores correlations, leaving `1/σᵢ²` weights; the
  diagonal metric from the section above.

A constant key has zero variance and cannot be regularised at *any* `λ` — the
factorisation runs when the distance is built, and the error names the cause,
so nothing unusable survives silently into the first particle.

## Determinism

These functions are pure and RNG-free. The same summaries give the same
distance, the same pilot gives the same shrunk covariance, and — because a
distance feeds `d <= ε` tests — the engines' replay guarantee is unchanged:
same config, same `base_seed`, same accepted particles. A distance is not a
place to hide state.

## API

```@docs
l2_distance
weighted_l2
mahalanobis_distance
make_mahalanobis
SequentialABC.WeightedL2
SequentialABC.MahalanobisDistance
```
