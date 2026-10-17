# src/summaries/Summaries.jl
#
# Composable, named summary statistics.
#
# A summary here is a small object with three jobs: declare the keys it will
# produce (`summary_keys`), turn one simulator output into those keys
# (`compute_summary`), and be combinable with other summaries (`compose`,
# `scale`, `pca_reduce`). Names are what make the pieces composable: every value
# carries a stable `Symbol` key, so a composite can merge parts without the
# ordering ambiguity of a bare vector, and a distance function can be built from
# a NamedTuple in a fixed, documented order.
#
# The samplers themselves are unchanged: a summary enters the existing pipeline
# where summaries have always lived — `ContinuousSimulator(...; summary = …)` /
# `AgentsSimulator(...; summary = …)` — and `namedtuple_to_vector` adapts it to
# the vector a distance function consumes.

# `scale` is the *same generic function* as `Distributions.scale` (extended with
# methods below, in the spirit of `import Statistics: cov` in AdaptiveKernel.jl).
# Two independently-owned exported `scale`s would make `using SequentialABC,
# Distributions` — the standard pairing, since priors are distributions — raise
# an ambiguity error on every `scale` call instead of resolving one function.
import Distributions: scale

"""
    AbstractSummary

Supertype of the named summary types: [`FunctionSummary`](@ref),
[`CompositeSummary`](@ref), [`ScaledSummary`](@ref) and
[`PCAProjection`](@ref).

A summary declares the keys it produces ([`summary_keys`](@ref)) and turns one
simulator output into a NamedTuple with exactly those keys
([`compute_summary`](@ref)). Because keys are declared rather than discovered,
composites can be validated and distance vectors have a documented order.
"""
abstract type AbstractSummary end

# A summary is a scalar in broadcasting — `compute_summary.(base, outputs)` maps
# over the outputs. Without this, Julia's generic `broadcastable` fallback would
# try to iterate the summary itself.
Base.broadcastable(s::AbstractSummary) = Ref(s)

"""
    summary_keys(s::AbstractSummary) -> Vector{Symbol}

The keys `s` produces, in order. Declared by the summary itself, so it can be
called before any simulation runs — that is what lets [`pca_reduce`](@ref) check
a data matrix and [`namedtuple_to_vector`](@ref) fix the layout of a distance
vector.
"""
function summary_keys end

"""
    compute_summary(s::AbstractSummary, sim_output) -> NamedTuple

Apply `s` to a single simulator output and return a NamedTuple whose keys are
exactly `summary_keys(s)`.

Vector-valued input must be aggregated explicitly with [`aggregate`](@ref): a
`Vector` here is a legitimate *single* output (a time series, a field), so the
replicate case is deliberately not inferred from the container type.
"""
function compute_summary end

# ── single function ──────────────────────────────────────────────────────────

"""
    FunctionSummary(fn::Function, name::Symbol) <: AbstractSummary

Wrap a plain function `fn(sim_output) -> value` as a named one-key summary: the
value is reported under `name`. If `fn` returns a NamedTuple, its keys are
prefixed with `name` when the summary is used inside a
[`CompositeSummary`](@ref) — `(mean = 1.0, sd = 0.5)` under `:age` becomes
`:age_mean` and `:age_sd`.

```julia
count_summary = FunctionSummary(m -> length(m.agents), :count)
compute_summary(count_summary, model)     # (count = 42,)
```
"""
struct FunctionSummary{F<:Function} <: AbstractSummary
    fn::F
    name::Symbol
    function FunctionSummary(fn::Function, name::Symbol)
        isempty(string(name)) && throw(ArgumentError("FunctionSummary needs a non-empty name"))
        return new{typeof(fn)}(fn, name)
    end
end

summary_keys(s::FunctionSummary) = Symbol[s.name]

function compute_summary(s::FunctionSummary, sim_output)
    value = s.fn(sim_output)
    return NamedTuple{(s.name,)}((value,))
end

# ── composite ────────────────────────────────────────────────────────────────

"""
    CompositeSummary(summaries::AbstractVector{<:AbstractSummary}, name::Symbol) <: AbstractSummary

An ordered group of summaries. Keys are the concatenation of the parts' keys, in
part order, so `summary_keys` of a composite reads exactly like the vector the
distance function will compare; nested NamedTuples are flattened with `_`
prefixes ([`flatten_namedtuple`](@ref)) and duplicate keys are rejected when the
composite is constructed.

```julia
abm = CompositeSummary([FunctionSummary(m -> length(m.agents), :count),
                        FunctionSummary(m -> mean_age(m), :mean_age),
                        FunctionSummary(m -> spatial_var(m), :spatial_var)], :abm)
summary_keys(abm)        # [:count, :mean_age, :spatial_var]
```
"""
struct CompositeSummary <: AbstractSummary
    summaries::Vector{AbstractSummary}
    name::Symbol
    function CompositeSummary(summaries::Vector{AbstractSummary}, name::Symbol)
        isempty(summaries) && throw(ArgumentError("CompositeSummary needs at least one summary"))
        isempty(string(name)) && throw(ArgumentError("CompositeSummary needs a non-empty name"))

        seen = Set{Symbol}()
        for s in summaries, k in summary_keys(s)
            k in seen && throw(ArgumentError(
                "duplicate summary key `$k` in composite `$name`; " *
                "rename one of the parts (keys so far: $(Tuple(sort!(collect(seen))))"))
            push!(seen, k)
        end

        return new(summaries, name)
    end
end

# `Vector{AbstractSummary}` is invariant — a literal `[fs1, fs2]` is a
# `Vector{FunctionSummary}` — so accept any vector of summaries and widen it.
CompositeSummary(summaries::AbstractVector{<:AbstractSummary}, name::Symbol) =
    CompositeSummary(convert(Vector{AbstractSummary}, summaries), name)

summary_keys(s::CompositeSummary) =
    Symbol[k for part in s.summaries for k in summary_keys(part)]
function compute_summary(s::CompositeSummary, sim_output)
    parts = (compute_summary(part, sim_output) for part in s.summaries)
    return flatten_namedtuple(merge(parts...))
end

"""
    compose(s1::AbstractSummary, s2::AbstractSummary; name = :composed) -> CompositeSummary

Chain two summaries into a [`CompositeSummary`](@ref). Operands that are already
composites are spliced, so `compose(compose(a, b), c)` and `compose(a, compose(b, c))`
both produce one three-part composite with the same key order.

```julia
sizes = compose(FunctionSummary(x -> x.n, :n), FunctionSummary(x -> x.m, :m))
summary_keys(sizes)      # [:n, :m]
```
"""
function compose(s1::AbstractSummary, s2::AbstractSummary; name::Symbol = :composed)
    parts = vcat(_summary_parts(s1), _summary_parts(s2))
    return CompositeSummary(parts, name)
end

# Parts of a summary as a vector, splicing composites so nesting stays flat.
_summary_parts(s::CompositeSummary) = s.summaries
_summary_parts(s::AbstractSummary) = AbstractSummary[s]

# ── scaling ──────────────────────────────────────────────────────────────────

"""
    ScaledSummary(inner::AbstractSummary, scales::Dict{Symbol,Real}, name::Symbol)
        <: AbstractSummary

A summary whose keys are multiplied by fixed factors — one per key, e.g. to put
summaries with different units on a comparable scale before they share a
distance function. Keys absent from `scales` are left unchanged; keys in
`scales` that the inner summary does not produce throw an `ArgumentError`, so a
typo cannot silently scale nothing. See [`scale`](@ref).
"""
struct ScaledSummary <: AbstractSummary
    inner::AbstractSummary
    scales::Dict{Symbol,Float64}
    name::Symbol
    function ScaledSummary(inner::AbstractSummary, scales::AbstractDict{Symbol,<:Real},
                           name::Symbol)
        known = Set(summary_keys(inner))
        for k in keys(scales)
            k in known || throw(ArgumentError(
                "scale key `$k` is not produced by summary `$(inner.name)` " *
                "(keys: $(sort!(collect(known))))"))
        end
        return new(inner, Dict{Symbol,Float64}(k => Float64(v) for (k, v) in scales), name)
    end
end

summary_keys(s::ScaledSummary) = summary_keys(s.inner)

function compute_summary(s::ScaledSummary, sim_output)
    nt = compute_summary(s.inner, sim_output)
    ks = keys(nt)
    return NamedTuple{ks}(ntuple(i -> nt[i] * get(s.scales, ks[i], 1.0), length(ks)))
end

"""
    scale(s::AbstractSummary, scales::AbstractDict{Symbol,<:Real};
          name = Symbol(s.name, :_scaled)) -> ScaledSummary

Multiply selected keys of `s` by fixed factors:

```julia
summary_keys(scale(abm, Dict(:count => 1e-3)))    # unchanged; the values are scaled
```

Useful when one key would otherwise dominate a Euclidean distance because of its
units. Factors are validated against `s`'s keys at construction time.

This is a method of `Distributions.scale` — the scale parameter accessor for
distributions, here meaning the rescaled summary — so `using SequentialABC,
Distributions` resolves `scale` without qualification.
"""
scale(s::AbstractSummary, scales::AbstractDict{Symbol,<:Real};
      name::Symbol = Symbol(s.name, :_scaled)) = ScaledSummary(s, scales, name)

# ── PCA reduction ────────────────────────────────────────────────────────────

"""
    PCAProjection <: AbstractSummary

A fitted PCA map from a source summary's keys to `k` principal-component scores
named `:pc1, :pc2, …, :pcK`. Built by [`pca_reduce`](@ref); apply it to a single
simulator output with [`compute_summary`](@ref), to a whole matrix of summary
vectors by calling it — `proj(X)` — or with `SequentialABC.transform(proj, X)`.

`transform` is deliberately **not exported**: in this package `transform`
already names the config's prior→model-parameter map (`cfg.transform`, written
`transform(θ_raw) = …` in the quickstart), so PCA's map stays qualified while
the callable form `proj(X)` needs no name at all.

The components follow a canonical sign convention (the entry of largest absolute
value in each component is positive), so a projection is reproducible and
comparable across fits — plain eigenvector signs are arbitrary.

Fields: `source`, `components` (`k × p`), `mean` (length `p`), `explained`
(fraction of total variance per component), `name`.
"""
struct PCAProjection <: AbstractSummary
    source::AbstractSummary
    components::Matrix{Float64}
    mean::Vector{Float64}
    explained::Vector{Float64}
    name::Symbol
end

summary_keys(p::PCAProjection) = [Symbol(:pc, i) for i in 1:size(p.components, 1)]

function compute_summary(p::PCAProjection, sim_output)
    nt = compute_summary(p.source, sim_output)
    validate_summary_keys(nt, p.source)
    z = p.components * (namedtuple_to_vector(nt) .- p.mean)
    return vector_to_namedtuple(z, summary_keys(p))
end

"""
    transform(p::PCAProjection, X::AbstractMatrix{<:Real}) -> Matrix{Float64}

Project each row of `X` (one summary vector per row, in `summary_keys(p.source)`
order) onto the fitted components: an `n × k` matrix of scores. `X` must have
one column per source key.

Not exported — `proj(X)` is the short form and `compute_summary(proj, output)`
is the per-output one (see [`PCAProjection`](@ref) for why `transform` stays
qualified here).

```julia
proj(X)                          # n × k scores
SequentialABC.transform(proj, X) # the same map, by name
compute_summary(proj, output)    # the same map for one simulator output
```
"""
function transform(p::PCAProjection, X::AbstractMatrix{<:Real})
    size(X, 2) == length(p.mean) || throw(ArgumentError(
        "X has $(size(X, 2)) columns but the projection expects " *
        "$(length(p.mean)) (keys: $(summary_keys(p.source)))"))
    return (Matrix{Float64}(X) .- p.mean') * p.components'
end

(p::PCAProjection)(X::AbstractMatrix{<:Real}) = transform(p, X)

"""
    pca_reduce(summary::AbstractSummary, X::AbstractMatrix{<:Real}, k::Int;
               center = true, name = :pca) -> PCAProjection

Fit a `k`-component PCA on `X` — one row per observation, one column per key of
`summary`, in `summary_keys(summary)` order — and return the projection. The
data has to come from somewhere: a pilot ensemble, a prior predictive run, or
the observed data. A convenient source is the simulator itself — one raw output
per prior draw, one summary vector per row:

```julia
rng = MersenneTwister(2026)
pilot = [sim.evolve(sim.model_builder(θ), rng) for θ in prior_draws]
X = reduce(vcat, permutedims.(namedtuple_to_vector.(compute_summary.(base, pilot))))
proj = pca_reduce(base, X, 3)
```

`center = true` subtracts the column means first. `explained` on the result
reports the fraction of total variance each component captures, which is the
diagnostic to check before shrinking a summary.
"""
function pca_reduce(summary::AbstractSummary, X::AbstractMatrix{<:Real}, k::Int;
                    center::Bool = true, name::Symbol = :pca)
    ks = summary_keys(summary)
    n, p = size(X)
    p == length(ks) || throw(ArgumentError(
        "X has $p columns but summary `$(summary.name)` declares $(length(ks)) keys: $ks"))
    1 <= k <= min(n, p) || throw(ArgumentError(
        "k must satisfy 1 ≤ k ≤ min(n, p) = $(min(n, p)), got $k"))

    μ = center ? vec(sum(X; dims = 1)) ./ n : zeros(p)
    Xc = Matrix{Float64}(X) .- μ'
    C = Symmetric((Xc' * Xc) ./ max(n - 1, 1))
    E = eigen(C)

    # eigen returns ascending eigenvalues; take the k largest.
    order = sortperm(E.values; rev = true)[1:k]
    λ = max.(E.values[order], 0.0)
    V = E.vectors[:, order]                      # p × k

    # Canonical sign: the largest-|value| entry of each component is positive.
    for j in 1:k
        v = view(V, :, j)
        if v[argmax(abs.(v))] < 0
            V[:, j] .= .-v
        end
    end

    total = sum(max.(E.values, 0.0))
    explained = total > 0 ? λ ./ total : zeros(k)

    return PCAProjection(summary, permutedims(V), μ, explained, name)
end

# ── aggregation over replicates ──────────────────────────────────────────────

"""
    aggregate(reps::AbstractVector, summary::AbstractSummary;
              aggs = (mean = mean, sd = std)) -> NamedTuple

Apply `summary` to each element of `reps` — repeated simulator outputs of the
same model, e.g. `Vector` of replicate runs — and aggregate every key. With the
default `aggs` a key `:count` becomes `:mean_count` and `:sd_count`; the key
order is aggregator-major (all means, then all standard deviations), matching the
order of `aggs`.

`aggs` maps a name to any function of a vector, so
`aggregate(reps, s; aggs = (mean = mean,))` or `(median = median, q90 = x -> quantile(x, 0.9))`
both work. This is the summary-level counterpart of the samplers' own `M`
replicate handling, which averages internally: use `aggregate` to build the
*observed* summary from repeated measurements, or to summarise a pilot ensemble
before [`pca_reduce`](@ref).

```julia
s_obs_nt = aggregate(observed_runs, base)     # (mean_count = …, sd_count = …, …)
s_obs = namedtuple_to_vector(s_obs_nt)
```
"""
function aggregate(reps::AbstractVector, summary::AbstractSummary;
                   aggs = (mean = mean, sd = std))
    isempty(reps) && throw(ArgumentError("aggregate needs at least one replicate"))
    isempty(aggs) && throw(ArgumentError("aggregate needs at least one aggregator"))

    ks = summary_keys(summary)
    nts = map(r -> compute_summary(summary, r), reps)
    for nt in nts
        validate_summary_keys(nt, ks)
    end

    # reps × keys matrix of values (every declared key must be Real to aggregate)
    M = Matrix{Float64}(undef, length(nts), length(ks))
    for (i, nt) in enumerate(nts)
        M[i, :] = namedtuple_to_vector(nt)
    end

    keys_out = Symbol[]
    vals_out = Any[]
    for (agg_name, agg_fn) in pairs(aggs), (j, k) in enumerate(ks)
        push!(keys_out, Symbol(agg_name, :_, k))
        push!(vals_out, agg_fn(view(M, :, j)))
    end

    return NamedTuple{Tuple(keys_out)}(Tuple(vals_out))
end
