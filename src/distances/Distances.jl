# src/distances/Distances.jl
#
# Distance functions for the one slot every engine consults:
#
#     cfg.distance(summary_sim, summary_obs) -> Real
#
# The engines only ever call that expression (see `ABCSMCConfig`), so a distance
# is *any* callable of two summaries — a plain function, a closure, or one of
# the callables built here. Three metrics cover the common cases:
#
#   * `l2_distance`          — plain Euclidean; the drop-in for the ad-hoc
#                              `(s, so) -> sqrt(sum(abs2, s .- so))` one-liner
#   * `weighted_l2`          — Euclidean with per-key weights (units,
#                              importance, muting); `w = 1 ./ σ²` is Mahalanobis
#                              with a diagonal covariance
#   * `mahalanobis_distance` — the full covariance-aware metric, for an explicit
#                              `Σ` or a precomputed `Cholesky` factor
#     `make_mahalanobis`     — estimate `Σ` from a pilot ensemble *with
#                              shrinkage* and factorise it once
#
# Named summaries (`NamedTuple`, from the summaries framework) are converted
# with `namedtuple_to_vector` and compared **by key**, so these functions plug
# directly into a config whose summaries are named. Everything here is
# deterministic — no RNG anywhere — and a distance call allocates nothing beyond
# the intermediate difference vectors.

# ── shared helpers ───────────────────────────────────────────────────────────

# Align two named summaries for value-wise comparison. The key *sets* must be
# equal; values are matched by name, never by position, so declaration order
# does not matter (a reordered observed summary compares the same way).
function _aligned_summary_vectors(a::NamedTuple, b::NamedTuple)
    ka, kb = keys(a), keys(b)
    if ka == kb
        return namedtuple_to_vector(a), namedtuple_to_vector(b)
    end
    Set(ka) == Set(kb) || throw(ArgumentError(
        "summary keys differ: $(Tuple(ka)) vs $(Tuple(kb)); the two summaries " *
        "must declare the same keys"))
    va = namedtuple_to_vector(a)
    vb = namedtuple_to_vector(b)
    return va, vb[Int[findfirst(==(k), kb) for k in ka]]
end

# Weight validation shared by `weighted_l2` and `WeightedL2`: weights define a
# metric only if they are finite, non-negative, and not all zero.
function _check_weights(w::AbstractVector{<:Real})
    isempty(w) && throw(ArgumentError("weights must not be empty"))
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError(
        "weights must be finite and ≥ 0, got $w"))
    any(>(0), w) || throw(ArgumentError(
        "weights must have at least one positive entry — all-zero weights " *
        "make every distance zero"))
    return w
end

# Expand named weights into a full weight vector over the key order `ks`:
# unlisted keys keep weight 1, unknown keys throw (a typo must not weigh
# nothing — the same rule `scale` applies to summaries).
function _named_weight_vector(w::NamedTuple, ks)
    for k in keys(w)
        k in ks || throw(ArgumentError(
            "weight key `$k` is not a summary key (keys: $(Tuple(ks)))"))
    end
    wv = ones(Float64, length(ks))
    for (k, v) in pairs(w)
        wv[Int(something(findfirst(==(k), ks)))] = Float64(v)
    end
    return _check_weights(wv)
end

# ── Euclidean ────────────────────────────────────────────────────────────────

"""
    l2_distance(s, s_obs) -> Real

Euclidean distance `√(Σᵢ (sᵢ - s_obsᵢ)²)` — the direct, named replacement for
the ad-hoc `(s, so) -> sqrt(sum(abs2, s .- so))` every ABC script grows:

```julia
l2_distance([1.0, 2.0], [4.0, 6.0])            # 5.0
l2_distance(3.0, 1.5)                           # 1.5   (scalar summaries)
l2_distance((count = 4.0, mean_age = 3.75),
            (count = 2.0, mean_age = 4.75))     # 2.236… (named summaries)
```

Accepts two `Real`s, two arrays of the same shape (the Frobenius norm for
matrices), or two `NamedTuple`s — named summaries are converted with
[`namedtuple_to_vector`](@ref) and matched **by key**, so declaration order does
not matter but the key sets must be identical.

It is a plain function, so it goes straight into the config:

```julia
cfg = ABCSMCConfig(prior, sample_prior, transform, l2_distance,
                   GaussianKernel(), Σ, simulator)
```
"""
l2_distance(a::Real, b::Real) = abs(a - b)

function l2_distance(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    size(a) == size(b) || throw(DimensionMismatch(
        "summaries have different shapes: $(size(a)) vs $(size(b))"))
    # Deliberately the same expression as the ad-hoc one-liner
    # `sqrt(sum(abs2, s .- so))`: replacing it with `l2_distance` cannot change
    # a single rounding bit — or an acceptance decision — of an existing run.
    return sqrt(sum(abs2, a .- b))
end

function l2_distance(a::NamedTuple, b::NamedTuple)
    va, vb = _aligned_summary_vectors(a, b)
    return l2_distance(va, vb)
end

# ── weighted Euclidean ───────────────────────────────────────────────────────

"""
    weighted_l2(s, s_obs, w) -> Real
    weighted_l2(w) -> WeightedL2

Weighted Euclidean distance `√(Σᵢ wᵢ (sᵢ - s_obsᵢ)²)`. Weights rescale how much
each summary key contributes: `w = 1 ./ σ²` turns the metric into Mahalanobis
with a diagonal covariance, and `w = 0` mutes a key entirely.

`w` may be a vector (one entry per summary value) or a `NamedTuple` for named
summaries — unlisted keys keep weight 1, unknown keys throw, mirroring
[`scale`](@ref):

```julia
weighted_l2([1.0, 1.0], [0.0, 0.0], [16.0, 9.0])     # 5.0
weighted_l2([1.0, 100.0], [0.0, 0.0], [1.0, 0.0])    # 1.0 — second key muted

# named weights: only :count contributes, :mean_age keeps weight 1
weighted_l2((count = 2000.0, mean_age = 30.0),
            (count = 1900.0, mean_age = 31.0), (count = 1e-3,))
```

Called with a single argument, `weighted_l2` validates the weights once and
returns a callable `(s, s_obs) -> Real` — the form every engine expects:

```julia
cfg = ABCSMCConfig(prior, sample_prior, transform, weighted_l2(w),
                   GaussianKernel(), Σ, simulator)
```

Weights must be finite, non-negative, and not all zero; vector weights must
match the summary length, named weights are checked against the summary keys at
call time (a key the summary does not produce is an error, never a silent no-op).
"""
function weighted_l2(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                     w::AbstractVector{<:Real})
    length(a) == length(b) || throw(DimensionMismatch(
        "summaries have different lengths: $(length(a)) vs $(length(b))"))
    length(w) == length(a) || throw(DimensionMismatch(
        "got $(length(w)) weights for a summary of length $(length(a))"))
    _check_weights(w)
    acc = 0.0
    @inbounds for i in eachindex(a, b, w)
        acc += w[i] * abs2(a[i] - b[i])
    end
    return sqrt(acc)
end

function weighted_l2(a::NamedTuple, b::NamedTuple, w::AbstractVector{<:Real})
    va, vb = _aligned_summary_vectors(a, b)
    return weighted_l2(va, vb, w)
end

function weighted_l2(a::NamedTuple, b::NamedTuple, w::NamedTuple)
    va, vb = _aligned_summary_vectors(a, b)
    return weighted_l2(va, vb, _named_weight_vector(w, keys(a)))
end

function weighted_l2(a::Real, b::Real, w::Real)
    (isfinite(w) && w >= 0) || throw(ArgumentError(
        "weight must be finite and ≥ 0, got $w"))
    return sqrt(w) * abs(a - b)
end

"""
    WeightedL2

Callable returned by `weighted_l2(w)`: stores the validated weights (a copy of
a vector, so mutating the source array afterwards cannot change the metric) and
applies [`weighted_l2`](@ref) to each pair of summaries. The field `weights`
holds the vector or the named weights exactly as constructed.

```julia
d = weighted_l2([16.0, 9.0])
d([1.0, 1.0], [0.0, 0.0])   # 5.0
d.weights                   # [16.0, 9.0]
```
"""
struct WeightedL2{W<:Union{Vector{Float64},NamedTuple}}
    weights::W
    function WeightedL2(weights::AbstractVector{<:Real})
        return new{Vector{Float64}}(Float64.(_check_weights(weights)))
    end
    function WeightedL2(weights::NamedTuple)
        isempty(weights) && throw(ArgumentError("weights must name at least one key"))
        for (k, v) in pairs(weights)
            v isa Real || throw(ArgumentError(
                "weight `$k` has type $(typeof(v)); weights must be Real"))
            (isfinite(v) && v >= 0) || throw(ArgumentError(
                "weight `$k` must be finite and ≥ 0, got $v"))
        end
        return new{typeof(weights)}(weights)
    end
end

weighted_l2(w::AbstractVector{<:Real}) = WeightedL2(w)
weighted_l2(w::NamedTuple) = WeightedL2(w)

(d::WeightedL2)(a, b) = weighted_l2(a, b, d.weights)

Base.show(io::IO, d::WeightedL2) =
    print(io, "WeightedL2(", _weights_str(d.weights), ")")
_weights_str(w::AbstractVector) = string(length(w), "-element weights")
_weights_str(w::NamedTuple) = string("named weights ", w)

# ── Mahalanobis ──────────────────────────────────────────────────────────────

"""
    mahalanobis_distance(s, s_obs, Σ) -> Real
    mahalanobis_distance(s, s_obs, F::Cholesky) -> Real

Mahalanobis distance `√((s - s_obs)' Σ⁻¹ (s - s_obs))`: Euclidean after
de-correlating and unit-scaling by the covariance `Σ`. Use it when summary keys
move together — plain Euclidean counts a shared signal once per key, several
times over.

`Σ` must be symmetric positive definite (only one triangle is read, as with
`Symmetric`). Build it from `cov(X; dims = 1)` for a one-off, or — better — from
[`make_mahalanobis`](@ref), which shrinks the estimate and factorises it once.
The `Cholesky` form skips re-factorisation when the same covariance is reused;
that is what the callable returned by `make_mahalanobis` calls internally.

```julia
Σ = [4.0 1.0; 1.0 2.0]
mahalanobis_distance([1.0, 1.0], [0.0, 0.0], Σ)     # 0.7559…
```

Named summaries are matched by key, exactly as in [`l2_distance`](@ref).
"""
function mahalanobis_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                              Σ::AbstractMatrix{<:Real})
    length(a) == length(b) || throw(DimensionMismatch(
        "summaries have different lengths: $(length(a)) vs $(length(b))"))
    size(Σ, 1) == size(Σ, 2) || throw(DimensionMismatch(
        "Σ must be square, got size $(size(Σ))"))
    size(Σ, 1) == length(a) || throw(DimensionMismatch(
        "Σ is $(size(Σ, 1))×$(size(Σ, 2)) but the summaries have length $(length(a))"))
    F = try
        cholesky(Symmetric(Matrix{Float64}(Σ)))
    catch err
        err isa PosDefException || rethrow()
        throw(ArgumentError(
            "Σ is not positive definite, so its inverse does not exist. Raw " *
            "sample covariances are singular when observations are few or keys " *
            "are collinear; make_mahalanobis(X) shrinks them toward the " *
            "diagonal and returns a usable metric"))
    end
    return mahalanobis_distance(a, b, F)
end

function mahalanobis_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                              F::Cholesky)
    length(a) == length(b) || throw(DimensionMismatch(
        "summaries have different lengths: $(length(a)) vs $(length(b))"))
    size(F, 1) == length(a) || throw(DimensionMismatch(
        "the covariance factor is $(size(F, 1))×$(size(F, 1)) but the summaries " *
        "have length $(length(a))"))
    z = Float64.(a) .- b
    # z' Σ⁻¹ z = ‖L⁻¹ z‖² for the Cholesky factor; clip at 0 because floating
    # point can round a quadratic form very slightly negative.
    return sqrt(max(0.0, dot(z, F \ z)))
end

function mahalanobis_distance(a::Real, b::Real, σ::Real)
    σ > 0 || throw(ArgumentError("variance σ must be positive, got $σ"))
    return abs(a - b) / sqrt(σ)
end

function mahalanobis_distance(a::NamedTuple, b::NamedTuple, Σ)
    va, vb = _aligned_summary_vectors(a, b)
    return mahalanobis_distance(va, vb, Σ)
end

"""
    make_mahalanobis(X::AbstractMatrix{<:Real}; shrinkage = 0.1) -> MahalanobisDistance

Estimate the summary covariance from a pilot ensemble `X` — one row per
observation, one column per summary key, in `summary_keys` order — and return a
callable Mahalanobis distance with the covariance shrunk toward its diagonal
and factorised once:

```julia
# One summary vector per prior draw (or per real replicate); the seed is fixed
# because the pilot is a data choice, like the observed summary itself.
rng   = MersenneTwister(2026)
pilot = [sim.evolve(sim.model_builder(θ), rng) for θ in prior_draws]
X     = reduce(vcat, permutedims.(namedtuple_to_vector.(compute_summary.(base, pilot))))

d   = make_mahalanobis(X)                    # shrinkage 0.1 by default
cfg = ABCSMCConfig(prior, sample_prior, transform, d, GaussianKernel(), Σ, sim)
```

The estimate is the sample covariance shrunk toward its diagonal,

    Σ̂ = (1 - λ) · cov(X) + λ · Diagonal(cov(X)),

with `λ = shrinkage ∈ [0, 1]`. `λ = 0` is the raw sample covariance — exact, but
singular whenever observations are few relative to keys or keys are collinear,
the usual ABC regime; any `λ > 0` adds the diagonal back and makes the metric
well defined.
`λ = 1` drops the correlations entirely, leaving `1/σᵢ²` weights — the diagonal
Mahalanobis metric, expressible as [`weighted_l2`](@ref).

`X` must have at least 2 rows (a sample covariance needs `n - 1` denominators)
and finite entries. A constant column has zero variance and cannot be
regularised at any `λ` — drop it; collinear ones are exactly what the shrinkage
is for.
"""
function make_mahalanobis(X::AbstractMatrix{<:Real}; shrinkage::Real = 0.1)
    (isfinite(shrinkage) && 0 <= shrinkage <= 1) || throw(ArgumentError(
        "shrinkage must lie in [0, 1], got $shrinkage"))
    n, p = size(X)
    n >= 2 || throw(ArgumentError(
        "make_mahalanobis needs at least 2 rows (observations), got $n"))
    p >= 1 || throw(ArgumentError(
        "make_mahalanobis needs at least 1 column (summary key), got $p"))
    all(isfinite, X) || throw(ArgumentError(
        "X contains non-finite values (NaN or Inf); clean the pilot ensemble first"))

    Xf = Matrix{Float64}(X)
    μ = vec(sum(Xf; dims = 1)) ./ n
    Xc = Xf .- μ'
    C = (Xc' * Xc) ./ (n - 1)                     # sample covariance, as pca_reduce
    λ = Float64(shrinkage)
    Σ = (1 - λ) .* C .+ λ .* Diagonal(diag(C))

    F = try
        cholesky(Symmetric(Σ))
    catch err
        err isa PosDefException || rethrow()
        throw(ArgumentError(
            "the shrunk covariance is not positive definite (n = $n, p = $p, " *
            "shrinkage = $λ): a constant key has zero variance and cannot be " *
            "regularised at any shrinkage — drop it; otherwise raise " *
            "`shrinkage` (e.g. 0.5) or add observations"))
    end

    return MahalanobisDistance(Σ, F, λ, n)
end

"""
    MahalanobisDistance

Callable returned by [`make_mahalanobis`](@ref): the shrunk covariance estimate,
already factorised, plus the metadata it was made from. Fields: `Σ` (the shrunk
covariance), `chol` (its Cholesky factor), `shrinkage`, and `n` (rows of the
pilot matrix). Accepted summaries are the same as for
[`mahalanobis_distance`](@ref) — vectors, scalars (single-key fits), or named
summaries.
"""
struct MahalanobisDistance
    Σ::Matrix{Float64}
    chol::Cholesky{Float64,Matrix{Float64}}
    shrinkage::Float64
    n::Int
end

(d::MahalanobisDistance)(a, b) = mahalanobis_distance(a, b, d.chol)

function (d::MahalanobisDistance)(a::Real, b::Real)
    size(d.Σ, 1) == 1 || throw(DimensionMismatch(
        "this distance was fitted on $(size(d.Σ, 1)) keys; pass vectors or " *
        "named summaries"))
    return mahalanobis_distance(a, b, d.Σ[1, 1])
end

Base.show(io::IO, d::MahalanobisDistance) = print(io,
    "MahalanobisDistance(", size(d.Σ, 1), " keys, n = ", d.n,
    ", shrinkage = ", d.shrinkage, ")")
