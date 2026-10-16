# src/kernels/AdaptiveKernel.jl
#
# Adaptive multivariate kernels for ABC-SMC and ABC-MCMC.
#
# Both engines move particles with a Gaussian random walk, θ' ~ 𝒩(θ, Σ): the
# ABC-SMC *perturbation kernel* and the ABC-MCMC *proposal*. Fixing Σ in advance
# is the classic pain point of ABC — too small and the population cannot move
# (particles stay near-duplicates of their ancestors), too large and almost every
# proposal misses the acceptance region and the sampler wastes simulations on
# redraws. The standard remedy is to estimate Σ from the current particle
# population and update it as the run progresses.
#
# Two estimators are provided: a batch one (`estimate_covariance`, one call per
# population) and a streaming one (`OnlineCovEstimator`, one call per particle,
# O(d²) memory). Both use the same weighted reliability normalisation, so they
# agree to floating-point precision.
#
# Only `LinearAlgebra`, `Statistics` and `Random` are used: no new dependencies.
# Private helpers are prefixed `_kernel_` so they cannot collide with helpers of
# other optional modules in this package.

using LinearAlgebra
using Statistics
using Random

# `cov` is extended for the estimator and the kernel below, so the binding is
# pulled in explicitly (a `using`-imported name cannot be extended).
import Statistics: cov

# --- small helpers -----------------------------------------------------------

_kernel_vector(θ) = isa(θ, AbstractVector) ? collect(Float64, θ) : [Float64(θ)]

# Accept either an N × d matrix or a length-N vector of parameter vectors (the
# layout of `ABCSMCState.θ`).
_kernel_matrix(X::AbstractMatrix) = Matrix{Float64}(X)

function _kernel_matrix(v::AbstractVector)
    isempty(v) && throw(ArgumentError("need at least one sample"))
    return permutedims(reduce(hcat, [_kernel_vector(x) for x in v]))
end

function _kernel_weights(weights, N::Integer)
    w = weights === nothing ? fill(1.0 / N, N) : collect(Float64, weights)
    length(w) == N || throw(DimensionMismatch("got $(length(w)) weights for $N samples"))
    any(<(0.0), w) && throw(ArgumentError("weights must be non-negative"))
    all(isfinite, w) || throw(ArgumentError("weights must be finite"))
    s = sum(w)
    s > 0 || throw(ArgumentError("weights must not sum to zero"))
    return w ./ s
end

# --- scaling rules -----------------------------------------------------------

"""
    scaling_factor(d::Integer; rule = :rosenthal) -> Float64

Multiplier that turns an estimated covariance into a proposal covariance.

| `rule` | factor | use |
|---|---|---|
| `:rosenthal` (alias `:gelman`) | `2.38² / d` | classic optimal random‑walk scaling for MCMC (Gelman, Roberts & Gilks 1996) |
| `:beaumont` (alias `:abc`) | `2.0` | twice the population covariance, the usual ABC‑SMC perturbation kernel (Beaumont et al. 2009) |
| `:unit` (alias `:none`) | `1.0` | proposal covariance = estimated covariance |
| `::Real` | the value | any custom factor |

Any other `rule` raises an `ArgumentError`.
"""
function scaling_factor(d::Integer; rule::Union{Symbol,Real} = :rosenthal)
    d >= 1 || throw(ArgumentError("dimension must be ≥ 1, got $d"))
    rule isa Real && return Float64(rule)
    (rule === :rosenthal || rule === :gelman) && return 2.38^2 / d
    (rule === :beaumont || rule === :abc) && return 2.0
    (rule === :unit || rule === :none) && return 1.0
    throw(ArgumentError("unknown scaling rule $(repr(rule)); " *
                        "use :rosenthal, :beaumont, :unit or a number"))
end

# Weighted covariance core. Returns `(Σ, degenerate)`; `degenerate` is `true`
# when the population carries no variance information at all (a single sample, or
# every sample identical), in which case `Σ = shrink · I`.
function _covariance_estimate(X::AbstractMatrix, w::AbstractVector, shrink::Real)
    N, d = size(X)
    I_d = Matrix{Float64}(I, d, d)

    denom = 1 - sum(abs2, w)
    denom > 1e-12 || return (Float64(shrink) .* I_d, true)

    # Deviations are accumulated around the first sample rather than around the
    # weighted mean. The shift cancels in the centred sums, so the estimate is
    # mathematically identical, but it avoids the cancellation error of forming
    # the mean first: an exactly constant population then yields an exactly zero
    # covariance and is detected as degenerate below instead of being mistaken
    # for one with variance ~1e-32.
    pivot = vec(X[1, :])
    D = X .- permutedims(pivot)
    μ = vec(sum(D .* w, dims = 1))
    Dc = D .- permutedims(μ)
    Σhat = (permutedims(Dc) * (Dc .* w)) / denom
    Σhat = (Σhat + permutedims(Σhat)) / 2

    μtr = tr(Σhat) / d
    μtr > 0 || return (Float64(shrink) .* I_d, true)

    Σ = (1 - shrink) .* Σhat + (shrink * μtr) .* I_d
    return ((Σ + permutedims(Σ)) / 2, false)
end

"""
    estimate_covariance(samples, weights = nothing; shrink = 1e-6) -> Matrix{Float64}

Weighted covariance of a particle population, symmetrised and shrunk towards
isotropy. This is the **batch** estimator: it takes the whole population at once.
For per‑particle streaming updates use [`OnlineCovEstimator`](@ref); the two
agree to floating‑point precision.

  - `samples` — an `N × d` matrix, or a length‑`N` vector of parameter vectors
    (exactly the layout of `ABCSMCState.θ`)
  - `weights` — importance weights; `nothing` means uniform
  - `shrink`  — shrinkage intensity `λ ∈ [0, 1]` towards `(tr Σ̂ / d) I`

The normalisation uses reliability weights, `1 - Σᵢ wᵢ²` with the weights
normalised to sum to one, which reduces to the unbiased `N - 1` denominator for
uniform weights.

Shrinkage makes the estimate **positive definite even for a rank‑deficient
population** (`N ≤ d`, duplicate particles, …), which is what keeps the Cholesky
factorisation of the perturbation step from failing. Fully degenerate inputs (a
single sample, or every sample identical) fall back to `shrink · I`; the result
is always symmetric. [`update!`](@ref) detects that fallback and refuses to adopt
it, so a frozen population cannot collapse the kernel to zero variance.
"""
function estimate_covariance(samples, weights = nothing; shrink::Real = 1e-6)
    (0 <= shrink <= 1) || throw(ArgumentError("shrink must be in [0, 1], got $shrink"))
    X = _kernel_matrix(samples)
    w = _kernel_weights(weights, size(X, 1))
    return first(_covariance_estimate(X, w, shrink))
end

"""
    adaptive_scale(Σ, d = size(Σ, 1); rule = :rosenthal) -> Matrix{Float64}

Scale an estimated covariance `Σ` into a proposal covariance,
`scaling_factor(d; rule) · Σ`. See [`scaling_factor`](@ref) for the available
rules.
"""
function adaptive_scale(Σ, d::Integer = size(Matrix{Float64}(Σ), 1); rule = :rosenthal)
    Σm = Matrix{Float64}(Σ)
    size(Σm, 1) == size(Σm, 2) ||
        throw(ArgumentError("Σ must be square, got size $(size(Σm))"))
    size(Σm, 1) == d ||
        throw(DimensionMismatch("Σ is $(size(Σm, 1))-dimensional but d = $d"))
    return scaling_factor(d; rule = rule) .* Σm
end

# --- online (streaming) covariance estimator ---------------------------------

"""
    OnlineCovEstimator(d::Integer; shrink = 1e-6)

Streaming weighted covariance estimator — Welford's algorithm for a `d`‑dimensional
weighted population, with `O(d²)` memory and no need to keep the samples.

```julia
est = OnlineCovEstimator(2)
for (θ, w) in zip(population, weights)
    update!(est, θ, w)          # incremental, one particle at a time
end
Σ̂ = cov(est)                    # the same estimate `estimate_covariance` would give
```

The weighted mean/covariance pair is updated as

```
W  += w;  W2 += w²
μ  += (w / W) (x − μ)
M2 += w (x − μ_old)(x − μ_new)ᵀ        # Welford, uses the updated mean
Σ   = M2 · W / (W² − W2)
```

so the denominator is the reliability normaliser `1 − Σw̃ᵢ²` (`w̃` normalised to sum
to one) written in unnormalised weights; for uniform weights it is the unbiased
`N − 1` denominator, exactly as in [`estimate_covariance`](@ref).

Numerical guards:

  - the mean is updated incrementally and the second moment uses both the old and
    the new mean, which avoids the cancellation of a naive `E[x xᵀ] − μμᵀ`;
  - `cov` shrinks towards isotropy (`λ = shrink`) and then applies an **eigenvalue
    floor** `λ_min = max(shrink, (tr Σ)/d · 10⁻⁸, eps())`, so the returned matrix
    is always symmetric positive definite — even after a single sample;
  - a population with fewer than two effective samples (or exactly coincident
    samples) reports a zero‑variance raw estimate, which `cov` floors and
    [`update!`](@ref)`(kernel, …)`/`commit!` refuse to adopt.

Fields: `d`, `μ::Vector{Float64}`, `M2::Matrix{Float64}`, `W::Float64`
(sum of weights), `W2::Float64` (sum of squared weights), `shrink::Float64`,
`n::Int` (number of particles seen). Use [`reset!`](@ref) to clear the state.
"""
mutable struct OnlineCovEstimator
    d::Int
    μ::Vector{Float64}
    M2::Matrix{Float64}
    W::Float64
    W2::Float64
    shrink::Float64
    n::Int
end

function OnlineCovEstimator(d::Integer; shrink::Real = 1e-6)
    d >= 1 || throw(ArgumentError("dimension must be ≥ 1, got $d"))
    (0 <= shrink <= 1) || throw(ArgumentError("shrink must be in [0, 1], got $shrink"))
    return OnlineCovEstimator(Int(d), zeros(d), zeros(d, d), 0.0, 0.0, Float64(shrink), 0)
end

Base.copy(est::OnlineCovEstimator) =
    OnlineCovEstimator(est.d, copy(est.μ), copy(est.M2), est.W, est.W2, est.shrink, est.n)

"""
    update!(est::OnlineCovEstimator, θ, w = 1.0) -> OnlineCovEstimator

Fold one particle `θ` with weight `w` into the streaming estimate. `w = 0` is a
no‑op; negative or non‑finite weights raise an `ArgumentError`, and a `θ` of the
wrong dimension raises a `DimensionMismatch`. Returns the estimator so calls can
be chained.
"""
function update!(est::OnlineCovEstimator, θ::AbstractVector{<:Real}, w::Real = 1.0)
    x = _kernel_vector(θ)
    length(x) == est.d || throw(DimensionMismatch(
        "estimator is $(est.d)-dimensional but the sample is $(length(x))-dimensional"))

    wf = Float64(w)
    isfinite(wf) || throw(ArgumentError("weight must be finite, got $w"))
    wf >= 0 || throw(ArgumentError("weight must be non-negative, got $w"))
    wf == 0 && return est

    W_new = est.W + wf
    est.W2 += wf^2
    δ = x .- est.μ
    est.μ .+= (wf / W_new) .* δ
    est.M2 .+= wf .* (δ * (x .- est.μ)')     # note: the *updated* mean
    est.W = W_new
    est.n += 1
    return est
end

# Raw (unregularised) estimate, or `nothing` when there is no variance
# information (fewer than two effective samples). Used by `commit!` to tell
# "genuinely zero variance" apart from "floored to being positive definite".
function _raw_cov(est::OnlineCovEstimator)
    denom = est.W^2 - est.W2
    denom > 0 || return nothing
    Σ = est.M2 .* (est.W / denom)
    return (Σ + permutedims(Σ)) / 2
end

"""
    cov(est::OnlineCovEstimator; shrink = est.shrink) -> Matrix{Float64}

Current covariance estimate: the reliability‑weighted covariance, shrunk towards
isotropy by `shrink`, with an eigenvalue floor of
`max(shrink, (tr Σ)/d · 10⁻⁸, eps())` so the result is always positive definite.
An empty or single‑sample estimator therefore returns a small isotropic matrix
rather than a singular one.
"""
function cov(est::OnlineCovEstimator; shrink::Real = est.shrink)
    (0 <= shrink <= 1) || throw(ArgumentError("shrink must be in [0, 1], got $shrink"))
    d = est.d
    I_d = Matrix{Float64}(I, d, d)

    raw = _raw_cov(est)
    Σ = raw === nothing ? zeros(d, d) : raw

    μtr = tr(Σ) / d
    Σ = (1 - shrink) .* Σ + (shrink * μtr) .* I_d

    # Eigenvalue floor: guarantees positive definiteness unconditionally.
    λfloor = max(Float64(shrink), μtr * 1e-8, eps(Float64))
    λmin = minimum(eigvals(Symmetric(Σ)))
    λmin < λfloor && (Σ = Σ .+ (λfloor - λmin) .* I_d)

    return (Σ + permutedims(Σ)) / 2
end

"""
    reset!(est::OnlineCovEstimator) -> OnlineCovEstimator

Clear the accumulated state (`μ`, `M2`, `W`, `W2`, `n`) so the estimator can be
reused for the next generation. The dimension and shrinkage are kept.
"""
function reset!(est::OnlineCovEstimator)
    fill!(est.μ, 0.0)
    fill!(est.M2, 0.0)
    est.W = 0.0
    est.W2 = 0.0
    est.n = 0
    return est
end

# --- the kernel --------------------------------------------------------------

"""
    AdaptiveKernel(Σ0; shrink = 1e-6, rule = :rosenthal)

Gaussian random‑walk kernel `θ' ~ 𝒩(θ, Σ)` whose covariance is re‑estimated from
the particle population as a run progresses.

**Batch** update — hand over the whole population:

```julia
update!(kernel, population, weights)             # Σ ← scaling_factor(d; rule) · estimate_covariance(…)
state.cfg = with_covariance(state.cfg, kernel)   # ABC-SMC: pick up the new Σ
```

**Streaming** update — feed particles as they are generated, which needs `O(d²)`
memory instead of keeping the population, then commit once per generation:

```julia
for (θ, w) in zip(state.θ, state.w)
    update!(kernel, θ, w)      # accumulates into an internal OnlineCovEstimator
end
commit!(kernel)                # adopt Σ, reset the accumulator
```

Committing at the *generation boundary* is deliberate: the SMC importance weights
of a generation are computed with the kernel of that generation, so a `Σ` that
changed halfway through would make the weights inconsistent. Accumulate online,
commit once, then perturb.

Fields:
  - `Σ::Matrix{Float64}`          — current proposal covariance
  - `Σ_inv::Matrix{Float64}`      — its inverse, refreshed by the update
  - `logdetΣ::Float64`            — `log det Σ`, refreshed by the update
  - `shrink::Float64`             — shrinkage applied to every estimate
  - `rule::Union{Symbol,Float64}` — scaling rule, see [`scaling_factor`](@ref)
  - `n_updates::Int`              — how many covariance updates were applied
  - `est::OnlineCovEstimator`     — accumulator for per‑particle updates

`Σ0` must be symmetric positive definite. Sampling from the kernel mirrors
`GaussianKernel`: `perturb(kernel, θ, rng)` draws a proposal and calling the
kernel, `kernel(θ_new, θ_old)`, evaluates the proposal log density.
"""
mutable struct AdaptiveKernel
    Σ::Matrix{Float64}
    Σ_inv::Matrix{Float64}
    logdetΣ::Float64
    shrink::Float64
    rule::Union{Symbol,Float64}
    n_updates::Int
    est::OnlineCovEstimator
end

function AdaptiveKernel(Σ; shrink::Real = 1e-6, rule::Union{Symbol,Real} = :rosenthal)
    (0 <= shrink <= 1) || throw(ArgumentError("shrink must be in [0, 1], got $shrink"))
    Σm = Matrix{Float64}(Σ)
    size(Σm, 1) == size(Σm, 2) ||
        throw(ArgumentError("Σ must be square, got size $(size(Σm))"))
    isposdef(Symmetric(Σm)) ||
        throw(ArgumentError("initial covariance must be symmetric positive definite"))
    k = AdaptiveKernel(Σm, zeros(size(Σm)), 0.0, Float64(shrink),
                       rule isa Real ? Float64(rule) : rule, 0,
                       OnlineCovEstimator(size(Σm, 1); shrink = shrink))
    return _refresh!(k)
end

# recompute the cached inverse and log-determinant after Σ changes
function _refresh!(k::AdaptiveKernel)
    k.Σ_inv = Matrix{Float64}(inv(cholesky(Symmetric(k.Σ))))
    k.logdetΣ = logdet(Symmetric(k.Σ))
    return k
end

"""
    perturb(kernel::AdaptiveKernel, θ_old, rng) -> Vector{Float64}

Draw `θ' ~ 𝒩(θ_old, Σ)` with the kernel's current covariance, using a Cholesky
factorisation exactly as the `GaussianKernel` method does. Internal helper, like
its `GaussianKernel` counterpart.
"""
function perturb(k::AdaptiveKernel, θ_old::AbstractVector, rng::AbstractRNG)
    L = cholesky(Symmetric(k.Σ)).L
    return θ_old .+ L * randn(rng, length(θ_old))
end

"""
    (kernel::AdaptiveKernel)(θ_new, θ_old) -> Float64

Log density of the current proposal `𝒩(θ_old, Σ)` at `θ_new`, including the
normalising constant — the same convention as `GaussianKernel`. The Gaussian
random walk is symmetric, so swapping the two arguments returns the same value.
"""
function (k::AdaptiveKernel)(θ_new::AbstractVector, θ_old::AbstractVector)
    diff = θ_new .- θ_old
    return -0.5 * diff' * k.Σ_inv * diff -
           0.5 * (length(diff) * log(2π) + k.logdetΣ)
end

"""
    cov(kernel::AdaptiveKernel) -> Matrix{Float64}

The covariance the kernel is currently proposing with (a copy, so mutating the
result cannot corrupt the kernel).
"""
cov(k::AdaptiveKernel) = copy(k.Σ)

"""
    update!(kernel::AdaptiveKernel, samples, weights = nothing) -> AdaptiveKernel
    update!(kernel::AdaptiveKernel, θ, w = 1.0) -> AdaptiveKernel
    update!(kernel::AdaptiveKernel, est::OnlineCovEstimator) -> AdaptiveKernel

Re‑estimate the kernel covariance from a particle population, then refresh the
cached `Σ_inv` / `logdetΣ`:

`Σ ← scaling_factor(d; kernel.rule) · estimate_covariance(samples, weights; shrink = kernel.shrink)`

Three forms:

  - `samples` is a whole population — an `N × d` matrix or the
    `Vector{<:AbstractVector}` layout of `ABCSMCState.θ` — applied immediately;
  - a single parameter **vector** (`AbstractVector{<:Real}`) with weight `w`
    is *accumulated* into the kernel's [`OnlineCovEstimator`](@ref); call
    [`commit!`](@ref) once the generation is complete to adopt it;
  - an `OnlineCovEstimator` is copied in and committed immediately (convenience
    for a caller that maintains its own estimator).

A **degenerate** population (zero estimated variance — every particle identical)
is ignored: the kernel keeps its current covariance instead of collapsing to a
zero‑variance proposal that would freeze the sampler, and `n_updates` is left
unchanged. A population of the wrong dimension raises a `DimensionMismatch`.
Returns the kernel, so the call can be chained.
"""
function update!(k::AdaptiveKernel, samples::AbstractVector{<:AbstractVector},
                 weights = nothing)
    return _kernel_batch_update!(k, _kernel_matrix(samples), weights)
end

function update!(k::AdaptiveKernel, samples::AbstractMatrix, weights = nothing)
    return _kernel_batch_update!(k, Matrix{Float64}(samples), weights)
end

function update!(k::AdaptiveKernel, θ::AbstractVector{<:Real}, w::Real = 1.0)
    update!(k.est, θ, w)
    return k
end

function update!(k::AdaptiveKernel, est::OnlineCovEstimator)
    est.d == size(k.Σ, 1) || throw(DimensionMismatch(
        "kernel is $(size(k.Σ, 1))-dimensional but the estimator is $(est.d)-dimensional"))
    k.est = copy(est)
    return commit!(k)
end

function _kernel_batch_update!(k::AdaptiveKernel, X::Matrix{Float64}, weights)
    d = size(X, 2)
    size(k.Σ, 1) == d || throw(DimensionMismatch(
        "kernel is $(size(k.Σ, 1))-dimensional but the population is $d-dimensional"))

    w = _kernel_weights(weights, size(X, 1))
    Σhat, degenerate = _covariance_estimate(X, w, k.shrink)
    reset!(k.est)
    degenerate && return k                      # frozen population: keep Σ

    return _adopt!(k, Σhat)
end

"""
    commit!(kernel::AdaptiveKernel) -> AdaptiveKernel

Adopt the covariance accumulated by per‑particle `update!(kernel, θ, w)` calls
(through the kernel's internal [`OnlineCovEstimator`](@ref)), refresh
`Σ_inv` / `logdetΣ`, increment `n_updates` and [`reset!`](@ref) the accumulator.

A degenerate accumulation — fewer than two effective samples, or a population of
exactly coincident particles — is ignored (the current `Σ` is kept and
`n_updates` is not incremented), so a frozen population cannot collapse the
kernel to zero variance. The accumulator is reset either way.
"""
function commit!(k::AdaptiveKernel)
    raw = _raw_cov(k.est)
    if raw === nothing || tr(raw) <= 0
        reset!(k.est)
        return k
    end
    Σhat = cov(k.est; shrink = k.shrink)
    reset!(k.est)
    return _adopt!(k, Σhat)
end

# adopt an already-regularised estimate as the proposal covariance
function _adopt!(k::AdaptiveKernel, Σhat::Matrix{Float64})
    d = size(k.Σ, 1)
    Σ_new = adaptive_scale(Σhat, d; rule = k.rule)
    (tr(Σ_new) <= 0 || !isposdef(Symmetric(Σ_new))) && return k

    k.Σ = Σ_new
    _refresh!(k)
    k.n_updates += 1
    return k
end

# --- engine integration ------------------------------------------------------

"""
    with_covariance(cfg::ABCSMCConfig, Σ) -> ABCSMCConfig
    with_covariance(cfg::ABCSMCConfig, kernel::AdaptiveKernel) -> ABCSMCConfig

Copy `cfg` with a new perturbation covariance `Σ` (recomputing `Σ_inv` and
`logdetΣ`).

`ABCSMCConfig` is immutable, so this is the integration point for adaptive
kernels: after `update!(kernel, state.θ, state.w)`, assign

```julia
state.cfg = with_covariance(state.cfg, kernel)
```

and the next `run_generation!` perturbs, weights and resamples with the adapted
covariance. `ABCSMCState` is mutable and `state.cfg` keeps the same concrete
type, so the assignment is type‑stable. The streaming form is identical, with
`commit!(kernel)` standing in for the `update!`:
`state.cfg = with_covariance(state.cfg, commit!(kernel))` after feeding the
particles one at a time.
"""
with_covariance(cfg::ABCSMCConfig, Σ) =
    ABCSMCConfig(cfg.prior, cfg.sample_prior, cfg.transform, cfg.distance,
                 cfg.kernel, Matrix{Float64}(Σ), cfg.simulator)

with_covariance(cfg::ABCSMCConfig, kernel::AdaptiveKernel) =
    with_covariance(cfg, kernel.Σ)
