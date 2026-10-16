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
# Only `LinearAlgebra`, `Statistics` and `Random` are used: no new dependencies.
# Private helpers are prefixed `_kernel_` so they cannot collide with helpers of
# other optional modules in this package.

using LinearAlgebra
using Statistics
using Random

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
isotropy.

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

# --- the kernel --------------------------------------------------------------

"""
    AdaptiveKernel(Σ0; shrink = 1e-6, rule = :rosenthal)

Gaussian random‑walk kernel `θ' ~ 𝒩(θ, Σ)` whose covariance is re‑estimated from
the particle population as a run progresses:

```julia
update!(kernel, population, weights)             # Σ ← scaling_factor(d; rule) · estimate_covariance(…)
state.cfg = with_covariance(state.cfg, kernel)   # ABC-SMC: pick up the new Σ
```

Fields:
  - `Σ::Matrix{Float64}`          — current proposal covariance
  - `Σ_inv::Matrix{Float64}`      — its inverse, refreshed by `update!`
  - `logdetΣ::Float64`            — `log det Σ`, refreshed by `update!`
  - `shrink::Float64`             — shrinkage applied to every estimate
  - `rule::Union{Symbol,Float64}` — scaling rule, see [`scaling_factor`](@ref)
  - `n_updates::Int`              — how many covariance updates were applied

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
end

function AdaptiveKernel(Σ; shrink::Real = 1e-6, rule::Union{Symbol,Real} = :rosenthal)
    (0 <= shrink <= 1) || throw(ArgumentError("shrink must be in [0, 1], got $shrink"))
    Σm = Matrix{Float64}(Σ)
    size(Σm, 1) == size(Σm, 2) ||
        throw(ArgumentError("Σ must be square, got size $(size(Σm))"))
    isposdef(Symmetric(Σm)) ||
        throw(ArgumentError("initial covariance must be symmetric positive definite"))
    k = AdaptiveKernel(Σm, zeros(size(Σm)), 0.0, Float64(shrink),
                       rule isa Real ? Float64(rule) : rule, 0)
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
    update!(kernel::AdaptiveKernel, samples, weights = nothing) -> AdaptiveKernel

Re‑estimate the kernel covariance from a particle population,

`Σ ← scaling_factor(d; kernel.rule) · estimate_covariance(samples, weights; shrink = kernel.shrink)`

and refresh the cached `Σ_inv` / `logdetΣ`. Returns the kernel, so the call can
be chained.

A **degenerate** population (zero estimated variance — every particle identical)
is ignored: the kernel keeps its current covariance instead of collapsing to a
zero‑variance proposal that would freeze the sampler, and `n_updates` is left
unchanged. A population of the wrong dimension raises a `DimensionMismatch`.
"""
function update!(k::AdaptiveKernel, samples, weights = nothing)
    X = _kernel_matrix(samples)
    d = size(X, 2)
    size(k.Σ, 1) == d || throw(DimensionMismatch(
        "kernel is $(size(k.Σ, 1))-dimensional but the population is $d-dimensional"))

    w = _kernel_weights(weights, size(X, 1))
    Σhat, degenerate = _covariance_estimate(X, w, k.shrink)
    degenerate && return k                      # frozen population: keep Σ

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
type, so the assignment is type‑stable.
"""
with_covariance(cfg::ABCSMCConfig, Σ) =
    ABCSMCConfig(cfg.prior, cfg.sample_prior, cfg.transform, cfg.distance,
                 cfg.kernel, Matrix{Float64}(Σ), cfg.simulator)

with_covariance(cfg::ABCSMCConfig, kernel::AdaptiveKernel) =
    with_covariance(cfg, kernel.Σ)
