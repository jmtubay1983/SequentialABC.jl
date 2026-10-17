# ── Kernel bandwidth and scaling helpers ─────────────────────────────────────
#
# Small, pure, RNG-free selection utilities that sit *next to* the adaptive
# kernel rather than inside it: an estimated covariance, a scaling rule and a
# per-dimension bandwidth are three independent choices, and this file lets a
# user make each one explicitly. Nothing here touches engine internals.

using LinearAlgebra

# Correlations below this are treated as "not worth conditioning on" by
# `method = :auto` in `make_adaptive_kernel`.
const _AUTO_CORR_THRESHOLD = 0.1

"""
    auto_kernel_scale(d::Integer; rule = :rosenthal, factor = 1.0) -> Float64

Scaling factor that turns an estimated `d`-dimensional covariance into a
proposal covariance — the same rules as [`scaling_factor`](@ref), plus two
bandwidth-selection heuristics whose only input is the dimension:

| `rule` | factor | use |
|---|---|---|
| `:rosenthal` (alias `:gelman`) | `2.38² / d` | classic optimal random‑walk scaling (Gelman, Roberts & Gilks 1996) |
| `:beaumont` (alias `:abc`) | `2` | ABC‑SMC perturbation scaling (Beaumont et al. 2009) |
| `:unit` (alias `:none`) | `1` | no scaling |
| `:silverman` | `(4/(d+2))^(1/(d+4))` | Silverman's rule‑of‑thumb bandwidth |
| `:median` | `2·median(χ²_d)/d` | dimension‑only median heuristic |
| `::Real` | the value | any custom factor |

The result is multiplied by `factor` (a positive real, default `1.0`), which is
the knob for a coarse global adjustment without touching the rule.

`:silverman` is Silverman's `(4/(d+2))^(1/(d+4))` bandwidth factor for a
Gaussian kernel. `:median` has no data to work with here, so it uses the median
of the chi-squared distribution with `d` degrees of freedom — the typical squared
distance of a standard Gaussian in `d` dimensions — giving `2·median(χ²_d)/d`
(for `d = 2` this is `2·log 2 ≈ 1.386`).

An unknown `rule`, a non‑positive `factor`, or `d < 1` raises an `ArgumentError`.
"""
function auto_kernel_scale(d::Integer; rule::Union{Symbol,Real} = :rosenthal,
                           factor::Real = 1.0)
    d >= 1 || throw(ArgumentError("dimension must be ≥ 1, got $d"))
    factor > 0 || throw(ArgumentError("factor must be > 0, got $factor"))
    isfinite(factor) || throw(ArgumentError("factor must be finite, got $factor"))

    if rule === :silverman
        return Float64(factor) * (4 / (d + 2))^(1 / (d + 4))
    elseif rule === :median
        return Float64(factor) * (2 * quantile(Chisq(d), 0.5) / d)
    end

    base = try
        scaling_factor(d; rule = rule)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("unknown scaling rule $(repr(rule)); use :rosenthal (:gelman), " *
                            ":beaumont (:abc), :unit (:none), :silverman, :median or a number"))
    end
    return Float64(factor) * base
end

"""
    shrinkage_covariance(X::AbstractMatrix; λ = 1e-6) -> Matrix{Float64}

Ledoit–Wolf‑style shrinkage covariance of the rows of `X` (an `N × d` matrix):
the unbiased sample covariance pulled towards isotropy,

```
Σ = (1 − λ) S + λ (tr S / d) I,
```

symmetrised. The target is the scalar mean of the sample variances, so `λ = 1`
gives a purely isotropic estimate and `λ = 0` the plain sample covariance.

`λ > 0` (the default `1e-6`) makes the result **positive definite even for a
rank‑deficient population** — repeated rows, `N ≤ d`, a single row — which is
what keeps a Cholesky factorisation of the proposal from failing. A population
with no variance information at all (fewer than two rows, or every row
identical) falls back to `λ · I`, mirroring
[`estimate_covariance`](@ref)'s degenerate‑population rule; `λ = 0` does not
guarantee positive definiteness. `λ` must lie in `[0, 1]`.
"""
function shrinkage_covariance(X::AbstractMatrix; λ::Real = 1e-6)
    (0 <= λ <= 1) || throw(ArgumentError("λ must be in [0, 1], got $λ"))
    Xm = Matrix{Float64}(X)
    N, d = size(Xm)
    d >= 1 || throw(ArgumentError("need at least one column, got size $(size(Xm))"))
    I_d = Matrix{Float64}(I, d, d)

    N >= 2 || return Float64(λ) .* I_d

    S = Matrix{Float64}(cov(Xm; dims = 1))
    S = (S + permutedims(S)) / 2
    μtr = tr(S) / d
    (isfinite(μtr) && μtr > 0) || return Float64(λ) .* I_d

    Σ = (1 - λ) .* S + (λ * μtr) .* I_d
    return (Σ + permutedims(Σ)) / 2
end

"""
    bandwidth_from_cov(cov; method = :diag, min_bandwidth = 1e-8) -> Vector{Float64}

Per‑dimension bandwidths (standard deviations) read off a covariance matrix.

  - `method = :diag` — the marginal standard deviations, `sqrt.(diag(cov))`.
  - `method = :full` — the diagonal of the Cholesky factor, i.e. the
    **conditional** standard deviations `sd(θᵢ | θ₁…θᵢ₋₁)` of the same Gaussian
    covariance. Reduces to `:diag` when `cov` is diagonal, and stays smaller when
    dimensions are correlated.

Every bandwidth is floored at `min_bandwidth` (default `1e-8`), so a zero‑variance
or near‑zero dimension still yields a usable positive scale. `cov` must be square
and symmetric; `method = :full` additionally requires positive definiteness, and
an unknown `method` or a non‑positive `min_bandwidth` raises an `ArgumentError`.
"""
function bandwidth_from_cov(cov::AbstractMatrix; method::Symbol = :diag,
                            min_bandwidth::Real = 1e-8)
    C = Matrix{Float64}(cov)
    d1, d2 = size(C)
    d1 == d2 || throw(ArgumentError("covariance must be square, got size $(size(C))"))
    min_bandwidth > 0 ||
        throw(ArgumentError("min_bandwidth must be > 0, got $min_bandwidth"))
    C = (C + permutedims(C)) / 2

    if method === :diag
        bw = sqrt.(max.(diag(C), 0.0))
    elseif method === :full
        F = cholesky(Symmetric(C); check = false)
        issuccess(F) || throw(ArgumentError(
            "covariance is not positive definite; cannot read :full bandwidths — " *
            "shrink it first or use method = :diag"))
        bw = max.(diag(F.L), 0.0)
    else
        throw(ArgumentError("unknown method $(repr(method)); use :diag or :full"))
    end
    return max.(bw, Float64(min_bandwidth))
end

"""
    TunedKernel

Callable returned by [`make_adaptive_kernel`](@ref): maps a particle population
to per‑dimension bandwidths. See that function for the exact pipeline.
"""
struct TunedKernel
    d::Int
    method::Symbol
    rule::Symbol
    min_scale::Float64
end

"""
    (k::TunedKernel)(particles) -> Vector{Float64}

Bandwidths for the population `particles` (an `N × d` matrix, or a length‑`N`
vector of parameter vectors — the layout of `ABCSMCState.θ`):

```
bandwidths = auto_kernel_scale(d; rule) .* bandwidth_from_cov(
                 shrinkage_covariance(X); method, min_bandwidth = min_scale)
```

floored elementwise at `min_scale`. A population whose dimension does not match
the config's raises a `DimensionMismatch`.
"""
function (k::TunedKernel)(particles)
    X = _kernel_matrix(particles)
    size(X, 2) == k.d || throw(DimensionMismatch(
        "tuned kernel is $(k.d)-dimensional but the population is $(size(X, 2))-dimensional"))
    method = k.method === :auto ? _auto_bandwidth_method(X) : k.method
    C = shrinkage_covariance(X)
    bw = bandwidth_from_cov(C; method = method, min_bandwidth = k.min_scale)
    return max.(auto_kernel_scale(k.d; rule = k.rule) .* bw, k.min_scale)
end

# `:auto` reads the correlation structure: condition on the full covariance when
# some pair of dimensions is materially correlated, and fall back to marginals
# when the population is effectively axis-aligned. A zero-variance dimension has
# no correlation to read, so it also falls back.
function _auto_bandwidth_method(X::AbstractMatrix)
    N, d = size(X)
    (d <= 1 || N < 2) && return :diag
    S = Matrix{Float64}(cov(Matrix{Float64}(X); dims = 1))
    sd = sqrt.(max.(diag(S), 0.0))
    all(>(0.0), sd) || return :diag
    R = S ./ (sd * permutedims(sd))
    for j in 1:d, i in (j + 1):d
        abs(R[i, j]) > _AUTO_CORR_THRESHOLD && return :full
    end
    return :diag
end

"""
    make_adaptive_kernel(cfg; method = :auto, min_scale = 1e-6,
                         rule = :rosenthal) -> TunedKernel

Build a callable that reads per‑dimension bandwidths off a particle population,
composing the three helpers above:

1. [`shrinkage_covariance`](@ref) of the population (so a degenerate population
   still yields a positive‑definite matrix),
2. [`bandwidth_from_cov`](@ref) with `method` (`:auto` picks `:full` when some pair
   of dimensions is correlated beyond `$( _AUTO_CORR_THRESHOLD )` and `:diag`
   otherwise),
3. scaling by [`auto_kernel_scale`](@ref)`(d; rule)` and flooring at `min_scale`.

`cfg` supplies the dimension (`size(cfg.Σ, 1)`), and is validated eagerly — an
unknown `method`/`rule`, `min_scale ≤ 0`, or a config without a covariance field
fails at construction rather than at the first call.

The result is a pure function of the particles: no RNG, no engine state, and the
same population always yields the same bandwidths. It is **not** wired into the
engines — install the bandwidths yourself through the existing covariance
plumbing:

```julia
tuner = make_adaptive_kernel(cfg)
bw    = tuner(state.θ)                                   # per-dimension scales
state.cfg = with_covariance(state.cfg, Matrix(Diagonal(bw .^ 2)))
```

For a directly *correlated* proposal, [`AdaptiveKernel`](@ref) already adapts the
full covariance; this helper is for callers that want explicit, inspectable
per‑dimension scales.
"""
function make_adaptive_kernel(cfg; method::Symbol = :auto, min_scale::Real = 1e-6,
                              rule::Symbol = :rosenthal)
    (method === :auto || method === :diag || method === :full) ||
        throw(ArgumentError("method must be :auto, :diag or :full, got $(repr(method))"))
    min_scale > 0 || throw(ArgumentError("min_scale must be > 0, got $min_scale"))
    hasproperty(cfg, :Σ) ||
        throw(ArgumentError("cfg must expose a covariance field Σ to fix the dimension"))
    d = size(Matrix{Float64}(cfg.Σ), 1)
    auto_kernel_scale(d; rule = rule)   # validate the rule eagerly
    return TunedKernel(d, method, rule, Float64(min_scale))
end
