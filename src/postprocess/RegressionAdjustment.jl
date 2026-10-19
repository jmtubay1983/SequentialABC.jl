# src/postprocess/RegressionAdjustment.jl
#
# Regression adjustment postprocessing for ABC posteriors.
#
# Accepting particles with a finite tolerance ε distorts the posterior: the
# accepted parameters are correlated with the deviation of their simulated
# summaries from the observed summary, which biases the posterior and inflates
# its variance. Regression adjustment removes the leading part of that
# distortion by regressing the parameters on the summaries:
#
#   OLS           (Beaumont, Zhang & Balding 2002):
#       θ*_i = θ_i − β̂ᵀ (s_i − s_obs),  β̂ from a weighted linear fit of θ on s
#
#   local linear  (Blum & François 2010):
#       the same, but β̂ is a *kernel-weighted* fit centred at s_obs, which also
#       corrects the heteroscedasticity of the ABC acceptance kernel.
#
# Both estimators are weighted least squares: fully deterministic, and built
# only on LinearAlgebra / Statistics. No heavy dependency (e.g. GLM) is needed,
# so nothing has to be loaded behind Requires.jl.

using LinearAlgebra
using Statistics

const _KERNELS = (:epanechnikov, :gaussian, :uniform)

"""
    RegressionAdjustment

Result of the array-level [`adjusted_posterior`](@ref). Holds the adjusted
particles together with the fitted regression.

Fields:
  - `θ::Matrix{Float64}`                  — adjusted parameters (`N × d`)
  - `weights::Vector{Float64}`            — importance weights (normalised)
  - `method::Symbol`                      — `:ols` or `:local_linear`
  - `kernel::Symbol`                      — kernel used (`:local_linear` only)
  - `bandwidth::Union{Nothing,Float64}`   — kernel bandwidth (`:local_linear` only)
  - `slope::Matrix{Float64}`              — estimated regression slopes (`q × d`)
"""
struct RegressionAdjustment
    θ::Matrix{Float64}
    weights::Vector{Float64}
    method::Symbol
    kernel::Symbol
    bandwidth::Union{Nothing,Float64}
    slope::Matrix{Float64}
end

# --- input coercion ----------------------------------------------------------

_as_matrix(A::AbstractMatrix) = Matrix{Float64}(A)
function _as_matrix(A::AbstractVector)
    n = length(A)
    n == 0 && throw(ArgumentError("samples must be non-empty"))
    if A[1] isa Real
        return reshape(Float64[Float64(x) for x in A], :, 1)
    end
    d = length(A[1])
    M = Matrix{Float64}(undef, n, d)
    @inbounds for i in 1:n, j in 1:d
        M[i, j] = Float64(A[i][j])
    end
    return M
end

_as_summary(S::AbstractMatrix) = Matrix{Float64}(S)
function _as_summary(S::AbstractVector)
    n = length(S)
    n == 0 && throw(ArgumentError("summaries must be non-empty"))
    if S[1] isa Real
        return reshape(Float64[Float64(x) for x in S], :, 1)
    end
    q = length(S[1])
    M = Matrix{Float64}(undef, n, q)
    @inbounds for i in 1:n, j in 1:q
        M[i, j] = Float64(S[i][j])
    end
    return M
end

_as_obs(x::Real) = [Float64(x)]
function _as_obs(x::AbstractVector)
    isempty(x) && throw(ArgumentError("observed summary must be non-empty"))
    return Float64.(collect(x))
end

_uniform_weights(n::Integer) = fill(1.0 / n, n)

function _weights_or_uniform(w, n::Integer)
    w === nothing && return _uniform_weights(n)
    ww = Float64.(collect(w))
    length(ww) == n || throw(DimensionMismatch(
        "weights length $(length(ww)) does not match N = $n"))
    total = sum(ww)
    total > 0 || throw(ArgumentError("weights must sum to a positive value"))
    return ww ./ total
end

function _check_shapes(Θ::AbstractMatrix, S::AbstractMatrix, so::AbstractVector)
    n = size(Θ, 1)
    size(S, 1) == n || throw(DimensionMismatch(
        "summaries has $(size(S, 1)) rows but samples has $n"))
    length(so) == size(S, 2) || throw(DimensionMismatch(
        "observed summary has length $(length(so)) but summaries has $(size(S, 2)) columns"))
    n >= 2 || throw(ArgumentError("need at least 2 samples to adjust"))
    return nothing
end

# --- kernels & bandwidth -----------------------------------------------------

function _kernel_values(kernel::Symbol, u::AbstractVector{Float64})
    if kernel === :epanechnikov
        return [abs(x) <= 1 ? 0.75 * (1 - x^2) : 0.0 for x in u]
    elseif kernel === :gaussian
        return exp.(-0.5 .* u .^ 2)
    elseif kernel === :uniform
        return [abs(x) <= 1 ? 1.0 : 0.0 for x in u]
    else
        throw(ArgumentError("unknown kernel $(repr(kernel)); use one of $(_KERNELS)"))
    end
end

function _distances(S::AbstractMatrix, so::AbstractVector)
    return [sqrt(sum(abs2, view(S, i, :) .- so)) for i in 1:size(S, 1)]
end

# Largest weighted per-column standard deviation of the summaries.
function _summary_scale(S::AbstractMatrix, w::AbstractVector)
    scale = 0.0
    for j in 1:size(S, 2)
        col = view(S, :, j)
        m = sum(w .* col)
        scale = max(scale, sqrt(max(sum(w .* (col .- m) .^ 2), 0.0)))
    end
    return scale
end

# Smallest bandwidth whose kernel window still contains ≥ q + 2 particles.
function _min_bandwidth(S::AbstractMatrix, so::AbstractVector, q::Integer)
    d = sort!(_distances(S, so))
    need = min(length(d), q + 2)
    h = d[need]
    return h > 0 ? h * (1 + 1e-9) : 1.0
end

function _default_bandwidth(S::AbstractMatrix, so::AbstractVector,
                            w::AbstractVector, kernel::Symbol)
    n = size(S, 1)
    h_rule = 1.06 * _summary_scale(S, w) * n^(-1 / 5)
    if kernel === :gaussian
        return h_rule > 0 ? h_rule : 1.0
    end
    # bounded-support kernels: make sure the window is populated
    h = max(h_rule, _min_bandwidth(S, so, size(S, 2)))
    return h > 0 ? h : 1.0
end

# --- core weighted least squares --------------------------------------------

# Weighted least squares of θ on the summaries centred at s_obs.
# Returns (θ_adjusted, slopes) with slopes of size q × d.
function _linear_adjust(Θ::AbstractMatrix, S::AbstractMatrix, so::AbstractVector,
                        wreg::AbstractVector)
    n = size(Θ, 1)
    sc = S .- permutedims(so)                       # N × q centred summaries
    X = hcat(ones(n), sc)                           # N × (1 + q) design
    A = X' * Diagonal(wreg) * X
    A = A + eps(Float64) * (1 + opnorm(A, 1)) * I   # tiny ridge: numerical safety
    β = A \ (X' * (wreg .* Θ))                      # (1 + q) × d
    slopes = β[2:end, :]                            # q × d
    return Θ .- sc * slopes, slopes
end

# Shared local-linear fit: returns (adjusted, slopes, bandwidth).
function _local_linear(Θ::AbstractMatrix, S::AbstractMatrix, so::AbstractVector,
                       ww::AbstractVector, kernel::Symbol, h)
    kernel in _KERNELS || throw(ArgumentError(
        "unknown kernel $(repr(kernel)); use one of $(_KERNELS)"))
    hh = h === nothing ? _default_bandwidth(S, so, ww, kernel) : Float64(h)
    hh > 0 || throw(ArgumentError("bandwidth h must be positive"))
    kw = ww .* _kernel_values(kernel, _distances(S, so) ./ hh)
    if count(>(0.0), kw) < size(S, 2) + 2
        # enlarge the bandwidth so the fit stays identifiable
        hh = _min_bandwidth(S, so, size(S, 2))
        kw = ww .* _kernel_values(kernel, _distances(S, so) ./ hh)
    end
    adj, slopes = _linear_adjust(Θ, S, so, kw)
    return adj, slopes, hh
end

# --- public API --------------------------------------------------------------

"""
    ols_adjust(samples, summaries, observed_summary; w=nothing) -> Matrix{Float64}

Linear (OLS) regression adjustment of ABC particles (Beaumont, Zhang & Balding
2002).

`samples` is the `N × d` matrix of accepted parameters (a vector of parameter
vectors or a vector of scalars is also accepted), `summaries` the `N × q` matrix
of their simulated summaries and `observed_summary` the observed summary. The
optional importance weights `w` give a weighted least-squares fit. Returns the
adjusted `N × d` parameters

    θ*_i = θ_i − β̂ᵀ (s_i − s_obs).
"""
function ols_adjust(samples, summaries, observed_summary; w = nothing)
    Θ = _as_matrix(samples)
    S = _as_summary(summaries)
    so = _as_obs(observed_summary)
    _check_shapes(Θ, S, so)
    ww = _weights_or_uniform(w, size(Θ, 1))
    adj, _ = _linear_adjust(Θ, S, so, ww)
    return adj
end

"""
    local_linear_adjust(samples, summaries, observed_summary;
                        w=nothing, kernel=:epanechnikov, h=nothing) -> Matrix{Float64}

Local-linear regression adjustment (Blum & François 2010): the same correction
as [`ols_adjust`](@ref), but the slopes β̂ come from a **kernel-weighted** fit
centred at `observed_summary`, which also corrects the heteroscedasticity of the
ABC acceptance kernel.

`kernel` is one of `:epanechnikov` (default), `:gaussian` or `:uniform`. `h` is
the bandwidth; when omitted it defaults to Silverman's rule-of-thumb on the
summary scale, enlarged if necessary so the window contains at least `q + 2`
particles. Returns the adjusted `N × d` parameters.
"""
function local_linear_adjust(samples, summaries, observed_summary;
                             w = nothing, kernel::Symbol = :epanechnikov, h = nothing)
    Θ = _as_matrix(samples)
    S = _as_summary(summaries)
    so = _as_obs(observed_summary)
    _check_shapes(Θ, S, so)
    ww = _weights_or_uniform(w, size(Θ, 1))
    adj, _, _ = _local_linear(Θ, S, so, ww, kernel, h)
    return adj
end

"""
    adjusted_posterior(result; summaries_fn, observed_summary, method=:ols, kwargs...)

Regression-adjust an existing sampler result and return a result of the **same
structure** with the particles replaced by their adjusted values.

  - `result`           — an [`ABCSMCState`](@ref) (a `NamedTuple` with `θ`,
    `summaries` and optionally `w` fields also works);
  - `summaries_fn`     — maps one stored summary to a vector of regressors,
    e.g. `s -> [s]` for scalar summaries (`identity` for vector summaries);
  - `observed_summary` — the summary the run was conditioned on;
  - `method`           — `:ols` or `:local_linear`;
  - `kwargs...`        — forwarded to [`local_linear_adjust`](@ref)
    (`kernel`, `h`) when `method = :local_linear`.

```julia
adjusted = adjusted_posterior(state;
                              summaries_fn = s -> [s],
                              observed_summary = s_obs,
                              method = :local_linear)
```

The complementary array-level method

    adjusted_posterior(samples, summaries, observed_summary; method=:ols, w=nothing,
                       kernel=:epanechnikov, h=nothing) -> RegressionAdjustment

returns a [`RegressionAdjustment`](@ref) with the adjusted particles together
with the fitted slopes and bandwidth.
"""
function adjusted_posterior(result;
                            summaries_fn = _default_summaries_fn,
                            observed_summary = nothing,
                            method::Symbol = :ols,
                            kwargs...)
    observed_summary === nothing && throw(ArgumentError("`observed_summary` is required"))
    hasproperty(result, :θ) || throw(ArgumentError(
        "expected a result object with a `θ` field, got $(typeof(result))"))
    hasproperty(result, :summaries) || throw(ArgumentError(
        "expected a result object with a `summaries` field, got $(typeof(result))"))
    Θ = _as_matrix(result.θ)
    S = _summary_matrix(result.summaries, summaries_fn)
    so = _as_obs(observed_summary)
    _check_shapes(Θ, S, so)
    ww = _weights_or_uniform(hasproperty(result, :w) ? result.w : nothing, size(Θ, 1))
    θadj = _adjust(Θ, S, so, method, ww, kwargs)
    return _replace_samples(result, θadj)
end

function adjusted_posterior(samples, summaries, observed_summary;
                            method::Symbol = :ols, w = nothing,
                            kernel::Symbol = :epanechnikov, h = nothing)
    Θ = _as_matrix(samples)
    S = _as_summary(summaries)
    so = _as_obs(observed_summary)
    _check_shapes(Θ, S, so)
    ww = _weights_or_uniform(w, size(Θ, 1))
    if method === :ols
        adj, slopes = _linear_adjust(Θ, S, so, ww)
        return RegressionAdjustment(adj, ww, :ols, :none, nothing, slopes)
    elseif method === :local_linear
        adj, slopes, hh = _local_linear(Θ, S, so, ww, kernel, h)
        return RegressionAdjustment(adj, ww, :local_linear, kernel, hh, slopes)
    else
        throw(ArgumentError("unknown method $(repr(method)); use :ols or :local_linear"))
    end
end

# --- result helpers ----------------------------------------------------------

_default_summaries_fn(s) = s isa Real ? [Float64(s)] : collect(Float64, s)

function _summary_matrix(summaries, summaries_fn)
    n = length(summaries)
    n == 0 && throw(ArgumentError("result has no summaries to adjust"))
    rows = [collect(Float64, summaries_fn(s)) for s in summaries]
    q = length(rows[1])
    M = Matrix{Float64}(undef, n, q)
    for i in 1:n
        length(rows[i]) == q || throw(DimensionMismatch(
            "summary $i has length $(length(rows[i])), expected $q"))
        @views M[i, :] .= rows[i]
    end
    return M
end

function _adjust(Θ::AbstractMatrix, S::AbstractMatrix, so::AbstractVector,
                 method::Symbol, ww::AbstractVector, kwargs)
    if method === :ols
        isempty(kwargs) || throw(ArgumentError(
            "method = :ols does not accept keyword arguments $(collect(keys(kwargs)))"))
        return ols_adjust(Θ, S, so; w = ww)
    elseif method === :local_linear
        return local_linear_adjust(Θ, S, so; w = ww, kwargs...)
    else
        throw(ArgumentError("unknown method $(repr(method)); use :ols or :local_linear"))
    end
end

function _replace_samples(result::ABCSMCState, θadj::AbstractMatrix{Float64})
    θnew = [θadj[i, :] for i in 1:size(θadj, 1)]
    # Plain state: the adjusted samples carry no adaptive runtime state
    # (postprocess integration with adaptive runs is deferred). Post-processing
    # an adaptive run currently drops `adaptive = nothing` — the caller should
    # interpret adjusted results in the non-adaptive sense and should not expect
    # the original adaptive distance state to carry over to subsequent adjusted
    # inference steps.
    return ABCSMCState(result.cfg, θnew, result.w, result.summaries, result.dists,
                       result.ε, result.gen, result.ess, result.base_seed,
                       result.n_attempts, result.n_resamples, nothing)
end

function _replace_samples(result::NamedTuple, θadj::AbstractMatrix{Float64})
    θnew = [θadj[i, :] for i in 1:size(θadj, 1)]
    return merge(result, (θ = θnew,))
end

_replace_samples(result, θadj::AbstractMatrix{Float64}) = throw(ArgumentError(
    "don't know how to rebuild a $(typeof(result)); supported results are " *
    "ABCSMCState and NamedTuple"))
