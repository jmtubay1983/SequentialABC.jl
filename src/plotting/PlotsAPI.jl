# src/plotting/PlotsAPI.jl
#
# Lightweight, optional plotting API for ABC‑SMC results.
#
# The heavy plotting backends (Plots.jl, StatsPlots.jl) are deliberately NOT
# dependencies of SequentialABC. They are loaded lazily as Julia extensions —
# only when a plotting function is called with `backend = :plots` and the
# corresponding extension (PlotsExt or StatsPlotsExt) is available.
# Calling with `backend = :none` returns a dependency‑free `PlotSpec` that
# describes the figure, which keeps the package test suite plotting‑free.
#
# The API mirrors the common pyABC visualisations:
#   * plot_marginals    — 1‑D marginal posteriors
#   * plot_pairs        — pairwise joint posteriors (corner plot)
#   * plot_trace        — weighted parameter means across generations
#   * plot_diagnostics  — ε, ESS and attempt counts across generations
#   * plot_ppc          — posterior predictive check vs the observed summary

"""
    PlotSpec

Dependency‑free description of a figure, returned by the plotting functions when
called with `backend = :none`.

Fields:
  - `kind::Symbol`        — one of `:marginals, :pairs, :trace, :diagnostics, :ppc`
  - `data::NamedTuple`    — the extracted, ready‑to‑plot data
  - `options::NamedTuple` — rendering options supplied by the caller

`PlotSpec` holds only plain arrays and scalars, so it can be stored, compared or
serialised without loading any plotting backend — useful for smoke tests.
"""
struct PlotSpec
    kind::Symbol
    data::NamedTuple
    options::NamedTuple
end

"""
    _plots_available() -> Bool

`true` when the PlotsExt extension has been loaded into the current session,
i.e. when `backend = :plots` can actually render figures.
"""
_plots_available() = Base.get_extension(SequentialABC, :PlotsExt) !== nothing

# --- data extraction helpers ---

_default_labels(d::Integer) = ["θ$j" for j in 1:d]

_particle_vector(θi) = θi isa AbstractVector ? collect(Float64, θi) : Float64[Float64(θi)]

_as_history(s::ABCSMCState) = ABCSMCState[s]
_as_history(v::AbstractVector) = v
_final_state(h) = last(_as_history(h))

function _theta_matrix(st)
    θ = st.θ
    rows = [_particle_vector(θ[i]) for i in eachindex(θ)]
    d = length(rows[1])
    A = Matrix{Float64}(undef, length(rows), d)
    @inbounds for i in eachindex(rows), j in eachindex(rows[1])
        A[i, j] = rows[i][j]
    end
    return A
end

_wmean(A::AbstractMatrix, w::AbstractVector) = [sum(A[:, j] .* w) for j in 1:size(A, 2)]

_scalar_summary(x::Real) = Float64(x)
_scalar_summary(x::AbstractVector) = Float64(first(x))
_scalar_summary(x) = Float64(x)

_resolve_labels(labels, d) = labels === nothing ? _default_labels(d) : collect(String, labels)

# --- public plotting functions ---

"""
    plot_marginals(history; backend=:plots, labels=nothing, nbins=30)

One‑dimensional marginal posteriors of every parameter, drawn as weighted
histograms with the posterior mean marked.

`history` is either a single [`ABCSMCState`](@ref) or a vector of per‑generation
states (the last generation is plotted). Returns a `Plots.jl` figure when
`backend = :plots`, or a [`PlotSpec`](@ref) when `backend = :none`.
"""
function plot_marginals(history; backend::Symbol = :plots, labels = nothing, nbins::Int = 30)
    st = _final_state(history)
    A = _theta_matrix(st)
    w = collect(Float64, st.w)
    d = size(A, 2)
    labs = _resolve_labels(labels, d)
    data = (labels = labs, values = A, weights = w, means = _wmean(A, w))
    opts = (nbins = nbins,)
    return _dispatch(:marginals, data, opts; backend = backend)
end

"""
    plot_pairs(history; backend=:plots, labels=nothing, seed=0)

Pairwise joint posteriors (a pyABC‑style corner plot). Particles are resampled
according to their importance weights with the deterministic `seed`, so the
figure is reproducible.

`history` is either a single [`ABCSMCState`](@ref) or a vector of per‑generation
states. Returns a `Plots.jl` figure when `backend = :plots`, or a
[`PlotSpec`](@ref) when `backend = :none`.
"""
function plot_pairs(history; backend::Symbol = :plots, labels = nothing, seed::Int = 0)
    st = _final_state(history)
    A = _theta_matrix(st)
    w = collect(Float64, st.w)
    d = size(A, 2)
    labs = _resolve_labels(labels, d)
    idx = systematic_resample(w, MersenneTwister(seed))
    data = (labels = labs, values = A, weights = w, resample = idx)
    opts = NamedTuple()
    return _dispatch(:pairs, data, opts; backend = backend, seed = seed)
end

"""
    plot_trace(history; backend=:plots, labels=nothing)

Weighted posterior mean of each parameter as a function of the generation index.
`history` must contain one state per generation (e.g. loaded from checkpoints).
Returns a `Plots.jl` figure when `backend = :plots`, or a [`PlotSpec`](@ref)
when `backend = :none`.
"""
function plot_trace(history; backend::Symbol = :plots, labels = nothing)
    states = collect(_as_history(history))
    d = size(_theta_matrix(states[1]), 2)
    labs = _resolve_labels(labels, d)
    gens = [st.gen for st in states]
    means = Matrix{Float64}(undef, length(states), d)
    for (r, st) in enumerate(states)
        means[r, :] = _wmean(_theta_matrix(st), collect(Float64, st.w))
    end
    data = (labels = labs, gens = gens, means = means)
    opts = NamedTuple()
    return _dispatch(:trace, data, opts; backend = backend)
end

"""
    plot_diagnostics(history; backend=:plots)

ABC‑SMC run diagnostics across generations: the tolerance ε, the effective
sample size (ESS) and the number of simulation attempts spent in each
generation. Returns a `Plots.jl` figure when `backend = :plots`, or a
[`PlotSpec`](@ref) when `backend = :none`.
"""
function plot_diagnostics(history; backend::Symbol = :plots)
    states = collect(_as_history(history))
    gens = [st.gen for st in states]
    epsilons = [st.ε for st in states]
    essv = [st.ess for st in states]
    cum = [st.n_attempts for st in states]
    attempts = [cum[1]; diff(cum)]
    data = (gens = gens, epsilons = epsilons, ess = essv, attempts = attempts)
    opts = NamedTuple()
    return _dispatch(:diagnostics, data, opts; backend = backend)
end

"""
    plot_ppc(history, summary_obs; backend=:plots, nbins=20)

Posterior predictive check: distribution of the simulated summaries of the final
population against the observed summary `summary_obs`. For multi‑component
summaries the first component is used. Returns a `Plots.jl` figure when
`backend = :plots`, or a [`PlotSpec`](@ref) when `backend = :none`.
"""
function plot_ppc(history, summary_obs; backend::Symbol = :plots, nbins::Int = 20)
    st = _final_state(history)
    simulated = [_scalar_summary(s) for s in st.summaries]
    observed = _scalar_summary(summary_obs)
    data = (simulated = simulated, observed = observed)
    opts = (nbins = nbins,)
    return _dispatch(:ppc, data, opts; backend = backend)
end

# --- backend dispatch ---

function _dispatch(kind::Symbol, data::NamedTuple, opts::NamedTuple; backend::Symbol, kwargs...)
    if backend === :none
        return PlotSpec(kind, data, opts)
    elseif backend === :plots
        ext = Base.get_extension(SequentialABC, :PlotsExt)
        ext === nothing && error(
            "backend = :plots requires Plots.jl; run `using Plots` " *
            "(and `using StatsPlots` for pairwise plots) first, or use backend = :none."
        )
        return ext._plot(kind, data, opts; kwargs...)
    else
        throw(ArgumentError("unknown plotting backend $(repr(backend)); use :plots or :none"))
    end
end
