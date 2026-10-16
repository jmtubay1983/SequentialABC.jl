# src/plotting/PlotsAPI.jl
#
# Lightweight, optional plotting API for ABC‑SMC results.
#
# The heavy plotting backends (Plots.jl, StatsPlots.jl) are deliberately NOT
# dependencies of SequentialABC. They are loaded lazily — and only when a
# plotting function is called with `backend = :plots` — through Requires.jl.
# Calling with `backend = :none` returns a dependency‑free `PlotSpec` that
# describes the figure, which keeps the package test suite plotting‑free.
#
# The API mirrors the common pyABC visualisations:
#   * plot_marginals    — 1‑D marginal posteriors
#   * plot_pairs        — pairwise joint posteriors (corner plot)
#   * plot_trace        — weighted parameter means across generations
#   * plot_diagnostics  — ε, ESS and attempt counts across generations
#   * plot_ppc          — posterior predictive check vs the observed summary

using Requires

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

const _PLOTS_MOD = Ref{Any}(nothing)
const _STATSPLOTS_MOD = Ref{Any}(nothing)

"""
    _plots_available() -> Bool

`true` when Plots.jl has been loaded into the current session, i.e. when
`backend = :plots` can actually render figures.
"""
_plots_available() = _PLOTS_MOD[] !== nothing

# Requires hooks: the block body runs the moment Plots / StatsPlots is loaded,
# whether that happens before or after SequentialABC.
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        _PLOTS_MOD[] = Plots
    end
    @require StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd" begin
        _STATSPLOTS_MOD[] = StatsPlots
    end
    return nothing
end

# --- data extraction helpers -------------------------------------------------

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
    @inbounds for i in eachindex(rows), j in 1:d
        A[i, j] = rows[i][j]
    end
    return A
end

_wmean(A::AbstractMatrix, w::AbstractVector) = [sum(A[:, j] .* w) for j in 1:size(A, 2)]

_scalar_summary(x::Real) = Float64(x)
_scalar_summary(x::AbstractVector) = Float64(first(x))
_scalar_summary(x) = Float64(x)

_resolve_labels(labels, d) = labels === nothing ? _default_labels(d) : collect(String, labels)

# --- public plotting functions ----------------------------------------------

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
    return _render(:marginals, data, (nbins = nbins,); backend = backend)
end

"""
    plot_pairs(history; backend=:plots, labels=nothing, seed=0)

Pairwise joint posteriors (a pyABC‑style corner plot). Particles are resampled
according to their importance weights with the deterministic `seed`, so the
figure is reproducible.

Requires StatsPlots.jl for the corner layout; if StatsPlots is not loaded a
plain scatter matrix of the Plots.jl backend is used instead. Returns a `Plots.jl`
figure when `backend = :plots`, or a [`PlotSpec`](@ref) when `backend = :none`.
"""
function plot_pairs(history; backend::Symbol = :plots, labels = nothing, seed::Int = 0)
    st = _final_state(history)
    A = _theta_matrix(st)
    w = collect(Float64, st.w)
    d = size(A, 2)
    labs = _resolve_labels(labels, d)
    idx = systematic_resample(w, MersenneTwister(seed))
    data = (labels = labs, values = A, weights = w, resample = idx)
    return _render(:pairs, data, NamedTuple(); backend = backend)
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
    return _render(:trace, data, NamedTuple(); backend = backend)
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
    return _render(:diagnostics, data, NamedTuple(); backend = backend)
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
    return _render(:ppc, data, (nbins = nbins,); backend = backend)
end

# --- backend dispatch --------------------------------------------------------

function _render(kind::Symbol, data::NamedTuple, opts::NamedTuple; backend::Symbol)
    if backend === :none
        return PlotSpec(kind, data, opts)
    elseif backend === :plots
        P = _PLOTS_MOD[]
        P === nothing && error(
            "backend = :plots requires Plots.jl; run `using Plots` " *
            "(and `using StatsPlots` for pairwise plots) first, or use backend = :none."
        )
        return _render_impl(kind, data, opts, P, _STATSPLOTS_MOD[])
    else
        throw(ArgumentError("unknown plotting backend $(repr(backend)); use :plots or :none"))
    end
end

function _render_impl(kind::Symbol, data::NamedTuple, opts::NamedTuple, Plots, StatsPlots)
    if kind === :marginals
        d = size(data.values, 2)
        nb = get(opts, :nbins, 30)
        ps = map(1:d) do j
            p = Plots.histogram(data.values[:, j];
                weights = data.weights, bins = nb, normalize = :pdf,
                alpha = 0.6, legend = false, framestyle = :box,
                title = data.labels[j], xlabel = data.labels[j], ylabel = "density")
            Plots.vline!(p, [data.means[j]]; color = :red, linewidth = 2)
            return p
        end
        return Plots.plot(ps...; layout = (1, d), size = (360 * d, 340),
                          plot_title = "Marginal posteriors", top_margin = 8Plots.mm,
                          left_margin = 5Plots.mm, right_margin = 3Plots.mm,
                          bottom_margin = 6Plots.mm)

    elseif kind === :pairs
        d = size(data.values, 2)
        A = data.values[data.resample, :]
        if StatsPlots !== nothing && d > 1
            try
                return StatsPlots.cornerplot(A; framestyle = :box)
            catch
                # fall back to a plain scatter matrix below
            end
        end
        if d == 1
            return Plots.histogram(A[:, 1]; normalize = :pdf, legend = false,
                                   framestyle = :box,
                                   title = data.labels[1], plot_title = "Pairwise posteriors")
        end
        pairs = [(i, j) for i in 1:d for j in (i + 1):d]
        ps = map(pairs) do (i, j)
            Plots.scatter(A[:, j], A[:, i]; markersize = 2, alpha = 0.5, legend = false,
                          framestyle = :box,
                          xlabel = data.labels[j], ylabel = data.labels[i])
        end
        return Plots.plot(ps...; layout = (1, length(ps)),
                          size = (320 * length(ps), 340), plot_title = "Pairwise posteriors",
                          left_margin = 5Plots.mm, right_margin = 3Plots.mm,
                          bottom_margin = 6Plots.mm)

    elseif kind === :trace
        return Plots.plot(data.gens, data.means;
            label = permutedims(data.labels), marker = :circle, linewidth = 2,
            framestyle = :box,
            xlabel = "generation", ylabel = "posterior mean",
            plot_title = "Parameter trace (weighted means)")

    elseif kind === :diagnostics
        p1 = Plots.plot(data.gens, data.epsilons; label = "ε", marker = :circle,
                        linewidth = 2, framestyle = :box,
                        xlabel = "generation", ylabel = "ε", title = "Tolerance")
        p2 = Plots.plot(data.gens, data.ess; label = "ESS", marker = :circle,
                        linewidth = 2, framestyle = :box,
                        xlabel = "generation", ylabel = "ESS",
                        title = "Effective sample size")
        p3 = Plots.plot(data.gens, data.attempts; label = "attempts", marker = :circle,
                        linewidth = 2, framestyle = :box,
                        xlabel = "generation", ylabel = "attempts",
                        title = "Simulation attempts")
        return Plots.plot(p1, p2, p3; layout = (1, 3), size = (960, 340),
                          plot_title = "ABC‑SMC diagnostics",
                          plot_titlefontsize = 14, plot_titlevspan = 0.14,
                          top_margin = 6Plots.mm,
                          left_margin = 5Plots.mm, right_margin = 3Plots.mm,
                          bottom_margin = 6Plots.mm)

    elseif kind === :ppc
        nb = get(opts, :nbins, 20)
        p = Plots.histogram(data.simulated; bins = nb, normalize = :pdf, alpha = 0.6,
                            framestyle = :box, label = "posterior predictive",
                            xlabel = "summary",
                            ylabel = "density", plot_title = "Posterior predictive check")
        Plots.vline!(p, [data.observed]; label = "observed", color = :red, linewidth = 2)
        return p

    else
        throw(ArgumentError("unknown plot kind $(repr(kind))"))
    end
end
