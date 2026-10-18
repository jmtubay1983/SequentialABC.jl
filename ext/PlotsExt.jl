module PlotsExt

using SequentialABC
using Plots

"""
    _plot(kind::Symbol, data::NamedTuple, opts::NamedTuple; seed, kwargs...) -> Plots figure

Render a plot of the given `kind` using Plots.jl. Called by
`SequentialABC._dispatch` when `backend = :plots` and the PlotsExt extension
is loaded.
"""
function _plot(kind::Symbol, data::NamedTuple, opts::NamedTuple; seed::Int = 0, kwargs...)
    if kind === :marginals
        return _plot_marginals(data, opts; kwargs...)
    elseif kind === :pairs
        return _plot_pairs(data, opts; seed = seed)
    elseif kind === :trace
        return _plot_trace(data, opts)
    elseif kind === :diagnostics
        return _plot_diagnostics(data, opts)
    elseif kind === :ppc
        return _plot_ppc(data, opts; kwargs...)
    end
    throw(ArgumentError("unknown plot kind $(repr(kind))"))
end

# --- marginals ---

function _plot_marginals(data::NamedTuple, opts::NamedTuple; nbins::Int = 30, kwargs...)
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
    layout_opts = (layout = (1, d), size = (360 * d, 340),
                   plot_title = "Marginal posteriors", top_margin = 8Plots.mm,
                   left_margin = 5Plots.mm, right_margin = 3Plots.mm,
                   bottom_margin = 6Plots.mm)
    d == 1 && (layout_opts = merge(layout_opts, (plot_titlevspan = 0.12,)))
    return Plots.plot(ps...; layout_opts...)
end

# --- pairs ---

function _plot_pairs(data::NamedTuple, _opts::NamedTuple; seed::Int = 0)
    d = size(data.values, 2)
    A = data.values[data.resample, :]
    ste = Base.get_extension(SequentialABC, :StatsPlotsExt)
    if ste !== nothing && d > 1
        return ste._plot_pairs_corner(A)
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
end

# --- trace ---

function _plot_trace(data::NamedTuple, _opts::NamedTuple)
    return Plots.plot(data.gens, data.means;
        label = permutedims(data.labels), marker = :circle, linewidth = 2,
        framestyle = :box,
        xlabel = "generation", ylabel = "posterior mean",
        plot_title = "Parameter trace (weighted means)")
end

# --- diagnostics ---

function _plot_diagnostics(data::NamedTuple, _opts::NamedTuple)
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
end

# --- ppc ---

function _plot_ppc(data::NamedTuple, opts::NamedTuple; nbins::Int = 20)
    nb = get(opts, :nbins, 20)
    p = Plots.histogram(data.simulated; bins = nb, normalize = :pdf, alpha = 0.6,
                        framestyle = :box, label = "posterior predictive",
                        xlabel = "summary",
                        ylabel = "density", plot_title = "Posterior predictive check")
    Plots.vline!(p, [data.observed]; label = "observed", color = :red, linewidth = 2)
    return p
end

end