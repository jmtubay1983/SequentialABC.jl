# Plotting

SequentialABC.jl ships a **lightweight, optional** plotting API that reproduces
the common [pyABC](https://pyabc.readthedocs.io/) visualisations:

| Function | pyABC analogue | Shows |
|----------|----------------|-------|
| [`plot_marginals`](@ref) | `plot_kde_1d` | 1‑D marginal posteriors |
| [`plot_pairs`](@ref) | `plot_pairwise_heatmap` | pairwise joint posteriors (corner plot) |
| [`plot_trace`](@ref) | `plot_trace` | weighted parameter means per generation |
| [`plot_diagnostics`](@ref) | `plot_eps` / `plot_ess` | ε, ESS and attempt counts |
| [`plot_ppc`](@ref) | `plot_ppc` | posterior predictive check |

## Optional, lazy backends

Plotting backends are **not** dependencies of the package. `Plots.jl` (and
`StatsPlots.jl`, used for the corner plot) are loaded lazily — and only when a
plotting function is called with `backend = :plots` — via
[Requires.jl](https://github.com/JuliaPackaging/Requires.jl).

Calling a function with `backend = :none` returns a dependency‑free
[`PlotSpec`](@ref) describing the figure instead of rendering it, which keeps the
package test suite free of plotting dependencies.

## Setup

The plotting dependencies live in a small example environment:

```julia
using Pkg
Pkg.activate("examples/plots_env")
Pkg.instantiate()
```

## Usage

```julia
using SequentialABC
using Plots
using StatsPlots

# `history` is a vector of per-generation states — e.g. loaded from checkpoints:
#   history = [load_checkpoint(p, cfg) for (_, p) in list_checkpoints(ckpt)]
history = [load_checkpoint(p, cfg) for (_, p) in list_checkpoints(ckpt)]

p1 = plot_marginals(history; labels = ["μ", "σ"])
p2 = plot_pairs(history; labels = ["μ", "σ"])
p3 = plot_trace(history; labels = ["μ", "σ"])
p4 = plot_diagnostics(history)
p5 = plot_ppc(history, s_obs)

savefig(p1, "marginals.png")
```

A complete, runnable example that writes all five figures to `repro/plots/`
lives in `examples/plot_examples.jl`:

```bash
julia --project=examples/plots_env examples/plot_examples.jl
```

## API

```@docs
plot_marginals
plot_pairs
plot_trace
plot_diagnostics
plot_ppc
PlotSpec
```
