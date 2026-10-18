using Test
using Plots
using StatsPlots
using SequentialABC
using Random

# Tiny synthetic per-generation history — no simulator required, so the smoke
# tests stay fast and dependency-free.
function _synthetic_history(; N::Int = 40, d::Int = 2, G::Int = 3, seed::Int = 7)
    rng = MersenneTwister(seed)
    states = ABCSMCState[]
    for g in 1:G
        θ = [randn(rng, d) .+ 0.2 * g for _ in 1:N]
        w = fill(1.0 / N, N)
        sums = [sum(θ[i]) for i in 1:N]
        dists = abs.(sums)
        push!(states, ABCSMCState(nothing, θ, w, sums, dists, 0.5 / g, g, ess(w), seed, g * N, 0))
    end
    return states
end

@testset "Plotting API (smoke)" begin
    history = _synthetic_history()

    specs = (
        plot_marginals(history; backend = :none),
        plot_pairs(history; backend = :none),
        plot_trace(history; backend = :none),
        plot_diagnostics(history; backend = :none),
        plot_ppc(history, 0.0; backend = :none),
    )
    @test all(s -> s isa PlotSpec, specs)
    @test [s.kind for s in specs] == [:marginals, :pairs, :trace, :diagnostics, :ppc]

    # every function runs without error and returns ready-to-plot data
    m = plot_marginals(history; backend = :none)
    @test size(m.data.values) == (40, 2)
    @test length(m.data.means) == 2
    @test isapprox(sum(m.data.weights), 1.0)

    tr = plot_trace(history; backend = :none)
    @test size(tr.data.means) == (3, 2)

    dg = plot_diagnostics(history; backend = :none)
    @test length(dg.data.gens) == 3
    @test all(isfinite, dg.data.epsilons)

    pp = plot_ppc(history, 0.0; backend = :none)
    @test pp.data.observed == 0.0

    # a single state is accepted wherever a history is expected
    @test plot_marginals(history[end]; backend = :none) isa PlotSpec

    # backend = :plots without Plots loaded must fail with a clear message
    if !SequentialABC._plots_available()
        @test_throws ErrorException plot_marginals(history; backend = :plots)
    end

    # unknown backends are rejected
    @test_throws ArgumentError plot_marginals(history; backend = :bogus)
end

@testset "Plots extensions" begin
    @test Base.get_extension(SequentialABC, :PlotsExt) !== nothing
    @test Base.get_extension(SequentialABC, :StatsPlotsExt) !== nothing
    @test SequentialABC._plots_available()

    history = _synthetic_history()

    # plot functions work with backend = :plots (return Plots figures, not PlotSpec)
    m = plot_marginals(history; backend = :plots)
    @test !(m isa PlotSpec)

    tr = plot_trace(history; backend = :plots)
    @test !(tr isa PlotSpec)

    dg = plot_diagnostics(history; backend = :plots)
    @test !(dg isa PlotSpec)

    pp = plot_ppc(history, 0.0; backend = :plots)
    @test !(pp isa PlotSpec)

    # plot_pairs with backend = :plots
    p = plot_pairs(history; backend = :plots)
    @test !(p isa PlotSpec)
end
