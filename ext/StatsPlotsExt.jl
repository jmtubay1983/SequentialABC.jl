module StatsPlotsExt

using SequentialABC
using StatsPlots

"""
    _plot_pairs_corner(A::AbstractMatrix) -> Plots figure

Render a pairwise corner plot using StatsPlots.cornerplot. Called by
`PlotsExt._plot` when the StatsPlotsExt extension is loaded and `d > 1`.
"""
function _plot_pairs_corner(A::AbstractMatrix)
    return StatsPlots.cornerplot(A; framestyle = :box)
end

end