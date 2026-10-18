module StatsPlotsExt

using SequentialABC: _STATSPLOTS_MOD
using StatsPlots

function __init__()
    _STATSPLOTS_MOD[] = StatsPlots
end

end
