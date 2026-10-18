module PlotsExt

using SequentialABC: _PLOTS_MOD
using Plots

function __init__()
    _PLOTS_MOD[] = Plots
end

end
