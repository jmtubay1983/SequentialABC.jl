using Serialization

const CHECKPOINT_RE = r"^state_gen_(\d+)\.jls$"

_checkpoint_name(gen::Int) = "state_gen_" * lpad(string(gen), 3, '0') * ".jls"

"""
    save_checkpoint(state::ABCSMCState, dir::AbstractString) -> String

Atomically checkpoint the sampler to `dir/state_gen_XXX.jls`.

Only **data** is serialized (θ, w, summaries, ε, gen, ess, seeds, counters) —
never `cfg` and its closures. This makes checkpoints portable across Julia
versions, code revisions, and Slurm nodes: reload with the same (or updated)
`cfg` via `load_latest_state`.

Atomicity: the payload is written to a `.tmp` file first and then renamed.
POSIX rename is atomic on the same filesystem, so a job killed mid‑write can
never leave a corrupted checkpoint — at worst a stale `.tmp` file, which the
loader ignores.
"""
function save_checkpoint(state::ABCSMCState, dir::AbstractString)
    mkpath(dir)
    path = joinpath(dir, _checkpoint_name(state.gen))
    tmp = path * ".tmp"

    data = (θ = state.θ,
            w = state.w,
            summaries = state.summaries,
            dists = state.dists,
            ε = state.ε,
            gen = state.gen,
            ess = state.ess,
            base_seed = state.base_seed,
            n_attempts = state.n_attempts,
            n_resamples = state.n_resamples)

    open(tmp, "w") do io
        serialize(io, data)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    list_checkpoints(dir::AbstractString) -> Vector{Tuple{Int,String}}

Sorted (gen, path) pairs of all valid checkpoints in `dir`.
Files not matching `state_gen_NNN.jls` (including stale `.tmp` files) are ignored.
"""
function list_checkpoints(dir::AbstractString)
    isdir(dir) || return Tuple{Int,String}[]
    out = Tuple{Int,String}[]
    for f in readdir(dir)
        m = match(CHECKPOINT_RE, f)
        m === nothing && continue
        push!(out, (parse(Int, m.captures[1]), joinpath(dir, f)))
    end
    sort!(out; by = first)
    return out
end

"""
    load_checkpoint(path::AbstractString, cfg) -> ABCSMCState

Load a single checkpoint file and reattach `cfg` (an `ABCSMCConfig`; it is
intentionally not serialized — checkpoints carry data only).
"""
function load_checkpoint(path::AbstractString, cfg)
    data = open(deserialize, path)
    Tθ   = eltype(data.θ)          # struct parameters hold ELEMENT types (fields are Vector{Tθ})
    Tsum = eltype(data.summaries)
    return ABCSMCState{typeof(cfg), Tθ, Tsum}(
        cfg, data.θ, data.w, data.summaries, data.dists, data.ε, data.gen,
        data.ess, data.base_seed, data.n_attempts, data.n_resamples
    )
end

"""
    load_latest_state(dir::AbstractString, cfg) -> ABCSMCState

Load the highest‑generation checkpoint from `dir` and reattach `cfg` (an
`ABCSMCConfig`; it is intentionally not serialized — checkpoints carry data only).
Resume a run with `run_smc!(state, summary_obs; T=…, schedule=…)`, which
continues from `state.gen + 1` using the stored deterministic seeds — the
resumed trajectory is bit‑identical to an uninterrupted one.
"""
function load_latest_state(dir::AbstractString, cfg)
    cps = list_checkpoints(dir)
    isempty(cps) && error("No checkpoints found in $dir")
    return load_checkpoint(cps[end][2], cfg)
end