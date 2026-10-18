using Serialization

const CHECKPOINT_RE = r"^state_gen_(\d+)\.jls$"
const CHECKPOINT_SCHEMA_VERSION = 1

_checkpoint_name(gen::Int) = "state_gen_" * lpad(string(gen), 3, '0') * ".jls"

"""
    save_checkpoint(state::ABCSMCState, dir::AbstractString) -> String

Atomically checkpoint the sampler to `dir/state_gen_XXX.jls`.

Only **data** is serialized (θ, w, summaries, ε, gen, ess, seeds, counters,
plus adaptive state when present) — never `cfg` and its closures. This makes
checkpoints portable across Julia versions, code revisions, and Slurm nodes:
reload with the same (or updated) `cfg` via `load_latest_state`.

A `schema_version` tag is written with every checkpoint so older code can
recognize checkpoints it cannot read. New code reads both legacy (no tag) and
new (tagged) checkpoints.

Atomicity: the payload is written to a `.tmp` file first and then renamed.
POSIX rename is atomic on the same filesystem, so a job killed mid‑write can
never leave a corrupted checkpoint — at worst a stale `.tmp` file, which the
loader ignores.
"""
function save_checkpoint(state::ABCSMCState, dir::AbstractString)
    mkpath(dir)
    path = joinpath(dir, _checkpoint_name(state.gen))
    tmp = path * ".tmp"

    data = (schema_version = CHECKPOINT_SCHEMA_VERSION,
            θ = state.θ,
            w = state.w,
            summaries = state.summaries,
            dists = state.dists,
            ε = state.ε,
            gen = state.gen,
            ess = state.ess,
            base_seed = state.base_seed,
            n_attempts = state.n_attempts,
            n_resamples = state.n_resamples)

    # Adaptive state is checkpointed when present so a resumed run can
    # continue with identical weights and epsilon-distances.
    if state.adaptive !== nothing
        ad = state.adaptive
        data = merge(data, (adaptive = (
            strategy = ad.strategy,
            key_order = ad.key_order,
            policy = ad.policy,
            weights = ad.weights,
            gen_estimated = ad.gen_estimated,
            epsilon_distances = ad.epsilon_distances,
            epsilon = ad.epsilon,
        ),))
    end

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

Schema handling:
- New checkpoints (`schema_version` present) restore adaptive state when
  included in the payload.
- Legacy checkpoints (no `schema_version` tag, pre‑adaptive) load with
  `adaptive = nothing` — identical to pre‑adaptive behavior. Old code cannot
  read new adaptive checkpoints (it will see an unknown `schema_version` key
  and the extra `adaptive` field).
"""
function load_checkpoint(path::AbstractString, cfg)
    data = open(deserialize, path)
    Tθ   = eltype(data.θ)          # struct parameters hold ELEMENT types (fields are Vector{Tθ})
    Tsum = eltype(data.summaries)

    # Detect schema version; legacy checkpoints (pre‑adaptive) have no tag.
    schema_version = get(data, :schema_version, 0)

    # Build the base state (fields common to all schema versions).
    state = ABCSMCState{typeof(cfg), Tθ, Tsum}(
        cfg, data.θ, data.w, data.summaries, data.dists, data.ε, data.gen,
        data.ess, data.base_seed, data.n_attempts, data.n_resamples, nothing
    )

    # Restore adaptive state if present (schema_version >= 1).
    if schema_version >= 1 && haskey(data, :adaptive)
        ad = data.adaptive
        state.adaptive = AdaptiveDistanceState(
            ad.strategy,
            ad.key_order,
            ad.policy,
            ad.weights,
            ad.gen_estimated,
            ad.epsilon_distances,
            ad.epsilon,
        )
    end

    return state
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