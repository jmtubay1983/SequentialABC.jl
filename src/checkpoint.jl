using Serialization

const CHECKPOINT_RE = r"^state_gen_(\d+)\.jls$"

# Checkpoint schema versions (full matrix in `load_checkpoint`):
#   0 — unversioned legacy payload (pre-adaptive releases; no tag)
#   1 — versioned non-adaptive payload
#   2 — versioned adaptive payload (carries the :adaptive block)
const CHECKPOINT_SCHEMA_LEGACY = 0
const CHECKPOINT_SCHEMA_NONADAPTIVE = 1
const CHECKPOINT_SCHEMA_ADAPTIVE = 2
# Highest schema version this code can read or write.
const CHECKPOINT_SCHEMA_VERSION = CHECKPOINT_SCHEMA_ADAPTIVE

_checkpoint_name(gen::Int) = "state_gen_" * lpad(string(gen), 3, '0') * ".jls"

"""
    save_checkpoint(state::ABCSMCState, dir::AbstractString) -> String

Atomically checkpoint the sampler to `dir/state_gen_XXX.jls`.

Only **data** is serialized (θ, w, summaries, ε, gen, ess, seeds, counters,
plus adaptive state when present) — never `cfg` and its closures. This makes
checkpoints portable across Julia versions, code revisions, and Slurm nodes:
reload with the same (or updated) `cfg` via `load_latest_state`.

A `schema_version` tag is written with every checkpoint: `1` for non-adaptive
payloads, `2` for adaptive payloads (see [`load_checkpoint`](@ref) for the
compatibility matrix). New code reads legacy (untagged, schema 0), schema-1
and schema-2 checkpoints; old code cannot read schema-2 adaptive checkpoints
(unsupported — it ignores the tag and would silently resume without the
adaptive state).

Atomicity: the payload is written to a `.tmp` file first and then renamed.
POSIX rename is atomic on the same filesystem, so a job killed mid‑write can
never leave a corrupted checkpoint — at worst a stale `.tmp` file, which the
loader ignores.
"""
function save_checkpoint(state::ABCSMCState, dir::AbstractString)
    mkpath(dir)
    path = joinpath(dir, _checkpoint_name(state.gen))
    tmp = path * ".tmp"

    data = (schema_version = state.adaptive === nothing ?
                             CHECKPOINT_SCHEMA_NONADAPTIVE :
                             CHECKPOINT_SCHEMA_ADAPTIVE,
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

    # Adaptive runtime state is checkpointed when present so a resumed run can
    # continue with identical weights, key order, epsilon-selection distances
    # and epsilon. The immutable configuration is NOT serialized — the caller
    # re-supplies it when resuming.
    if state.adaptive !== nothing
        ad = state.adaptive
        data = merge(data, (adaptive = (
            key_order = ad.key_order,
            weights = ad.weights,
            gen_applied = ad.gen_applied,
            epsilon_selection_distances = ad.epsilon_selection_distances,
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

Schema handling — the declared version and the payload shape must agree:

| schema | writer | payload | this loader |
|---|---|---|---|
| 0 (untagged) | legacy releases | base fields only | loads, `adaptive = nothing` |
| 1 | this code, non-adaptive | base fields + tag | loads, `adaptive = nothing` |
| 2 | this code, adaptive | base fields + tag + `:adaptive` | restores adaptive state |
| > 2 | future code | unknown | **rejected** (`ArgumentError`) |

A schema-2 payload without an `:adaptive` block, or a schema ≤ 1 payload with
one, is rejected as malformed. **Old (schema-0) code cannot read schema-2
adaptive checkpoints** — it would ignore the tag and the block and silently
resume without adaptive state; that is unsupported and not claimed here.
"""
function load_checkpoint(path::AbstractString, cfg)
    data = open(deserialize, path)
    if !(data isa NamedTuple)
        throw(ArgumentError(
            "checkpoint $path is malformed: payload must be a NamedTuple (got $(typeof(data)))"))
    end
    Tθ   = eltype(data.θ)          # struct parameters hold ELEMENT types (fields are Vector{Tθ})
    Tsum = eltype(data.summaries)
    # Detect schema version; legacy (pre‑adaptive) checkpoints carry no tag.
    schema_version = get(data, :schema_version, CHECKPOINT_SCHEMA_LEGACY)
    schema_version isa Int || throw(ArgumentError(
        "checkpoint schema_version must be an Int in $path; got " *
        "$(typeof(schema_version))"))
    if !(CHECKPOINT_SCHEMA_LEGACY <= schema_version <= CHECKPOINT_SCHEMA_VERSION)
        throw(ArgumentError(
            "unsupported checkpoint schema version $schema_version in $path; " *
            "this code supports 0 (legacy), 1 (non-adaptive) and 2 (adaptive)"))
    end
    has_adaptive = haskey(data, :adaptive)
    if schema_version == CHECKPOINT_SCHEMA_ADAPTIVE && !has_adaptive
        throw(ArgumentError(
            "checkpoint $path declares schema version 2 (adaptive) but its " *
            "payload has no :adaptive block"))
    end
    if schema_version <= CHECKPOINT_SCHEMA_NONADAPTIVE && has_adaptive
        throw(ArgumentError(
            "checkpoint $path declares non-adaptive schema version " *
            "$schema_version but its payload carries an :adaptive block"))
    end

    # Build the base state (fields common to all schema versions).
    state = ABCSMCState{typeof(cfg), Tθ, Tsum}(
        cfg, data.θ, data.w, data.summaries, data.dists, data.ε, data.gen,
        data.ess, data.base_seed, data.n_attempts, data.n_resamples, nothing
    )

    # Restore the adaptive runtime state (schema 2). The immutable config is
    # re-supplied by the caller via run_smc!(...; adaptive = ...).
    if has_adaptive
        state.adaptive = _restore_adaptive_checkpoint(data.adaptive, path, data.gen, length(data.dists))
    end

    return state
end

function _restore_adaptive_checkpoint(ad, path::AbstractString, state_gen::Int, n_particles::Int)
    ad isa NamedTuple || throw(ArgumentError(
        "checkpoint $path has a malformed :adaptive block of type " *
        "$(typeof(ad)); expected a NamedTuple"))
    required = (:key_order, :weights, :gen_applied,
                :epsilon_selection_distances, :epsilon)
    missing = filter(name -> !haskey(ad, name), required)
    isempty(missing) || throw(ArgumentError(
        "checkpoint $path adaptive block is missing fields $(Tuple(missing))"))

    key_order = ad.key_order
    key_order isa AbstractVector && all(x -> x isa Symbol, key_order) ||
        throw(ArgumentError(
            "checkpoint $path adaptive key_order must be a vector of Symbol"))
    weights = ad.weights
    weights isa AbstractVector && all(x -> x isa Real, weights) ||
        throw(ArgumentError(
            "checkpoint $path adaptive weights must be a vector of Real"))
    ad.gen_applied isa Int || throw(ArgumentError(
        "checkpoint $path adaptive gen_applied must be an Int"))
    if !(1 <= ad.gen_applied <= state_gen)
        throw(ArgumentError(
            "checkpoint $path adaptive gen_applied $(ad.gen_applied) must be in " *
            "[1, state.gen = $state_gen]"))
    end
    distances = ad.epsilon_selection_distances
    distances isa AbstractVector && all(x -> x isa Real, distances) ||
        throw(ArgumentError(
            "checkpoint $path adaptive epsilon_selection_distances must be " *
            "a vector of Real"))
    length(distances) == n_particles || throw(ArgumentError(
        "checkpoint $path adaptive epsilon_selection_distances length " *
        "$(length(distances)) does not match particle population length " *
        "$n_particles"))
    ad.epsilon isa Real && isfinite(ad.epsilon) || throw(ArgumentError(
        "checkpoint $path adaptive epsilon must be finite Real; got $(ad.epsilon)"))
    ad.epsilon >= 0 || throw(ArgumentError(
        "checkpoint $path adaptive epsilon must be >= 0, got $(ad.epsilon)"))

    return AdaptiveDistanceState(
        collect(Symbol, key_order), weights, ad.gen_applied, distances,
        ad.epsilon)
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
