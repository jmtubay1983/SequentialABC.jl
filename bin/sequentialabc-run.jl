#!/usr/bin/env julia
# SequentialABC.jl HPC driver: ensure ABC-SMC generations up to a target
# generation are complete, checkpointing each one.
#
# Usage:
#   julia --project=<pkg> bin/sequentialabc-run.jl \
#       --config my_config.jl --checkpoint-dir ckpt/ [--target-gen K]
#
# The config file must evaluate (last expression) to a NamedTuple with keys:
#   cfg, summary_obs, N, ε0, base_seed, T, schedule
# optional keys: Tθ (default Vector{Float64}), M (default 1),
#                max_attempts (default 10_000)
#
# Slurm-safe by construction:
#   * idempotent   — a job whose target generation already exists exits 0
#   * deterministic— requeued/preempted/duplicated tasks reproduce the exact
#                    same checkpoints (seeds are pure functions of
#                    gen/particle/attempt/replicate/base_seed)
#   * atomic       — checkpoints are written tmp→rename; a killed job leaves
#                    at most a stale .tmp, which the loader ignores
#
# NOTE: the config file is `include`d at script top level (below), never
# inside a function. Julia's world-age rule forbids precompiled SequentialABC
# code from calling closures defined in a too-new world age; top-level include
# puts the config's closures in the same world age as this script.

using SequentialABC
using Printf
using Dates

function parse_cli(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected argument: $a (flags are --key value)")
        i == length(args) && error("missing value for $a")
        opts[a[3:end]] = args[i + 1]
        i += 2
    end
    return opts
end

function run_to_target(spec, opts)
    Tθ           = get(spec, :Tθ, Vector{Float64})
    M            = get(spec, :M, 1)
    max_attempts = get(spec, :max_attempts, 10_000)
    dir          = opts["checkpoint-dir"]
    target       = haskey(opts, "target-gen") ? parse(Int, opts["target-gen"]) : spec.T
    (target >= 1) || error("--target-gen must be >= 1")
    target > spec.T && @warn "target-gen $target > config T=$(spec.T); clamping" _module=nothing
    target = min(target, spec.T)

    # Progress = highest generation-indexed checkpoint (stale .tmp ignored)
    cps = list_checkpoints(dir)
    gen = isempty(cps) ? 0 : cps[end][1]

    if gen >= target
        @printf("[sequentialabc] up to date: gen %d >= target %d — nothing to do\n", gen, target)
        return 0
    end

    state = gen == 0 ? make_state(spec.cfg, Tθ, spec.N, spec.ε0, spec.base_seed) :
                       load_latest_state(dir, spec.cfg)
    @printf("[sequentialabc] %s: resuming from gen %d → target gen %d\n", Dates.now(), gen, target)
    flush(stdout)

    if state.gen == 0
        t0 = time()
        initialize!(state, spec.summary_obs; max_attempts = max_attempts)
        save_checkpoint(state, dir)
        @printf("[sequentialabc] gen %2d  ε=%.4g  ESS=%.1f  attempts=%d  (%.1fs)\n",
                state.gen, state.ε, state.ess, state.n_attempts, time() - t0)
        flush(stdout)
    end

    for t in (state.gen + 1):target
        t0  = time()
        a0  = state.n_attempts
        ε_new = spec.schedule(state, t)
        run_generation!(state, spec.summary_obs; ε = ε_new, M = M, max_attempts = max_attempts)
        save_checkpoint(state, dir)
        @printf("[sequentialabc] gen %2d  ε=%.4g  ESS=%.1f  attempts=%d  (%.1fs)\n",
                state.gen, state.ε, state.ess, state.n_attempts - a0, time() - t0)
        flush(stdout)
    end

    @printf("[sequentialabc] done: gen %d, ε=%.4g, ESS=%.1f, resamplings=%d\n",
            state.gen, state.ε, state.ess, state.n_resamples)
    return 0
end

# ── Top-level entry point (config include happens here, outside any function) ──
opts = parse_cli(ARGS)

haskey(opts, "config")         || error("--config is required")
haskey(opts, "checkpoint-dir") || error("--checkpoint-dir is required")

spec = include(abspath(opts["config"]))
for k in (:cfg, :summary_obs, :N, :ε0, :base_seed, :T, :schedule)
    haskey(spec, k) || error("config must provide key :$k")
end

try
    exit(run_to_target(spec, opts))
catch err
    showerror(stderr, err)
    println(stderr)
    exit(1)
end
