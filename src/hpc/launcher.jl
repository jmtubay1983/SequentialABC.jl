# ── Slurm job templates and launcher helpers ─────────────────────────────────
#
# Everything in this file is **offline and pure**. Templates are rendered to
# strings, scripts are written to disk, and the submission preview returns the
# `sbatch` command you *would* run. No scheduler is ever contacted — no `sbatch`
# call, no cluster API, no network I/O — so the helpers are safe to exercise from
# tests, from a login node, or from a laptop while preparing a sweep.

"""
    SlurmTemplate(; job_name = "abc", time = "24:00:00", mem = "4G", cpus = 1,
                    partition = "", modules = String[],
                    env = Dict{String,String}()) -> SlurmTemplate

Description of a Slurm batch script: the `#SBATCH` headers, the modules to load,
and the environment variables to export.

| field | rendered as |
|---|---|
| `job_name` | `#SBATCH --job-name=<job_name>` |
| `time` | `#SBATCH --time=<time>` |
| `mem` | `#SBATCH --mem=<mem>` |
| `cpus` | `#SBATCH --cpus-per-task=<cpus>` |
| `partition` | `#SBATCH --partition=<partition>` (omitted when empty) |
| `modules` | one `module load <m>` line each, in the given order |
| `env` | one `export K=V` line each, keys sorted for byte‑determinism |

`job_name` is a *prefix*: [`make_generation_job`](@ref) appends `_gen_NNN`, so
`SlurmTemplate(job_name = "abc")` yields the block names `abc_gen_001`,
`abc_gen_002`, …

See [`render_slurm_script`](@ref) for the exact layout.
"""
struct SlurmTemplate
    job_name::String
    time::String
    mem::String
    cpus::Int
    partition::String
    modules::Vector{String}
    env::Dict{String,String}
end

function SlurmTemplate(; job_name::AbstractString = "abc",
                         time::AbstractString = "24:00:00",
                         mem::AbstractString = "4G",
                         cpus::Integer = 1,
                         partition::AbstractString = "",
                         modules = String[],
                         env = Dict{String,String}())
    isempty(job_name) && throw(ArgumentError("job_name must not be empty"))
    cpus >= 1 || throw(ArgumentError("cpus must be ≥ 1, got $cpus"))
    return SlurmTemplate(String(job_name), String(time), String(mem), Int(cpus),
                         String(partition), collect(String, modules),
                         Dict{String,String}(env))
end

# `extra_headers` entries are emitted verbatim when they already look like a
# `#SBATCH` directive, and prefixed with one otherwise.
function _sbatch_header(h::AbstractString)
    s = strip(h)
    return startswith(s, "#SBATCH") ? String(s) : "#SBATCH " * String(s)
end

"""
    render_slurm_script(t::SlurmTemplate, cmd::AbstractString;
                        extra_headers = String[], comments = String[]) -> String

Render a complete, byte‑deterministic Slurm batch script as a string.

```
#!/bin/bash
# <comment>                    (one per entry of `comments`, in order)
#SBATCH --job-name=…
#SBATCH --time=…
#SBATCH --mem=…
#SBATCH --cpus-per-task=…
#SBATCH --partition=…         (omitted when `t.partition == ""`)
#SBATCH <extra header>        (one per entry; `#SBATCH ` prepended if missing)
module load <m>               (one per entry of `t.modules`)
export K=V                    (one per entry of `t.env`, keys sorted)
<cmd>
```

Nothing is time‑ or host‑dependent and the environment block is sorted, so the
same `(t, cmd)` always yields the same bytes — the script can be diffed,
committed, or regenerated identically on any node. The command is **never run**;
rendering only builds a string.
"""
function render_slurm_script(t::SlurmTemplate, cmd::AbstractString;
                             extra_headers = String[],
                             comments = String[])
    io = IOBuffer()
    println(io, "#!/bin/bash")
    for c in comments
        println(io, "# ", c)
    end
    println(io, "#SBATCH --job-name=", t.job_name)
    println(io, "#SBATCH --time=", t.time)
    println(io, "#SBATCH --mem=", t.mem)
    println(io, "#SBATCH --cpus-per-task=", t.cpus)
    isempty(t.partition) || println(io, "#SBATCH --partition=", t.partition)
    for h in extra_headers
        println(io, _sbatch_header(h))
    end
    for m in t.modules
        println(io, "module load ", m)
    end
    for k in sort!(collect(keys(t.env)))
        println(io, "export ", k, "=", t.env[k])
    end
    println(io, cmd)
    return String(take!(io))
end

"""
    write_slurm_script(path, t::SlurmTemplate, cmd;
                       extra_headers = String[], comments = String[]) -> String

Render the script with [`render_slurm_script`](@ref) and write it to `path`,
creating the parent directory if necessary. Returns `path`.

The script is written only — nothing is executed and no scheduler is contacted.
"""
function write_slurm_script(path::AbstractString, t::SlurmTemplate, cmd::AbstractString;
                            extra_headers = String[], comments = String[])
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    write(path, render_slurm_script(t, cmd; extra_headers = extra_headers, comments = comments))
    return String(path)
end

_generation_job_name(prefix::AbstractString, gen::Integer) =
    string(prefix, "_gen_", lpad(gen, 3, '0'))

"""
    make_generation_job(cfg, gen::Int; template = SlurmTemplate(),
                        config_path = "cli_config.jl",
                        checkpoint_dir = "checkpoints",
                        script_dir = ".") -> (script_path, job_name)

Write a Slurm batch script that drives `bin/sequentialabc-run.jl` to generation
`gen`, and return `(script_path, job_name)`.

The command is the idempotent CLI runner,

```
julia --project=. bin/sequentialabc-run.jl \\
    --config <config_path> --checkpoint-dir <checkpoint_dir> --target-gen <gen>
```

so re‑submitting the job — or running it on a preempted node — is a no‑op once
generation `gen` already exists in `checkpoint_dir`. The run parameters
themselves (`N`, `ε0`, `T`, `base_seed`, the schedule) live in the config file
`config_path`, which the CLI `include`s at top level; only the target generation
is baked into the script.

Naming is deterministic: the job is `<template.job_name>_gen_NNN`, zero‑padded to
three digits (`abc_gen_003`), and the script is
`joinpath(script_dir, job_name * ".sbatch")`. `cfg` supplies the simulator
backend name for the script header and is otherwise unused.
"""
function make_generation_job(cfg, gen::Int;
                             template::SlurmTemplate = SlurmTemplate(),
                             config_path::AbstractString = "cli_config.jl",
                             checkpoint_dir::AbstractString = "checkpoints",
                             script_dir::AbstractString = ".")
    gen >= 1 || throw(ArgumentError("gen must be ≥ 1, got $gen"))
    job_name = _generation_job_name(template.job_name, gen)
    script_path = joinpath(String(script_dir), job_name * ".sbatch")

    # Same template, but with the generation‑specific job name.
    t = SlurmTemplate(job_name, template.time, template.mem, template.cpus,
                      template.partition, template.modules, template.env)

    cmd = "julia --project=. bin/sequentialabc-run.jl" *
          " --config " * String(config_path) *
          " --checkpoint-dir " * String(checkpoint_dir) *
          " --target-gen " * string(gen)

    sim = hasproperty(cfg, :simulator) ? nameof(typeof(cfg.simulator)) : typeof(cfg)
    comments = ["simulator: " * string(sim),
                "generation: " * string(gen)]

    write_slurm_script(script_path, t, cmd; comments = comments)
    return (script_path, job_name)
end

# checkpoint-collection helpers -------------------------------------------------

# Split a `*`‑only glob into its literal prefix and suffix. A single wildcard is
# all `save_checkpoint`'s naming needs, and being strict keeps the matcher tiny
# and fully deterministic.
function _checkpoint_glob(pattern::AbstractString)
    parts = split(pattern, '*')
    length(parts) == 2 ||
        throw(ArgumentError("pattern must contain exactly one '*' wildcard, got $(repr(pattern))"))
    return String(parts[1]), String(parts[2])
end

function _matches_glob(name::AbstractString, prefix::AbstractString, suffix::AbstractString)
    startswith(name, prefix) && endswith(name, suffix) &&
        length(name) >= length(prefix) + length(suffix)
end

"""
    collect_checkpoints!(out_dir, from = pwd(); pattern = "state_gen_*.jls",
                         mode = :copy) -> Vector{String}

Gather checkpoint files from one or more directories into `out_dir`.

`from` is a directory or a vector of directories — a Slurm array job typically
leaves one checkpoint directory per task, and this flattens them into a single
archive. Files whose names match `pattern` (exactly one `*` wildcard, by default
the checkpoint convention `state_gen_NNN.jls`) are copied (`mode = :copy`) or
moved (`mode = :move`) into `out_dir` **under their original names**, so
[`list_checkpoints`](@ref) and [`load_latest_state`](@ref) keep working on the
result.

Returns the sorted list of destination paths. A destination name that already
exists — because two source directories hold the same generation, or because the
file is already in `out_dir` — raises an `ArgumentError` instead of silently
overwriting: colliding generations are ambiguous, and only the caller knows which
copy is authoritative. Collect each run into its own `out_dir`.

No files are read or parsed, and nothing is transferred off the machine.
"""
function collect_checkpoints!(out_dir::AbstractString, from = pwd();
                              pattern::AbstractString = "state_gen_*.jls",
                              mode::Symbol = :copy)
    mode === :copy || mode === :move ||
        throw(ArgumentError("mode must be :copy or :move, got $(repr(mode))"))
    (prefix, suffix) = _checkpoint_glob(pattern)
    sources = from isa AbstractString ? (String(from),) : Tuple(String(d) for d in from)
    dest = String(out_dir)
    mkpath(dest)

    collected = String[]
    seen = Set{String}()
    for src in sources
        isdir(src) || continue
        for f in sort!(readdir(src))
            _matches_glob(f, prefix, suffix) || continue
            target = joinpath(dest, f)
            if f in seen || isfile(target)
                throw(ArgumentError("checkpoint $(f) would overwrite an existing file " *
                                    "in $(dest); collect each run into its own out_dir"))
            end
            push!(seen, f)
            mode === :copy ? cp(joinpath(src, f), target) : mv(joinpath(src, f), target)
            push!(collected, target)
        end
    end
    sort!(collected)
    return collected
end

"""
    simulate_slurm_submission_preview(t::SlurmTemplate, script_path;
                                      deps = nothing) -> String

Return the `sbatch` command line for `script_path` — a **string only**; nothing
is submitted and no scheduler is contacted.

`deps` optionally encodes job dependencies for chaining a sweep: a string
(`"12345"`) or a vector of job ids (`["12345", "12346"]`), rendered as
`--dependency=afterok:12345:12346`.

```julia
simulate_slurm_submission_preview(t, "abc_gen_003.sbatch"; deps = "12345")
# "sbatch --job-name=abc_gen_003 --dependency=afterok:12345 abc_gen_003.sbatch"
```
"""
function simulate_slurm_submission_preview(t::SlurmTemplate, script_path::AbstractString;
                                           deps = nothing)
    cmd = "sbatch --job-name=" * t.job_name
    if deps !== nothing
        ids = deps isa AbstractString ? [String(deps)] : collect(String, deps)
        isempty(ids) && throw(ArgumentError("deps must not be empty"))
        cmd *= " --dependency=afterok:" * join(ids, ":")
    end
    return cmd * " " * String(script_path)
end
