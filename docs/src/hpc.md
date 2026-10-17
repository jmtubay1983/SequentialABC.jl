# HPC / Slurm

SequentialABC.jl is designed for high-performance computing workloads. This guide covers cluster execution, array jobs, checkpoint/restart, and preemption safety.

## 1. CLI runner

**File:** `bin/sequentialabc-run.jl`

A command-line entry point for batch submission:

```bash
julia bin/sequentialabc-run.jl --config cli_config.jl --checkpoint-dir checkpoints/
```

The config file (`cli_config.jl`) defines the model, prior, simulator, and ABC‑SMC parameters. See the example in `bin/sequentialabc-run.jl` for the expected structure.

## 2. Checkpoint/restart on Slurm

Enable checkpointing in your run:

```julia
run_smc!(state, s_obs; T=100, schedule=schedule, checkpoint_dir="checkpoints/")
```

Checkpoints are written after initialization and after every generation as `checkpoints/state_gen_001.jls`, `state_gen_002.jls`, etc.

### Resubmitting after preemption

If your Slurm job is preempted or hits its time limit, resubmit with the same script:

```julia
state = load_latest_state("checkpoints/", cfg)
run_smc!(state, s_obs; T=100, schedule=schedule, checkpoint_dir="checkpoints/")
```

`run_smc!` continues from `state.gen + 1` using the stored deterministic seeds — the resumed trajectory is **bit-identical** to an uninterrupted run.

## 3. Slurm array jobs

For parameter sweeps or multiple independent runs, use a Slurm array job. Each array task gets its own `base_seed`:

```bash
#SBATCH --array=1-20
```

```julia
base_seed = 1000 + parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
state = make_state(cfg, Vector{Float64}, N, ε0, base_seed)
run_smc!(state, s_obs; T=50, schedule=schedule, checkpoint_dir="checkpoints/run_$(ENV["SLURM_ARRAY_TASK_ID"])")
```

Because every RNG is derived from `base_seed`, different array tasks produce independent, reproducible streams.

## 4. File hygiene

| File | Purpose |
|------|---------|
| `state_gen_001.jls` | Checkpoint after generation 1 |
| `state_gen_002.jls` | Checkpoint after generation 2 |
| ... | ... |
| `state_gen_XXX.jls.tmp` | Stale temp file from an interrupted write — safe to delete |

Stale `.tmp` files are ignored by `list_checkpoints` and `load_latest_state`. You can safely delete them at any time.

## 5. Preemption safety

Checkpoints use **atomic write**:

1. Serialize data to `state_gen_XXX.jls.tmp`
2. `mv(tmp, path)` — POSIX rename is atomic on the same filesystem

A job killed mid-write can never leave a corrupted checkpoint — at worst a stale `.tmp` file. The next `save_checkpoint` will overwrite it.

## 6. Recommended Slurm script

```bash
#!/bin/bash
#SBATCH --job-name=abc-smc
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/abc_%A_%a.out

julia --project=. -e '
using Pkg
Pkg.instantiate()

using SequentialABC
using Distributions
using Random

# ... build cfg, state ...

run_smc!(state, s_obs; T=100, schedule=schedule,
         checkpoint_dir="checkpoints/run_$(get(ENV, "SLURM_ARRAY_TASK_ID", "0"))")
'
```

## 7. Generating job scripts from Julia

Hand‑writing a script per generation does not scale. `SlurmTemplate` plus
`render_slurm_script` build the script text, and `make_generation_job` writes one
script per generation — **without ever submitting it**:

```julia
using SequentialABC

t = SlurmTemplate(; job_name = "abc", time = "24:00:00", mem = "4G",
                  cpus = 1, partition = "short",
                  modules = ["julia/1.13"],
                  env = Dict("JULIA_DEPOT_PATH" => "/scratch/\$USER/.julia"))

(path, name) = make_generation_job(cfg, 3;
                                   template = t,
                                   config_path = "examples/cli_config.jl",
                                   checkpoint_dir = "ckpt",
                                   script_dir = "jobs")
# ("jobs/abc_gen_003.sbatch", "abc_gen_003")
```

The command baked into the script is the idempotent CLI runner,

```bash
julia --project=. bin/sequentialabc-run.jl \
    --config examples/cli_config.jl --checkpoint-dir ckpt --target-gen 3
```

Rendering is **byte‑deterministic** — no timestamps, and the environment block is
sorted by key — so a script can be committed, diffed, or regenerated identically
on any node. `render_slurm_script(t, cmd; extra_headers = [...], comments = [...])`
is the string‑only form for inspecting or post‑processing the script yourself;
`write_slurm_script(path, t, cmd)` writes it to disk.

### sbatch preview

`simulate_slurm_submission_preview` returns the `sbatch` command **as a string** —
no scheduler is contacted:

```julia
simulate_slurm_submission_preview(t, path; deps = "12345")
# "sbatch --job-name=abc_gen_003 --dependency=afterok:12345 jobs/abc_gen_003.sbatch"
```

`deps` accepts one job id or a vector; the preview is meant for logging, dry runs
and documentation, not for execution inside Julia.

## 8. Generate → submit → collect → resume

A per‑generation sweep on a cluster:

1. **Generate** one script per generation locally:
   `make_generation_job(cfg, g; script_dir = "jobs", checkpoint_dir = "ckpt")`.
2. **Submit** them as an array (each task with its own checkpoint directory):
   `sbatch --array=1-10 jobs/abc_gen_*.sbatch`.
3. **Collect** the scattered checkpoints into one archive with
   `collect_checkpoints!` (below).
4. **Resume** from the flattened archive on a login node:
   `load_latest_state("archive/", cfg)` then `run_smc!(state, s_obs; T = …)`.

Because every checkpoint is named by generation and the runner is idempotent,
step 4 works whether the sweep finished on one node or was preempted and
requeued several times.

## 9. Collecting checkpoints

A sweep usually leaves one checkpoint directory per task. `collect_checkpoints!`
flattens them into a single directory, keeping the original `state_gen_NNN.jls`
names so `list_checkpoints` and `load_latest_state` keep working:

```julia
collect_checkpoints!("archive", ["ckpt/run_1", "ckpt/run_2", "ckpt/run_3"])
# ["archive/state_gen_001.jls", "archive/state_gen_002.jls", …]
```

The default `pattern` is `"state_gen_*.jls"` (a single `*` wildcard); `mode =
:move` moves instead of copies. Colliding generations — two runs that produced
the same generation file — raise an `ArgumentError` rather than silently
overwriting, so collect each run into its own `out_dir`.

## 10. Memory and performance

- Population size `N` is the main memory driver — each particle stores its θ, weight, summary, and distance.
- `M` replicates per particle multiply simulation cost but reduce summary noise.
- Checkpoint I/O is minimal — only data is serialized, never `cfg` closures.
- For very long runs, checkpoint every generation (the default) to minimize lost work on preemption.

## 11. API

```@docs
SlurmTemplate
render_slurm_script
write_slurm_script
make_generation_job
collect_checkpoints!
simulate_slurm_submission_preview
```