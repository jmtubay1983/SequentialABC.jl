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

## 7. Memory and performance

- Population size `N` is the main memory driver — each particle stores its θ, weight, summary, and distance.
- `M` replicates per particle multiply simulation cost but reduce summary noise.
- Checkpoint I/O is minimal — only data is serialized, never `cfg` closures.
- For very long runs, checkpoint every generation (the default) to minimize lost work on preemption.