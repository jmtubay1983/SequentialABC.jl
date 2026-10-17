#!/bin/bash
# Example Slurm batch script for one ABC-SMC generation.
#
# This file is *documentation*, not a driver: nothing in the package runs it.
# `make_generation_job` / `render_slurm_script` only ever emit script *text* —
# submitting is your job (`sbatch examples/slurm_template.sh`).
#
#SBATCH --job-name=abc_gen_003
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/abc_gen_003_%A.out

# Module names depend on your cluster's module tree.
module load julia/1.13

# The runner is idempotent: if generation 3 already exists in ckpt/ the job
# exits immediately, so a requeued or preempted job costs nothing. The seeds are
# pure functions of the config's base_seed, so re-running reproduces byte-identical
# checkpoints.
julia --project=. bin/sequentialabc-run.jl \
    --config examples/cli_config.jl \
    --checkpoint-dir ckpt \
    --target-gen 3
