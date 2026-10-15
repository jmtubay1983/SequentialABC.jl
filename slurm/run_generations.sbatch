#!/bin/bash
#SBATCH --job-name=sequentialabc
#SBATCH --array=1-4               # one task per target generation; set to your T
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/sequentialabc_%A_%a.out
#SBATCH --error=logs/sequentialabc_%A_%a.err
#
# Each array task ensures generation SLURM_ARRAY_TASK_ID is complete: it loads
# the latest checkpoint and runs only the missing generations.
#
# Why a plain array is safe here (no locking, no dependency chains):
#   * idempotent   — a task that finds its generation already checkpointed
#                    exits 0 immediately
#   * deterministic— if two tasks race on generation g, both compute the
#                    IDENTICAL result (seeds are pure functions), and the
#                    checkpoint write is atomic (tmp → rename), so the file
#                    is always a complete, correct state for some generation
# To avoid duplicated compute on slow generations, chain tasks instead:
#   sbatch --array=1 --dependency=afterok:<jobid> ... per generation
#
# Site setup — edit for your cluster (WUR rockhopper / NCI / ...):
#   module load julia
#
# Required environment:
#   PROJECT_DIR              SequentialABC package root
# Optional:
#   JULIA_BIN                julia binary           (default: julia)
#   CONFIG                   config file            (default: examples/cli_config.jl)
#   CHECKPOINT_DIR           checkpoint directory   (default: checkpoints/run1)

set -euo pipefail
mkdir -p logs

JULIA_BIN=${JULIA_BIN:-julia}
PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR to the SequentialABC package root}

"$JULIA_BIN" --project="$PROJECT_DIR" "$PROJECT_DIR/bin/sequentialabc-run.jl" \
    --config "${CONFIG:-$PROJECT_DIR/examples/cli_config.jl}" \
    --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints/run1}" \
    --target-gen "$SLURM_ARRAY_TASK_ID"
