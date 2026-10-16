#!/usr/bin/env bash
#
# Reproduce the tests, the docs build, and the two example runs for
# SequentialABC.jl. Stops (and exits non-zero) on the first failure.
# All output is captured under repro/output/.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JULIA="${JULIA:-julia}"
OUT="$ROOT/repro/output"
CKPT="$OUT/checkpoints"
mkdir -p "$OUT" "$CKPT"

step() { printf '\n=== %s ===\n' "$*"; }

step "julia version"
"$JULIA" --version | tee "$OUT/julia_version.txt"

step "Pkg.instantiate / Pkg.build / Pkg.test"
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.test()' 2>&1 | tee "$OUT/test.log"

step "docs build (docs/make.jl)"
"$JULIA" --project=docs docs/make.jl 2>&1 | tee "$OUT/docs.log"

step "example: examples/continuous_logistic.jl"
"$JULIA" --project=. examples/continuous_logistic.jl 2>&1 | tee "$OUT/continuous_logistic.log"

step "example: examples/agents_schelling.jl (optional; requires Agents.jl)"
if "$JULIA" --project=examples/agents_env -e 'using Agents' >/dev/null 2>&1; then
  "$JULIA" --project=examples/agents_env examples/agents_schelling.jl 2>&1 | tee "$OUT/agents_schelling.log"
else
  echo "Agents.jl not available in examples/agents_env -- skipping agents example." | tee "$OUT/agents_schelling.log"
fi

step "deterministic checkpoints"
"$JULIA" --project=. -e '
using SequentialABC, Distributions, Random
dir = ARGS[1]
cfg = ABCSMCConfig(
    θ -> logpdf(Normal(0.0, 5.0), θ[1]),
    rng -> [rand(rng, Normal(0.0, 5.0))],
    identity,
    (s, s_obs) -> abs(s - s_obs),
    GaussianKernel(), [0.5;;],
    ContinuousSimulator(model_builder = μ -> μ[1],
                        evolve = (μ, rng) -> μ + randn(rng),
                        summary = y -> y),
)
st = make_state(cfg, Vector{Float64}, 50, 3.0, 1234)
run_smc!(st, 1.0; T = 3, schedule = QuantileSchedule(0.5), checkpoint_dir = dir)
println("wrote ", length(list_checkpoints(dir)), " checkpoint(s) to ", dir)
' "$CKPT" 2>&1 | tee "$OUT/checkpoints.log"

step "checksums"
( cd "$OUT" && find checkpoints -type f -name '*.jls' | sort | xargs -r sha256sum ) > "$OUT/checksums.txt"
cat "$OUT/checksums.txt"

step "done"
echo "All steps passed. Outputs: $OUT"
echo "Checksums:  $OUT/checksums.txt"
