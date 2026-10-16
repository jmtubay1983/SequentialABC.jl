# Reproducibility bundle

This directory reproduces the tests, the docs build, and the two example runs
for **SequentialABC.jl**.

## Verified environment

- Verified base commit: `main` @ `3f352e7`
- Julia versions tested: **1.11.9** and **1.13.2**
- `Pkg.test()`: 146/146 passing; docs build (`julia --project=docs docs/make.jl`): exit 0

## One-shot reproduction

```bash
bash repro/run_all.sh
```

The script runs with `set -euo pipefail` (it stops and exits non-zero on the
first failure) and performs, in order:

1. `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.test()'`
2. `julia --project=docs docs/make.jl`
3. `julia --project=. examples/continuous_logistic.jl`
4. `julia --project=examples/agents_env examples/agents_schelling.jl`
   (skipped gracefully if `Agents.jl` is not installed)
5. Deterministic checkpoint generation into `repro/output/checkpoints/`
6. SHA-256 of the produced checkpoints → `repro/output/checksums.txt`

All console output is captured under `repro/output/`. Point at a specific Julia
binary with `JULIA=/path/to/julia bash repro/run_all.sh`.

## Manual commands

```bash
# Tests
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.test()'

# Docs
julia --project=docs docs/make.jl

# Continuous-backend example
julia --project=. examples/continuous_logistic.jl

# Agents.jl-backend example (separate env so Agents stays out of the core package)
julia --project=examples/agents_env -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/agents_env examples/agents_schelling.jl
```

## Checkpoint checksums

The example scripts write checkpoints into internal `mktempdir()` directories
that are cleaned up on exit, so `run_all.sh` additionally emits deterministic
checkpoints into `repro/output/checkpoints/` and records their SHA-256 in
`repro/output/checksums.txt`.

> The sampler derives per-generation seeds via Julia's `hash`, which is not
> guaranteed stable across Julia versions. Checkpoint bytes — and therefore the
> checksums — can differ between Julia releases, so `checksums.txt` records a
> specific run rather than acting as a cross-version invariant.

## Manifest policy

No `repro/Manifest.toml` is committed: a committed manifest would be tied to one
machine's exact dependency resolution and could not be guaranteed to match the
verified environment. To pin an exact dependency set locally, generate one from
a clean resolve:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.status(; manifest=true)'
```

## Ignored artifacts (do not commit)

`docs/Manifest.toml` and `docs/build/` are gitignored and must **not** be
committed to `main`. Confirm with:

```bash
git check-ignore -v docs/Manifest.toml docs/build/
```

## Docker (optional)

A minimal image is provided:

```bash
docker build -f repro/Dockerfile -t sequentialabc-repro .
docker run --rm sequentialabc-repro
```

It installs the tested Julia (1.13.2), copies the repository, and runs
`repro/run_all.sh`.
