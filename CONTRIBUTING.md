# Contributing

Thank you for contributing.

- Do not commit `docs/Manifest.toml` or `docs/build/`. These are environment artifacts and are ignored by .gitignore.
- Docs environment: use `julia --project=docs docs/make.jl` to build locally.
- Tests: run `julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.
- Branch naming: use `chore/*`, `feat/*`, or `fix/*`.
- Open a PR and include a short description of changes and verification steps.

## Running tests and examples

Always activate the repository project when running tests or examples from the repo root:

```bash
# from repo root
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# build docs
julia --project=docs docs/make.jl

# run examples
julia --project=. examples/continuous_logistic.jl
julia --project=examples/agents_env examples/agents_schelling.jl
```

Note: Running `julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` without `--project=.` may use your global environment and produce "Package not found" errors.
