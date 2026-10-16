# Contributing

Thank you for contributing.

- Do not commit `docs/Manifest.toml` or `docs/build/`. These are environment artifacts and are ignored by .gitignore.
- Docs environment: use `julia --project=docs docs/make.jl` to build locally.
- Tests: run `julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.
- Branch naming: use `chore/*`, `feat/*`, or `fix/*`.
- Open a PR and include a short description of changes and verification steps.
