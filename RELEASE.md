# Release procedure

1. Ensure `main` CI and docs pass.
2. Update `Project.toml` version and CHANGELOG.
3. Create annotated tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
4. Push tag: `git push origin vX.Y.Z`.
5. Publish release on GitHub and add release notes.
6. If registering package, follow General registry instructions.
