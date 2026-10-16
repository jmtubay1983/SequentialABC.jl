# Design Notes

Key design decisions behind SequentialABC.jl, and why they matter for scientific inference.

## 1. Deterministic seed architecture

Every stochastic step in the sampler is driven by a `MersenneTwister` seeded from:

```julia
make_particle_seed(gen, i, attempt, m, base_seed)
```

| Argument | Role |
|----------|------|
| `gen` | Generation number |
| `i` | Particle index within the generation |
| `attempt` | Redraw-until-accepted attempt counter |
| `m` | Role tag (0 = ancestor, -1 = perturbation, -2 = population resample, ≥1 = simulation replicate) |
| `base_seed` | User-provided base seed |

### Why this matters

- **Bit-identical restarts**: Resume a checkpointed run and you get the exact same trajectory as an uninterrupted run — same particles, same weights, same ε.
- **No silent divergence**: Two runs with the same seed cannot diverge due to differing acceptance counts.
- **HPC safe**: Safe for Slurm array jobs — each particle's RNG is independent and reproducible.
- **Replay for debugging**: Reproduce any particle's simulation history from its `(gen, i, attempt)` coordinates.

### The `attempt` counter

Without the `attempt` counter, redraw-until-accepted would replay the identical rejected proposal forever with a deterministic RNG. Incrementing `attempt` on each rejection ensures a fresh draw every time.

## 2. Transform vs prior space

Particles \(\theta_{\text{raw}}\) live in the space where the prior is defined. The transform \(\theta_{\text{raw}} \to \theta_{\text{model}}\) is applied **only at simulation time**.

This means:

- The prior is a proper log-density on the stored parameter space.
- Importance weights use the prior density on \(\theta_{\text{raw}}\) (no Jacobian correction needed).
- The transform can be any project-specific map (e.g., logit, exponentiation, ODE parameterization).

## 3. Summary identifiability

ABC‑SMC identifies parameters only through the summary statistic \(s(\theta)\). If two different \(\theta\) values produce the same summary distribution, they are observationally equivalent — no amount of data will distinguish them.

**Practical advice:**
- Choose summaries that are sensitive to the parameters you want to infer.
- Use multiple summaries when parameters affect different aspects of the dynamics.
- Check posterior correlation plots for ridge-like structures (signs of non-identifiability).

## 4. ESS collapse behavior

The effective sample size is tracked every generation:

\[
\text{ESS} = \frac{1}{\sum_i w_i^2}
\]

When \(\text{ESS} < N/2\), the population is automatically resampled via systematic resampling and weights are reset to \(1/N\).

**What to watch:**
- ESS dropping to near 1 every generation means \(\epsilon\) is decreasing too fast — increase \(\epsilon\) or use a larger \(N\).
- ESS staying at \(N\) means the tolerance may be too loose — the posterior hasn't concentrated.
- The `n_resamples` counter in `ABCSMCState` tells you how often resampling triggered.

## 5. Kernel normalization

The Gaussian perturbation kernel is fully normalized:

\[
\log \mathcal{N}(x \mid \mu, \Sigma) = -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) - \frac{1}{2}(k \log 2\pi + \log\det\Sigma)
\]

This is numerically identical to `Distributions.logpdf(MvNormal(μ, Σ), x)` (unit-tested in 1-D and 2-D). Proper normalization matters because the kernel appears in the importance-weight denominator — an unnormalized kernel would bias the posterior.

## 6. Checkpoint portability

Checkpoints serialize **data only** — never `cfg` and its closures:

```julia
data = (θ, w, summaries, dists, ε, gen, ess, base_seed, n_attempts, n_resamples)
```

This means:

- A checkpoint saved with Julia 1.10 can be loaded with Julia 1.11.
- A checkpoint saved with one version of your model code can be loaded with an updated version — you just reattach the new `cfg`.
- Checkpoints are safe to copy across filesystems and Slurm nodes.

Atomicity is guaranteed by writing to `.tmp` then renaming (POSIX `rename` is atomic on the same filesystem). A job killed mid-write leaves at worst a stale `.tmp` file, which the loader ignores.