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

## 6. Checkpoint guarantees

Checkpoints serialize **data only** — never `cfg` and its closures:

```julia
data = (θ, w, summaries, dists, ε, gen, ess, base_seed, n_attempts, n_resamples)
```

What a checkpoint does and does not give you:

- **Exact continuation.** Loaded with a compatible `cfg` in a compatible
  environment, a checkpoint resumes the run with the same deterministic
  trajectory as an uninterrupted one — same particles, same weights, same ε.
  This is the guarantee the checkpoint-resume conformance tests pin down.
- **Current sampler state, not history.** A checkpoint holds the state of the
  generations completed so far. It is not a record of the run; for that, see
  the [run store](runstore.md).
- **Native format.** The payload is written with Julia's `Serialization`, so a
  checkpoint is tied to the Julia version that wrote it and to the types it
  holds. It is not a version-independent or cross-language format, and a
  checkpoint written before a change to the particle or summary types has to
  be migrated explicitly.
- **Configuration reattached by the caller.** `cfg` and its closures are
  never stored, so the model configuration is reattached on load. The stored
  *data* still has to be interpretable: if the meaning of a particle or a
  summary changed, migrate the checkpoint.
- **Write protocol.** The payload is written to a `.tmp` file and then renamed
  into place. On POSIX that rename is atomic within one filesystem, so a
  reader never sees a partially published checkpoint, and a process killed
  before the rename leaves the canonical file untouched and at worst a stale
  `.tmp` file, which the loader ignores. The guarantee is per filesystem: a
  move across filesystems is a copy, filesystems without an atomic rename do
  not provide it, and power loss during the write is not covered.
- **No integrity metadata.** A checkpoint carries no checksum, no integrity
  manifest, and no completion marker, and it is not authenticated. A damaged
  file is detected when it fails to deserialize, not before.

### Checkpoints and the run store are different mechanisms

The [run store](runstore.md) and the checkpoint answer different questions,
and neither replaces the other.

**Checkpoint — the exact sampler-restart authority:**

- holds the current sampler state: particles, weights, ε, seeds, counters, and
  the adaptive runtime state;
- resumes inference when it is loaded with a compatible `cfg`;
- is written with Julia's `Serialization`, so it is native and trusted input;
- carries no checksum, no manifest, and no completion marker.

**Run store — the queryable scientific history:**

- holds the accepted populations and the recorded evaluations of every
  generation, with provenance, diagnostics, and timings;
- adds SHA-256 digests, a deterministic manifest per generation, and a
  `COMPLETE` marker, so corruption and truncation are detected *before* the
  data is used, and every deserialized record is revalidated against the
  package's own invariants;
- is still a native, trusted-input format, not a portable cross-language one,
  and it is not authenticated either;
- is **not** sampler-restart state: it does not resume inference, and it does
  not make a checkpoint verifiable or replace one.