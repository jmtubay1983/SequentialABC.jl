# Adaptive kernels

ABC‑SMC and ABC‑MCMC both move particles with a Gaussian random walk
\(\theta' \sim \mathcal{N}(\theta, \Sigma)\). The perturbation kernel of the
sampler and the proposal of the chain are the *same* object, so both engines
benefit from the same machinery — and both suffer from the same hand‑tuning
problem:

- **\(\Sigma\) too small** — the population barely moves. Particles stay
  near‑duplicates of their ancestors, the accepted region is never explored, and
  the tolerance stops shrinking.
- **\(\Sigma\) too large** — almost every proposal lands outside the acceptance
  region. Simulations are wasted on redraws, and the importance weights become
  extremely uneven.

The standard remedy is to **estimate \(\Sigma\) from the current population and
update it as the run progresses**. `AdaptiveKernel` does exactly that, using only
`LinearAlgebra`, `Statistics` and `Random`.

## Estimating a covariance

`estimate_covariance` accepts either an \(N \times d\) matrix or a length‑\(N\)
vector of parameter vectors — exactly the layout of `ABCSMCState.θ` — together
with optional importance weights:

```julia
Σ̂ = estimate_covariance(state.θ, state.w)          # weighted, d × d
Σ̂ = estimate_covariance(matrix_of_samples)         # uniform weights
```

Two details matter for stability:

- **Reliability weights.** The normalisation is \(1 - \sum_i w_i^2\) with the
  weights normalised to sum to one, which reduces to the usual unbiased
  \(N - 1\) denominator for uniform weights.
- **Shrinkage.** The estimate is pulled towards isotropy,
  \(\tilde\Sigma = (1-\lambda)\hat\Sigma + \lambda\,(\operatorname{tr}\hat\Sigma/d)I\),
  with \(\lambda\) = `shrink` (default `1e-6`). A positive `shrink` makes the
  result **positive definite even for a rank‑deficient population** (\(N \le d\),
  duplicate particles, …). That is what keeps the Cholesky factorisation in the
  perturbation step from failing. Fully degenerate inputs fall back to
  `shrink · I`.

Because the estimator is weighted, it also works on the *importance‑weighted*
population, not just on a resampled one.

## Scaling rules

The estimated covariance is not used directly: it is multiplied by a rule of
thumb factor, `Σ = scaling_factor(d; rule) · Σ̂`.

| `rule` | factor | intended use |
|---|---|---|
| `:rosenthal` (alias `:gelman`) | \(2.38^2 / d\) | classic optimal random‑walk scaling for MCMC (Gelman, Roberts & Gilks 1996) |
| `:beaumont` (alias `:abc`) | \(2\) | twice the population covariance — the usual ABC‑SMC perturbation kernel, which must compensate for the resampling step (Beaumont et al. 2009) |
| `:unit` / `:none` | \(1\) | proposal covariance = estimated covariance |
| a `Real` | the value | any custom factor |

The estimate itself is anisotropic — every dimension keeps its own variance and
the correlations are preserved — so per‑dimension scaling comes for free.

## Adaptive ABC-SMC

`ABCSMCConfig` is immutable, so the way to install a new covariance is to rebuild
the config with `with_covariance`:

```julia
kernel = AdaptiveKernel([0.5 0.0; 0.0 0.5]; rule = :beaumont)

state = make_state(cfg, Vector{Float64}, 500, 4.0, 42)
initialize!(state, s_obs)

for t in 2:T
    update!(kernel, state.θ, state.w)                # Σ ← 2 · Cov(population)
    state.cfg = with_covariance(state.cfg, kernel)   # type-stable assignment
    run_generation!(state, s_obs; ε = epsilon_quantile(state, 0.5))
end
```

`ABCSMCState` is mutable and `with_covariance` preserves the concrete
configuration type, so the reassignment is type‑stable. `kernel.n_updates` counts
how many adaptations were applied, and a degenerate population is ignored rather
than adopted (a zero‑variance kernel would freeze the sampler).

## Adaptive ABC-MCMC

The chain's proposal is `cfg.Σ`, so the adapted covariance is installed the same
way:

```julia
cfg_adapted = with_covariance(cfg, kernel)
chain = run_mcmc(cfg_adapted, s_obs; n_samples = 5000, ε = ε_final, base_seed = 1)
acceptance_rate(chain)
```

Sampling from the kernel directly uses the same naming as the built‑in
`GaussianKernel`: `perturb(kernel, θ, rng)` draws a proposal (internal helper),
and calling the kernel — `kernel(θ_new, θ_old)` — returns the proposal log
density. The Gaussian random walk is symmetric, so that density is unchanged by
swapping its arguments; a *pre‑run* adaptation (estimate once, then freeze) is
therefore a valid Metropolis–Hastings proposal. Adapting the covariance
*continuously* inside a single chain requires vanishing adaptation to preserve
the stationary distribution and is deliberately out of scope here.

## Example

A runnable comparison lives in `examples/adaptive_kernel_example.jl`:

```bash
julia --project=. examples/adaptive_kernel_example.jl
```

It runs the same ABC‑SMC problem twice from a deliberately badly scaled
\(\Sigma_0\), once fixed and once adapted, and then reuses the adapted covariance
as the ABC‑MCMC proposal.

## References

- Beaumont, M. A., Cornuet, J.-M., Marin, J.-M., & Robert, C. P. (2009). Adaptive
  approximate Bayesian computation. *Biometrika*, 96(4), 983–990.
- Gelman, A., Roberts, G. O., & Gilks, W. R. (1996). Efficient Metropolis jumping
  rules. *Bayesian Statistics 5*, 599–607.

## API

```@docs
AdaptiveKernel
update!
estimate_covariance
adaptive_scale
scaling_factor
with_covariance
```
