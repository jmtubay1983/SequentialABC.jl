# ABC-MCMC

ABC-SMC walks a population through a sequence of tolerances; **ABC-MCMC**
(Marjoram, Molitor, Plagnol & Tavaré 2003) instead runs a single Markov chain
whose stationary distribution is the ABC posterior

\[
\pi_\epsilon(\theta \mid s_{\text{obs}}) \;\propto\; \pi(\theta)\,
\mathbf{1}\!\left[d\big(s(\theta), s_{\text{obs}}\big) \le \epsilon\right].
\]

At a fixed tolerance, a proposal \(\theta' \sim \mathcal{N}(\theta, \Sigma)\) is
accepted when

1. the simulated summary is close enough, \(d(s(\theta'), s_{\text{obs}}) \le
   \epsilon\), **and**
2. a Metropolis–Hastings test with the prior ratio passes,
   \(u < \min\!\big(1, \exp(\log \pi(\theta') - \log \pi(\theta))\big)\).

The Gaussian random walk is symmetric, so the proposal densities cancel and only
the prior ratio is left. Compared with ABC-SMC, ABC-MCMC needs no tolerance
schedule and no resampling — but the samples are autocorrelated, and a poor
starting point is expensive because the chain *stays* there.

## Running a chain

```julia
using SequentialABC, Distributions, Random

prior(θ) = logpdf(Normal(0.0, 2.0), θ[1])
sample_prior(rng) = [2.0 * randn(rng)]

simulator = ContinuousSimulator(model_builder = θ -> θ,
                                evolve = (θ, rng) -> θ[1] + 0.3 * randn(rng),
                                summary = identity)

cfg = ABCSMCConfig(prior, sample_prior, identity,
                   (s, so) -> abs(s - so), GaussianKernel(), [0.05;;], simulator)

chain = run_mcmc(cfg, s_obs; n_samples = 2000, ε = 0.3, base_seed = 2024)

acceptance_rate(chain)   # ≈ expected MH acceptance rate
chain.state.θ            # the samples
```

The chain is initialised by drawing from the prior until `d ≤ ε`, then advanced
for `n_samples * thin` iterations. One sample is stored per `thin` iterations,
plus the initial state, so `length(chain) == n_samples + 1`.

## Determinism

Every proposal, simulation and accept/reject draw uses a `MersenneTwister`
seeded from `make_particle_seed(1, iteration, 1, role, base_seed)`, with distinct
role sentinels for the three streams. The same configuration and `base_seed`
therefore replay a **bit-identical** chain, on any machine or Slurm node. The
uniform draw for the acceptance test is taken unconditionally, even when the
distance test already fails, so the random stream never depends on the data.

## Reusing the rest of the package

The chain stores its samples in an [`ABCSMCState`](@ref) with uniform weights —
the same container an ABC-SMC run produces. Everything that consumes a result
therefore works on MCMC output unchanged:

```julia
plot_marginals(chain.state; labels = ["a", "b"])                  # figures
adjusted_posterior(chain.state; observed_summary = s_obs)         # correction
save_checkpoint(chain.state, "chain.jls")                         # checkpoints
```

## Example

A runnable two-parameter demonstration lives in `examples/two_param_mcmc.jl`:

```bash
julia --project=. examples/two_param_mcmc.jl
```

It runs a chain on a linear-regression summary, reports the posterior means
against the true coefficients, verifies reproducibility, and applies the
regression correction to the chain.

## References

- Marjoram, P., Molitor, J., Plagnol, V., & Tavaré, S. (2003). Markov chain
  Monte Carlo without likelihoods. *PNAS*, 100(26), 15324–15328.

## API

```@docs
run_mcmc
ABCMCMCChain
acceptance_rate
abc_mcmc_accept
mcmc_propose
mcmc_log_proposal
```
