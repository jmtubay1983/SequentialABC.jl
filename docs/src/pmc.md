# Population Monte Carlo

ABC-SMC moves a population through a sequence of tolerances and, whenever the
weights degenerate, **resamples** the population so every particle carries equal
weight again. **Population Monte Carlo** (Beaumont, Cornuet, Marin & Robert 2009)
keeps the same redraw-until-accepted population loop but drops that resampling
step, and instead re-derives the perturbation covariance from the population at
every generation. It is the algorithm of Toni, Liepe & Stelling (2009) run with
the two choices that give PMC its name.

## What distinguishes PMC here

1. **The perturbation covariance is re-derived from the population at every
   generation**, as
   \[
   \Sigma_t \;=\; c(d,\text{rule}) \cdot \widehat{\mathrm{Cov}}\!\left(\theta_{t-1},
   w_{t-1}\right),
   \]
   with `kernel_rule = :beaumont` — a factor of 2 — as the default. `Σ_t` is
   adopted at the *generation boundary*, before any particle of generation `t`
   exists, so one covariance serves the whole generation and its importance
   weights. [`run_smc`](@ref) keeps `cfg.Σ` fixed instead, which is what makes
   hand tuning matter.

2. **The population is never resampled.** The ESS-triggered
   [`systematic_resample`](@ref) step that ABC-SMC fires at `ESS < 0.5 N` is
   skipped: the weights stay informative across generations and the *weighted
   ancestor draws* do the importance sampling — the mechanism PMC is named for.
   Consequently `state.ess` reports the real effective sample size of each
   generation rather than resetting to `N`, and `state.n_resamples` stays `0`.
   That counter is the quickest way to confirm you are running PMC.

Everything else is deliberately identical to `run_smc`, so the two engines are
directly comparable and the rest of the package works unchanged: the
perturbation-corrected weight

\[
w(\theta) \;\propto\; \frac{\pi(\theta)}{\sum_j w_j\, K(\theta \mid \theta_j, \Sigma)}
\]

from [`importance_weight`](@ref) — the same formula in both engines — the
deterministic per-particle streams from [`make_particle_seed`](@ref), the same
diagnostics, and the same [`ABCSMCState`](@ref) return value.

## Running a population

```julia
using SequentialABC, Distributions, Random

prior(θ) = sum(logpdf(Normal(0.0, 2.0), x) for x in θ)
sample_prior(rng) = 2.0 .* randn(rng, 2)

simulator = ContinuousSimulator(model_builder = θ -> θ,
                                evolve = (θ, rng) -> θ .+ 0.5 .* randn(rng, 2),
                                summary = identity)

cfg = ABCSMCConfig(prior, sample_prior, identity,
                   (s, so) -> sqrt(sum(abs2, s .- so)),
                   GaussianKernel(), [1.0 0.0; 0.0 1.0], simulator)
s_obs = [0.8, -0.8]

# A prescribed tolerance ladder — the historical input of Toni/Beaumont:
# generation t runs at εs[t], so generation 1 runs at εs[1].
state = run_pmc(cfg, s_obs; N = 500, T = 5,
                εs = [4.0, 2.0, 1.0, 0.6, 0.4], base_seed = 2026)

state.ε              # 0.4 — the last rung
state.n_resamples    # 0 — PMC never resamples
state.ess            # true weighted ESS of the final population
```

Passing no ladder instead selects an **adaptive** tolerance: the default
`schedule = QuantileSchedule(0.5)` picks each ε from the previous generation's
distances, exactly as `run_smc` does, with `ε0 = Inf` making generation 1 a plain
prior sample.

```julia
adaptive = run_pmc(cfg, s_obs; N = 500, T = 5, base_seed = 2026)
```

A `FixedSchedule([...])` or any closure `(state, t) -> ε_t` is accepted in place
of the quantile rule.

## The trade-off

Not resampling is what makes PMC's weights meaningful across generations, and it
is also its cost: without the reset that ABC-SMC gets from resampling, the ESS
can drift down as the tolerance tightens. PMC pays for that by re-estimating
`Σ_t` from the *weighted* population, which keeps the perturbations matched to
the region that still carries weight.

Watch `state.ess` — or compare the engines directly, which is what
`bench/compare_engines.jl` does on a Gaussian reference problem with a known
exact posterior. Its PMC row runs the same tolerance ladder as the ABC-SMC rows,
so the two are scored at identical ABC targets and differ only in the kernel rule
and the resampling choice; the harness reports the true weighted ESS for PMC
against the post-resampling ESS of the SMC rows.

## Determinism, checkpointing and resumption

Seeds come from the same `make_particle_seed(gen, particle, M, role, base_seed)`
roles as `run_smc`, so a given configuration and `base_seed` replays a
**bit-identical** population on any machine or Slurm node.

Checkpointing behaves the same way, and [`run_pmc!`](@ref) is the state-based
driver in the same relationship to `run_pmc` that `run_smc!` has to `run_smc`:

```julia
run_pmc(cfg, s_obs; N = 500, T = 5, εs = ladder,
        checkpoint_dir = "ck")                       # writes one file per generation

st = load_latest_state("ck", cfg)                    # resume
run_pmc!(st, s_obs; T = 10,                          # ← pass the same schedule …
         schedule = FixedSchedule(ladder_full),      #    and rule you started with
         checkpoint_dir = "ck")
```

Nothing about the schedule or `kernel_rule` is checkpointed — they are a symbol
and a function, not serialisable state — so resuming carries the same contract
`run_smc!` already has: pass the same `T`, `schedule` and `kernel_rule` you
started with. With those, a resumed run reproduces an uninterrupted one
bit-for-bit.

## API

```@docs
run_pmc
run_pmc!
```

## References

- Beaumont, M. A., Cornuet, J.-M., Marin, J.-M., & Robert, C. P. (2009).
  Adaptive approximate Bayesian computation. *Biometrika*, 96(4), 983–990.
- Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stelling, J. D. (2009).
  Approximate Bayesian computation scheme for parameter inference and model
  selection in dynamical systems. *Journal of the Royal Society Interface*,
  6(31), 187–202.
