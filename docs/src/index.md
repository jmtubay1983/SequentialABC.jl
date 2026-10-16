# SequentialABC.jl

**Deterministic, restartable, HPC-ready Approximate Bayesian Computation via Sequential Monte Carlo (ABC‑SMC).**

## What is ABC‑SMC?

Approximate Bayesian Computation (ABC) lets you do Bayesian inference when the likelihood is intractable but you can simulate from the model. Instead of computing \(p(y \mid \theta)\), you accept parameter draws \(\theta\) whose simulated summaries are close to the observed summary \(s_{\text{obs}}\), within a tolerance \(\epsilon\).

Sequential Monte Carlo (SMC) improves on rejection ABC by maintaining a population of weighted particles and gradually reducing \(\epsilon\) across generations, reweighting and resampling as the population concentrates toward the posterior.

## What SequentialABC.jl implements

SequentialABC.jl is a Julia implementation of the Toni et al. (2009) ABC‑SMC algorithm, designed from the ground up for **scientific reproducibility** and **high-performance computing**:

| Feature | Why it matters |
|---------|----------------|
| ✅ Deterministic RNG | Every random draw is seeded from `(gen, i, attempt, role, base_seed)` — runs are bit-identical across restarts, nodes, and Julia sessions. |
| ✅ Restartable checkpoints | Atomic `save_checkpoint` / `load_latest_state` let you resume long runs after preemption, with no data loss. |
| ✅ No global RNG | All RNGs are passed explicitly — safe for Slurm array jobs and parallel workers. |
| ✅ Systematic resampling | Lowest-variance standard resampling scheme, unit-tested against uniformity and point-mass cases. |
| ✅ Kernel normalization | Gaussian perturbation kernel exactly matches `Distributions.logpdf(MvNormal)`. |
| ✅ ESS diagnostics | Effective sample size tracked every generation; automatic resampling when ESS < N/2. |
| ✅ ε schedules | `FixedSchedule` (hand-picked) and `QuantileSchedule` (adaptive) tolerance schedules. |
| ✅ Dual simulator backends | `ContinuousSimulator` (ODE/PDE/SSA) and `AgentsSimulator` (agent-based models via Agents.jl). |

## Why determinism matters

Most ABC‑SMC implementations silently use the global RNG. That means:

- Two runs with the same seed can diverge if particle acceptance counts differ.
- A Slurm preemption mid-run forces you to start over — or worse, resume with different randomness and get a different posterior.
- You cannot reproduce a reviewer's results from a published seed.

SequentialABC.jl avoids all of this. Every stochastic step is a pure function of `(gen, particle_index, attempt, role, base_seed)`. Resume a checkpointed run and you get the **bit-identical** trajectory you would have had without interruption.

## Installation

```julia
using Pkg
Pkg.add("SequentialABC")
```

## Citation

If you use SequentialABC.jl in your research, please cite:

- Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. (2009). Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. *Journal of the Royal Society Interface*, 6(31), 187–202.

## Next steps

- [**Quickstart**](quickstart.md) — run your first ABC‑SMC in 20 lines.
- [**API Reference**](api.md) — full function and type documentation.
- [**Examples**](examples.md) — logistic growth and Schelling segregation.
- [**Design Notes**](design.md) — determinism, transform vs prior space, kernel normalization.
- [**HPC / Slurm**](hpc.md) — running on clusters with checkpoint/restart.