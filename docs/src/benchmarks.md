# Benchmarks

`bench/compare_engines.jl` runs the two samplers — ABC‑SMC and ABC‑MCMC — against
a reference problem whose exact posterior is known in closed form, and reports
what each kernel choice actually buys:

| row | what it tests |
|---|---|
| `SMC fixed (Σ0)` | a badly scaled hand‑picked perturbation covariance |
| `SMC fixed (oracle 2Σ*)` | \(2\Sigma^\*\) of the *exact* posterior: the best a hand‑tuned fixed kernel can do |
| `SMC adaptive (batch)` | \(\Sigma\) re‑estimated from the population, one call per generation |
| `SMC adaptive (streaming)` | the same, one particle at a time, committed at the generation boundary |
| `MCMC fixed (Σ0)`, `MCMC fixed (oracle)`, `MCMC adaptive (pre-run)` | the chain's random‑walk proposal, fixed or adapted |

The point is not to prove that adaptation helps — that is the design of
[Adaptive kernels](adaptive_kernels.md) — but to quantify the cost and the accuracy of each
option, and to check that the streaming and batch estimators agree exactly.

## The reference problem

The prior is \(N(0, \tau^2 I)\) and the summary is the parameter plus Gaussian
noise, \(s(\theta) = \theta + N(0, \sigma^2 I)\), observed at
\(s_{\text{obs}} = \theta_{\text{true}}\). Because the likelihood is Gaussian the
**exact** posterior is available in closed form,

\[
\Sigma^\* = (\tau^{-2} + \sigma^{-2})^{-1} I, \qquad
\mu^\* = \Sigma^\* s_{\text{obs}} / \sigma^2 ,
\]

which gives every row an absolute accuracy reference — no Monte‑Carlo reference
posterior is needed. The defaults are \(d = 2\), \(\tau = 2\),
\(\sigma = 0.5\), so
\(\mu^\* = (0.752941, -0.752941)\) and
\(\operatorname{diag}\Sigma^\* = (0.235294, 0.235294)\).

Two consequences are worth stating up front, because they shape how the table
must be read:

- The ABC posterior is the exact posterior *smoothed by \(\varepsilon\)*: it is
  systematically **wider**. `cov err` is therefore expected to be positive, and
  its size measures the tolerance, not the sampler.
- At equal tolerance every row targets the *same* ABC posterior, so rows can
  differ in RMS posterior‑mean error only through Monte‑Carlo noise. The
  `mc se` column, \(\sqrt{\operatorname{mean}(\operatorname{diag}\hat\Sigma)/\text{ESS}}\),
  is the scale of that noise; an `RMSE` far above it would indicate a bug, and an
  `RMSE` below it is not a meaningful win.

## Running it

```bash
julia --project=. bench/compare_engines.jl                      # standard run (~20 s)
julia --project=. bench/compare_engines.jl --mode tiny          # CI-sized smoke run (~15 s)
julia --project=. bench/compare_engines.jl --mode full --out bench/results
```

| mode | SMC population `N` | generations `T` | MCMC snapshots | repetitions |
|---|---|---|---|---|
| `tiny` | 150 | 4 | 400 | 1 |
| `standard` | 400 | 6 | 2 000 | 1 |
| `full` | 800 | 8 | 10 000 | 3 |

Other flags: `--d`, `--seed`, `--reps`. `--out DIR` writes a Markdown table and a
CSV of the raw metrics; nothing is committed (the directory is ignored).

Seeds are pinned, so the statistics are identical in every repetition and only
the wall clock varies — the table reports the **minimum** time. The script ends
with hard assertions (structural invariants; the streamed and batched estimators
must agree; adaptation must beat the badly scaled kernel; and every posterior mean
must be consistent with the exact posterior within five Monte‑Carlo standard
errors), so a regression fails the run rather than printing a plausible table.

## Results

`--mode full`, Julia 1.13.2 on WSL2 (Ubuntu), seed 42, single core. The
simulation counts, ESS values and acceptance rates are hardware‑independent;
the wall‑clock column is indicative only.

```
mode        full
problem     d = 2, prior N(0, 2.0²), noise σ = 0.5, s_obs = [0.8, -0.8]
exact       μ* = [0.752941, -0.752941], diag Σ* = [0.235294, 0.235294]
params      N = 800, T = 8, q = 0.60, ε0 = 4.0, n_mcmc = 10000, reps = 3, seed = 42
baselines   Σ0 = 4.0·I   Σ_oracle_smc = 0.4706·I   Σ_oracle_mcmc = 0.6664·I
ladder      4.0 → 2.5264 → 1.8718 → 1.4074 → 1.0491 → 0.7874 → 0.6118 → 0.4645
```

| engine | sims | wall (s) | ESS | resamp | ε | accept | RMSE | mc se | cov err | Σ₁₁ |
|---|---|---|---|---|---|---|---|---|---|---|
| SMC fixed (Σ0) | 82126 | 6.08 | 783.9 | 0 | 0.465 | — | 0.0189 | 0.0192 | 0.241 | 4.0000 |
| SMC fixed (oracle 2Σ\*) | 25709 | 2.72 | 718.9 | 0 | 0.465 | — | 0.0467 | 0.0198 | 0.206 | 0.4706 |
| SMC adaptive (batch) | 32723 | 3.35 | 718.3 | 0 | 0.465 | — | 0.0090 | 0.0208 | 0.321 | 4.0000 |
| SMC adaptive (streaming) | 32723 | 3.25 | 718.3 | 0 | 0.465 | — | 0.0090 | 0.0208 | 0.321 | 4.0000 |
| MCMC fixed (Σ0) | 10064 | 0.56 | 81.2 | 0 | 0.465 | 0.023 | 0.0069 | 0.0575 | 0.240 | 4.0000 |
| MCMC fixed (oracle) | 10064 | 0.55 | 143.8 | 0 | 0.465 | 0.084 | 0.0311 | 0.0448 | 0.236 | 0.6664 |
| MCMC adaptive (pre-run) | 10064 | 0.54 | 231.9 | 0 | 0.465 | 0.069 | 0.0522 | 0.0350 | 0.211 | 0.8464 |

`sims` = simulator calls; `wall (s)` = minimum over the repetitions; `ESS` = the
final population's ESS (ABC‑SMC) or Geyer's initial‑positive‑sequence estimate,
minimised over dimensions (ABC‑MCMC); `mc se` = Monte‑Carlo standard error of the
posterior mean; `Σ₁₁` = the kernel the row actually used.

### What the numbers say

- **Adaptation pays for SMC.** The adaptive kernel reaches the same tolerance with
  **2.5× fewer simulations** than the badly scaled \(\Sigma_0 = 4I\)
  (32 723 vs 82 126) and comes within 27 % of the cost of the oracle kernel that
  *knows* the answer (25 709), so adaptation closes most of the gap for free. The
  same ordering holds in `tiny` (1.3×) and `standard` (1.9×) mode: the efficiency
  gain grows with the runtime, which is what one expects from a kernel that
  converges to the population shape.
- **Adaptation pays for MCMC — through mixing, not through the simulation count.**
  ABC‑MCMC simulates exactly once per Metropolis step, so *every* MCMC row spends
  the same 10 064 simulations; only the acceptance rate and the ESS differ. The
  adapted proposal (\(2.38^2/d \cdot \operatorname{Cov}(\text{ABC population})\),
  \(\Sigma_{11} = 0.846\)) delivers **2.9× the ESS** of the badly scaled
  \(\Sigma_0\) (231.9 vs 81.2) and 1.6× the ESS of the oracle proposal tuned to the
  *exact* posterior (143.8).
- **The "oracle" is not the target.** The oracle kernel is optimal for the exact
  posterior, but the sampler targets the \(\varepsilon\)‑smoothed ABC posterior,
  which is wider; a proposal matched to the *widened* posterior mixes better. That
  is the same reason a fixed kernel needs to be *deliberately not* the exact
  covariance, and it is exactly what the adaptive estimator provides.
- **Streaming and batch agree exactly.** Both adaptation paths produce the same
  simulation count, the same population and the same posterior mean to within
  \(2\times10^{-15}\) — floating‑point noise. This is asserted, not just reported.
- **Accuracy is not the differentiator.** Every SMC row ran the *same* tolerance
  ladder and every MCMC row the same \(\varepsilon\), and all `RMSE` values sit at
  or below `mc se`; the maximum `RMSE / mc se` over the seven rows is 2.36, well
  inside the five‑standard‑error consistency bound. Different kernels buy the
  same posterior at different prices.

### Two design details

**One shared tolerance ladder.** ABC‑SMC with the quantile schedule chooses its own
\(\varepsilon\) per generation, so two runs that differ only in their kernel end
up targeting *different* posteriors — a well‑mixing kernel shrinks \(\varepsilon\)
faster and would be penalised for it. The harness therefore calibrates a ladder
once (an adaptive run at \(2N\) particles, so it is well mixed and tight) and
runs every SMC row at those tolerances. The MCMC rows share the ladder's final
\(\varepsilon\).

**Adaptation is frozen before the chain runs.** The MCMC proposal is estimated
from the ABC‑SMC population once and then held fixed. Adapting a proposal from
the chain's own history is the textbook way to break the Markov property and
invalidate the chain, so the benchmark uses a pre‑run estimate — the same
recommendation as [Adaptive kernels](adaptive_kernels.md).

### Caveats

- **ESS is not comparable across engines.** For ABC‑SMC it measures weight
  degeneracy in the final generation (and is reset to \(N\) whenever the
  population is resampled — the `resamp` column shows this did not happen here).
  For ABC‑MCMC it is an autocorrelation‑based effective sample size of the whole
  chain. Use each within its own column, not as a cross‑engine score.
- **`tiny` mode is a smoke test.** With \(T = 4\) generations the tolerance only
  reaches \(\varepsilon \approx 1.3\), far larger than \(\sigma = 0.5\): the ABC
  posterior is then dominated by smoothing and the accuracy columns say almost
  nothing. `tiny` checks the pipeline, the estimator parity and the efficiency
  ordering; the quantitative claims come from `full`.
- **Wall clock is single‑machine.** Only `sims`, `ESS` and `accept` are portable.
- The model is deliberately simple: the exact posterior only exists because the
  likelihood is Gaussian. The comparison is about the *kernels*, which is what the
  random‑walk proposals control.
