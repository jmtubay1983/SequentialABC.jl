# Examples

SequentialABC.jl ships with two example models demonstrating the continuous and agent-based simulator backends.

## 1. Continuous logistic growth

**File:** `examples/continuous_logistic.jl`

Infers the growth rate \(r\) of a logistic model from a noisy observation.

### Model

\[
N(t) = \frac{K}{1 + \left(\frac{K - N_0}{N_0}\right) e^{-r t}}
\]

with \(K = 100\), \(N_0 = 1\), observed at \(t = 3\).

### Prior

\(r \sim \text{Uniform}(0, 1)\)

### Distance

\[
d = |N_{\text{sim}}(3) - N_{\text{obs}}(3)|
\]

### Running

```julia
include("examples/continuous_logistic.jl")
```

### Output

```
Observed endpoint: N(3) = 53.170  (true r = 0.80)

Generation  ε         ESS     attempts
───────────────────────────────────────
    1      5.0000    150.0   1629
    2      2.7010    149.9   2941
    3      1.5451    150.0   5192
    4      0.8454    149.9   9161
    5      0.3965    150.0   17924
    6      0.2110    150.0   33283

Posterior r: mean = 0.771, sd = 0.031   (true = 0.80)
Final ε = 0.2110, ESS = 150.0, resamplings = 0
```

The posterior mean (0.771) is close to the true value (0.80), demonstrating that the sampler recovers the generating parameter.

## 2. Schelling segregation (Agents.jl)

**File:** `examples/agents_schelling.jl`

Infers the homophily threshold of a Schelling segregation model from an observed segregation index.

### Model

A 2D grid of agents of two types. Agents are unhappy if fewer than a threshold fraction of their neighbors are of the same type. Unhappy agents move to a random empty cell.

### Prior

Homophily threshold \(\tau \sim \text{Uniform}(0, 1)\)

### Distance

\[
d = |\text{segregation}_{\text{sim}} - \text{segregation}_{\text{obs}}|
\]

### Running

```julia
include("examples/agents_schelling.jl")
```

This example requires `Agents.jl`. If it is not installed, the example is skipped gracefully (as in the test suite).

## CLI runner

**File:** `bin/sequentialabc-run.jl`

A command-line entry point for HPC workloads:

```bash
julia bin/sequentialabc-run.jl --config cli_config.jl --checkpoint-dir checkpoints/
```

See [**HPC / Slurm**](hpc.md) for cluster usage.

## Reproducing the examples

Both examples are included in the package test suite (`test/test_examples.jl`). The logistic example always runs; the Schelling example runs only when `Agents.jl` is available.