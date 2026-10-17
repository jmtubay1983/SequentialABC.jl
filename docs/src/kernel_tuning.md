# Kernel tuning helpers

[`AdaptiveKernel`](@ref) adapts the *whole* covariance of the perturbation
\( \theta' \sim \mathcal{N}(\theta, \Sigma) \). When what you want instead is an
explicit, inspectable **per‑dimension bandwidth** — for a product kernel, a sweep
over scalings, or a proposal built outside the engines — the three helpers on
this page each make one choice, and `make_adaptive_kernel` composes them.

Everything here is pure and RNG‑free: the same population always yields the same
bandwidths, and nothing is wired into the engines automatically.

## Scaling rules

`auto_kernel_scale(d; rule, factor)` returns a scalar factor for a
`d`‑dimensional covariance. It covers the [`scaling_factor`](@ref) rules
(`:rosenthal`, `:beaumont`, `:unit`, or any real) and adds two dimension‑only
bandwidth heuristics:

| `rule` | factor | use |
|---|---|---|
| `:rosenthal` / `:gelman` | \( 2.38^2 / d \) | classic optimal random‑walk scaling (Gelman, Roberts & Gilks 1996) |
| `:beaumont` / `:abc` | \( 2 \) | ABC‑SMC perturbation scaling (Beaumont et al. 2009) |
| `:unit` / `:none` | \( 1 \) | no scaling |
| `:silverman` | \( (4/(d+2))^{1/(d+4)} \) | Silverman's rule‑of‑thumb bandwidth |
| `:median` | \( 2\,\mathrm{median}(\chi^2_d)/d \) | dimension‑only median heuristic |
| a `Real` | the value | any custom factor |

```julia
auto_kernel_scale(4)                      # 2.38² / 4 = 1.4161
auto_kernel_scale(4; rule = :silverman)   # (4/6)^(1/8) ≈ 0.9509
auto_kernel_scale(2; rule = :median)      # 2·log 2 ≈ 1.3863
auto_kernel_scale(4; factor = 0.5)        # half the Rosenthal factor
```

`:median` is the median heuristic in dimension‑only form: with no data to
measure, it uses the median of the \( \chi^2_d \) distribution — the typical
squared distance of a standard Gaussian in \( d \) dimensions.

## A positive‑definite covariance

`shrinkage_covariance(X; λ = 1e-6)` is the Ledoit–Wolf‑style shrinkage of the
sample covariance towards isotropy,

\[
\Sigma = (1-\lambda) S + \lambda \frac{\operatorname{tr} S}{d} I ,
\]

the same family used by [`estimate_covariance`](@ref). A positive `λ` makes the
result positive definite for **degenerate** input — repeated rows, \( N \le d \),
a single row — where the plain sample covariance is singular and a Cholesky
factorisation would fail. A population with no variance information at all falls
back to \( \lambda I \).

```julia
Σ̂ = shrinkage_covariance(matrix_of_samples)   # PD even if the rows repeat
```

## Bandwidths from a covariance

`bandwidth_from_cov(cov; method, min_bandwidth)` reads per‑dimension standard
deviations off a covariance:

- `:diag` (default) — the marginal standard deviations, `sqrt.(diag(cov))`;
- `:full` — the diagonal of the Cholesky factor, i.e. the **conditional**
  standard deviations \( \mathrm{sd}(\theta_i \mid \theta_1 \dots \theta_{i-1}) \),
  which reduce to `:diag` for a diagonal covariance and stay smaller under
  correlation.

Both are floored at `min_bandwidth` (default `1e-8`) so a zero‑variance dimension
is still usable.

## Putting it together

`make_adaptive_kernel(cfg; method = :auto, min_scale = 1e-6, rule = :rosenthal)`
returns a small callable that maps a particle population to bandwidths:

```julia
using SequentialABC

tuner = make_adaptive_kernel(cfg)     # cfg fixes the dimension
bw    = tuner(state.θ)                # per-dimension bandwidths, length d
```

The pipeline is exactly the three helpers: shrink the population covariance,
read bandwidths with `method` (`:auto` chooses `:full` when some pair of
dimensions is correlated beyond `0.1`, and `:diag` otherwise), scale by
`auto_kernel_scale(d; rule)`, and floor at `min_scale`.

Because the result is a vector of scales and not a kernel, installing it is
explicit — through the existing covariance plumbing, with no engine changes:

```julia
bw = tuner(state.θ)
state.cfg = with_covariance(state.cfg, Matrix(Diagonal(bw .^ 2)))
run_generation!(state, s_obs; ε = epsilon_quantile(state, 0.5))
```

For a directly *correlated* proposal, use [`AdaptiveKernel`](@ref), which adapts
the full covariance in place; these helpers are for callers that want the
per‑dimension view.

## API

```@docs
auto_kernel_scale
shrinkage_covariance
bandwidth_from_cov
make_adaptive_kernel
SequentialABC.TunedKernel
```
