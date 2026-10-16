# Regression adjustment

Accepting particles with a finite tolerance \(\epsilon\) distorts an ABC
posterior: the accepted parameters are correlated with the deviation of their
simulated summaries from the observed summary, which biases the posterior and
inflates its variance. **Regression adjustment** (the *ABC regression
correction*) removes the leading part of that distortion by regressing the
parameters on the summaries.

Given accepted particles \((\theta_i, s_i)\) with importance weights \(w_i\) and
the observed summary \(s_{\text{obs}}\), the corrected parameters are

\[
\theta_i^{\star} = \theta_i - \hat{\beta}^{\top} (s_i - s_{\text{obs}}),
\]

where \(\hat{\beta}\) is a (weighted) least-squares fit of \(\theta\) on \(s\).

| Method | \(\hat{\beta}\) | Reference |
|--------|-----------------|-----------|
| OLS | weighted linear fit of \(\theta\) on \(s\) | Beaumont, Zhang & Balding (2002) |
| local linear | **kernel-weighted** fit centred at \(s_{\text{obs}}\) | Blum & François (2010) |

The local-linear variant also corrects the heteroscedasticity of the ABC
acceptance kernel (distant particles are noisier) and is usually the better
default. Both estimators are weighted least squares: they are fully
**deterministic** and depend only on `LinearAlgebra` and `Statistics`, so no
extra packages are pulled in.

## Adjusting a run

`adjusted_posterior` accepts an existing result object and returns a result of
the **same structure** with the particles replaced by their adjusted values:

```julia
using SequentialABC

adjusted = adjusted_posterior(state;                       # state::ABCSMCState
                              summaries_fn = s -> [s],     # one summary -> regressors
                              observed_summary = s_obs,
                              method = :local_linear)

adjusted.θ          # adjusted particles — plot with plot_marginals(adjusted)
adjusted.w          # weights are unchanged
```

`method = :ols` switches to the plain linear correction; for the local-linear
method the bandwidth `h` and the kernel can be overridden:

```julia
adjusted_posterior(state; summaries_fn = s -> [s], observed_summary = s_obs,
                   method = :local_linear, kernel = :epanechnikov, h = 0.2)
```

## Adjusting plain arrays

The low-level entry points operate on arrays and return the adjusted parameter
matrix, so they can also be used outside the sampler:

```julia
θ_ols = ols_adjust(Θ, S, s_obs)                    # Θ: N×d, S: N×q
θ_ll  = local_linear_adjust(Θ, S, s_obs; kernel = :gaussian)
```

Both accept optional importance weights `w` and return an `N × d`
`Matrix{Float64}`. The matching array-level `adjusted_posterior(Θ, S, s_obs)`
instead returns a [`RegressionAdjustment`](@ref) that also exposes the fitted
slopes and the bandwidth actually used.

## Example

A runnable demonstration lives in `examples/regression_adjustment.jl`:

```bash
julia --project=. examples/regression_adjustment.jl
```

It runs a tiny ABC-SMC on a linear-Gaussian model, applies both corrections and
shows that the adjusted posterior has the correct mean with a smaller spread.

## References

- Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian
  computation in population genetics. *Genetics*, 162(4), 2025–2035.
- Blum, M. G. B., & François, O. (2010). Non-linear regression models for
  Approximate Bayesian Computation. *Statistics and Computing*, 20(1), 63–73.

## API

```@docs
ols_adjust
local_linear_adjust
adjusted_posterior
RegressionAdjustment
```
