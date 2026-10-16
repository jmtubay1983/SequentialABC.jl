# bench/bench_utils.jl
#
# Shared helpers for the engine benchmarks in `bench/compare_engines.jl`:
# the reference problem, its exact posterior, the metrics and the run wrappers.
#
# This file is `include`d by the benchmark scripts and is not part of the
# SequentialABC module.

using SequentialABC
using Distributions
using Statistics
using LinearAlgebra
using Random
using Printf

# ── the reference problem ────────────────────────────────────────────────────

"""
    GaussianProblem(; d = 2, τ = 2.0, σ = 0.5)

Reference problem shared by every benchmark row: a `N(0, τ²I)` prior on `θ` and
the summary `s(θ) = θ + N(0, σ²I)`, observed at `s_obs = θ_true` (a noiseless
observed summary, so the target is centred on it).

The likelihood is Gaussian, so the *exact* posterior is available in closed form
(see [`reference_posterior`](@ref)) — that is what every engine's posterior mean
is scored against. The ABC posterior is that reference smoothed by the tolerance,
so it is systematically *wider*: covariance errors are expected to be positive,
and the posterior-mean error is only a sharp diagnostic once `ε ≲ σ`.
"""
struct GaussianProblem
    d::Int
    τ::Float64
    σ::Float64
    θ_true::Vector{Float64}
    s_obs::Vector{Float64}
end

function GaussianProblem(; d::Integer = 2, τ::Real = 2.0, σ::Real = 0.5)
    d >= 1 || throw(ArgumentError("d must be ≥ 1, got $d"))
    τ > 0 || throw(ArgumentError("τ must be > 0, got $τ"))
    σ > 0 || throw(ArgumentError("σ must be > 0, got $σ"))
    θ_true = [isodd(i) ? 0.8 : -0.8 for i in 1:d]
    return GaussianProblem(Int(d), Float64(τ), Float64(σ), θ_true, copy(θ_true))
end

"""
    reference_posterior(p::GaussianProblem) -> NamedTuple{(:μ, :Σ)}

Exact posterior `N(μ, Σ)` of the reference problem:
`Σ = (τ⁻² + σ⁻²)⁻¹ I` and `μ = Σ s_obs / σ²`.
"""
function reference_posterior(p::GaussianProblem)
    Σ = (1 / (1 / p.τ^2 + 1 / p.σ^2)) * Matrix{Float64}(I, p.d, p.d)
    return (μ = Σ * (p.s_obs ./ p.σ^2), Σ = Σ)
end

problem_prior(p::GaussianProblem) = θ -> sum(logpdf(Normal(0.0, p.τ), x) for x in θ)
problem_sample_prior(p::GaussianProblem) = rng -> p.τ .* randn(rng, p.d)
problem_distance(p::GaussianProblem) = (s, so) -> sqrt(sum(abs2, s .- so))

"""
    problem_config(p::GaussianProblem, Σ; kernel = GaussianKernel()) -> ABCSMCConfig

An `ABCSMCConfig` for `p` with perturbation covariance `Σ`, usable by both engines
(ABC-SMC reads `Σ`; so does the ABC-MCMC proposal).
"""
function problem_config(p::GaussianProblem, Σ; kernel = GaussianKernel())
    simulator = ContinuousSimulator(model_builder = θ -> θ,
                                    evolve = (θ, rng) -> θ .+ p.σ .* randn(rng, p.d),
                                    summary = identity)
    return ABCSMCConfig(problem_prior(p), problem_sample_prior(p), identity,
                        problem_distance(p), kernel, Matrix{Float64}(Σ), simulator)
end

# ── metrics ──────────────────────────────────────────────────────────────────

weighted_mean(st::ABCSMCState) =
    [sum(st.w .* [θ[i] for θ in st.θ]) for i in 1:length(st.θ[1])]

# The weighted population covariance, for a posterior *covariance* diagnostic.
# (Reuses the package estimator with shrinkage switched off.)
weighted_cov(st::ABCSMCState) = estimate_covariance(st.θ, st.w; shrink = 0.0)

"""
    posterior_metrics(st, ref; sims, resamples, seconds, ess) -> NamedTuple

Score a posterior sample against the exact reference posterior `ref`:

  - `rmse`    — root-mean-square error of the posterior **mean**, per dimension
  - `mc_se`   — Monte-Carlo standard error of that mean, `√(mean(diag Σ̂) / ESS)`:
    the scale below which `rmse` differences between rows are noise, not signal
  - `cov_err` — relative Frobenius error `‖Σ̂ − Σ‖ / ‖Σ‖` of the posterior covariance
  - `ess`, `sims`, `resamples`, `ε`, `seconds` — run diagnostics

`st` may be an ABC-SMC state or the (uniformly weighted) state of an ABC-MCMC
chain; use `ess = chain_ess(chain)` for the latter, since a chain's `state.ess`
is just its sample count.
"""
function posterior_metrics(st::ABCSMCState, ref; sims::Integer, resamples::Integer,
                           seconds::Real, ess::Real = st.ess)
    μ = weighted_mean(st)
    Σ = weighted_cov(st)
    return (mean = μ, cov = Σ,
            rmse = sqrt(sum(abs2, μ .- ref.μ) / length(μ)),
            mc_se = sqrt(sum(diag(Σ)) / length(μ) / max(ess, 1)),
            cov_err = norm(Σ .- ref.Σ) / norm(ref.Σ),
            ess = Float64(ess), sims = Int(sims), resamples = Int(resamples),
            ε = st.ε, seconds = Float64(seconds))
end

"""
    mcmc_ess(x; max_lag = 500) -> Float64

Effective sample size of a scalar chain via Geyer's initial positive sequence:
successive autocovariance pairs `ρ_k + ρ_{k+1}` are accumulated while they stay
positive, and the estimator is `n / (1 + 2 Σ pairs)`. Callers take the minimum
over dimensions, i.e. the worst-mixing direction.
"""
function mcmc_ess(x::AbstractVector{<:Real}; max_lag::Integer = 500)
    n = length(x)
    n < 4 && return Float64(n)
    x̄ = mean(x)
    c0 = sum(abs2, x .- x̄) / n
    c0 > 0 || return Float64(n)

    τ = 1.0
    k = 1
    while k + 1 <= min(max_lag, n - 1)
        ck = sum((x[1:(n - k)] .- x̄) .* (x[(1 + k):n] .- x̄)) / n
        ck1 = sum((x[1:(n - k - 1)] .- x̄) .* (x[(2 + k):n] .- x̄)) / n
        pair = (ck + ck1) / c0
        pair <= 0 && break
        τ += 2 * pair
        k += 2
    end
    return n / τ
end

"""
    chain_ess(chain::ABCMCMCChain) -> Float64

Minimum over dimensions of [`mcmc_ess`](@ref) applied to the stored samples.
"""
function chain_ess(chain::ABCMCMCChain)
    d = length(chain.state.θ[1])
    return minimum(mcmc_ess([θ[i] for θ in chain.state.θ]) for i in 1:d)
end

# ── run wrappers ─────────────────────────────────────────────────────────────

# One SMC generation step: a prescribed tolerance ladder if given, otherwise the
# quantile schedule.
_step_ε(st::ABCSMCState, q, ladder, t) =
    ladder === nothing ? epsilon_quantile(st, q) : ladder[t]

"""
    run_smc_fixed(cfg, p; N, T, q, ε0, seed, ladder = nothing)
        -> (state, ladder, seconds)

ABC-SMC with the config's **fixed** perturbation covariance.

Pass `ladder` (one tolerance per generation, `T` entries) to run a *prescribed*
tolerance schedule instead of the adaptive quantile rule — that is how the
benchmark gives every row the same sequence of ABC targets.
"""
function run_smc_fixed(cfg, p::GaussianProblem; N::Int, T::Int, q::Real,
                       ε0::Real, seed::Int, ladder = nothing)
    st = make_state(cfg, Vector{Float64}, N, Float64(ε0), seed)
    t0 = time_ns()
    initialize!(st, p.s_obs)
    trace = Float64[st.ε]
    for t in 2:T
        run_generation!(st, p.s_obs; ε = _step_ε(st, q, ladder, t))
        push!(trace, st.ε)
    end
    return (state = st, ladder = trace, seconds = (time_ns() - t0) / 1e9)
end

"""
    run_smc_adaptive(cfg, p; N, T, q, ε0, seed, streaming, ladder = nothing,
                     kernel_rule = :beaumont) -> (state, ladder, kernel, seconds)

ABC-SMC with the perturbation covariance re-estimated from the population every
generation. `streaming = true` feeds the particles one at a time into the
kernel's `OnlineCovEstimator` and commits at the generation boundary;
`streaming = false` hands over the whole population in a single call. The two
must produce the same numbers — `compare_engines.jl` checks that.
"""
function run_smc_adaptive(cfg, p::GaussianProblem; N::Int, T::Int, q::Real,
                          ε0::Real, seed::Int, streaming::Bool, ladder = nothing,
                          kernel_rule = :beaumont)
    kernel = AdaptiveKernel(cfg.Σ; rule = kernel_rule)
    st = make_state(cfg, Vector{Float64}, N, Float64(ε0), seed)
    t0 = time_ns()
    initialize!(st, p.s_obs)
    trace = Float64[st.ε]
    for t in 2:T
        if streaming
            for i in eachindex(st.θ)
                update!(kernel, st.θ[i], st.w[i])
            end
            commit!(kernel)
        else
            update!(kernel, st.θ, st.w)
        end
        st.cfg = with_covariance(st.cfg, kernel)
        run_generation!(st, p.s_obs; ε = _step_ε(st, q, ladder, t))
        push!(trace, st.ε)
    end
    return (state = st, ladder = trace, kernel = kernel,
            seconds = (time_ns() - t0) / 1e9)
end

"""
    run_mcmc_bench(cfg, p; ε, n_samples, seed) -> (chain, seconds)

ABC-MCMC at tolerance `ε` with the proposal covariance of `cfg`.
"""
function run_mcmc_bench(cfg, p::GaussianProblem; ε::Real, n_samples::Int, seed::Int)
    t0 = time_ns()
    chain = run_mcmc(cfg, p.s_obs; n_samples = n_samples, ε = Float64(ε),
                     base_seed = seed)
    return (chain = chain, seconds = (time_ns() - t0) / 1e9)
end

"""
    repeat_min(f, reps) -> (result, min_seconds)

Call `f()` `reps` times. `f` must return a `NamedTuple` carrying a `seconds`
field. Seeds are pinned by the caller, so the statistics are identical in every
repetition and only the wall clock varies; the **minimum** elapsed time — the
least noisy estimator of the true cost — is returned alongside the last result.
"""
function repeat_min(f, reps::Integer)
    reps >= 1 || throw(ArgumentError("reps must be ≥ 1, got $reps"))
    local result
    best = Inf
    for _ in 1:reps
        result = f()
        best = min(best, result.seconds)
    end
    return result, best
end

# ── reporting ────────────────────────────────────────────────────────────────

"""
    markdown_table(headers, rows) -> String

Render a GitHub-flavoured Markdown table (used for the `--out` artifact).
"""
function markdown_table(headers::AbstractVector{<:AbstractString},
                        rows::AbstractVector{<:AbstractVector{<:AbstractString}})
    io = IOBuffer()
    println(io, "| ", join(headers, " | "), " |")
    println(io, "|", join(fill("---", length(headers)), "|"), "|")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    return String(take!(io))
end

"""
    run_header(mode, p, params) -> String

The parameter block printed at the top of every benchmark run (and written into
the artifact), so a result can always be tied back to its configuration.
"""
function run_header(mode::AbstractString, p::GaussianProblem, params)
    ref = reference_posterior(p)
    return join((
        "mode        $(mode)",
        "Julia       $(VERSION)",
        "problem     d = $(p.d), prior N(0, $(p.τ)²), noise σ = $(p.σ), s_obs = $(p.s_obs)",
        "exact       μ* = $(round.(ref.μ; digits = 6)), diag Σ* = $(round.(diag(ref.Σ); digits = 6))",
        "params      $(params)",
    ), "\n")
end
