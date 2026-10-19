# test/conformance_tests.jl
#
# Baseline statistical conformance of existing production behavior.
#
# These tests exercise current production APIs only. They use independent
# oracles rather than calling production helpers to calculate expected values.
# If any test exposes a production defect, stop and propose a separate
# bug-fix PR — do not change production source here.

using SequentialABC
using Test
using Distributions
using Random
using Statistics
using LinearAlgebra
using LogExpFunctions
using Serialization

# ── shared fixtures ──────────────────────────────────────────────────────────

# Scalar Gaussian benchmark (tests 1b): theta ~ N(0, 5^2), summary = theta + N(0,1),
# observed summary = 1.0.
PRIOR_SD = 5.0
NOISE = 1.0
S_OBS_1D = [1.0]

sample_prior_1d = rng -> PRIOR_SD .* randn(rng, 1)
prior_logpdf_1d = θ -> sum(logpdf(Normal(0.0, PRIOR_SD), x) for x in θ)
transform_identity = θ -> θ
distance_absdiff = (s, s_obs) -> sqrt(sum(abs2, s .- s_obs))
Σ_i1 = Matrix{Float64}(I, 1, 1)
sim_1d = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ NOISE .* randn(rng, 1),
    summary = identity,
)
cfg_gauss = ABCSMCConfig(
    prior_logpdf_1d, sample_prior_1d, transform_identity,
    distance_absdiff, GaussianKernel(), Σ_i1, sim_1d,
)

# Bimodal benchmark (tests 1d): theta ~ N(0, 4^2), s = theta^2 + N(0, 0.1),
# observed summary = 4.0 → posterior mass near theta = ±2.
PRIOR_SD_BI = 4.0
NOISE_BI = 0.1
S_OBS_BI = [4.0]

sample_prior_bi = rng -> PRIOR_SD_BI .* randn(rng, 1)
prior_logpdf_bi = θ -> sum(logpdf(Normal(0.0, PRIOR_SD_BI), x) for x in θ)
distance_absdiff_1d = (s, s_obs) -> abs(s[1] - s_obs[1])
sim_bi = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> [θ[1]^2 + NOISE_BI * randn(rng)],
    summary = identity,
)
cfg_bi = ABCSMCConfig(
    prior_logpdf_bi, sample_prior_bi, transform_identity,
    distance_absdiff_1d, GaussianKernel(), Matrix{Float64}(I, 1, 1), sim_bi,
)

# ── independent weight oracle (test 1a) ──────────────────────────────────────

# log w(theta) = log prior(theta) - logsumexp_j( log w_prev[j] + log K(theta | theta_prev[j]) )
# Standalone: does not call SequentialABC.importance_weight.
function weight_oracle(prior_logpdf, θ_prop, θ_prev, w_prev, Σ_inv, logdetΣ)
    k = length(θ_prop)
    logprior = prior_logpdf(θ_prop)
    terms = Vector{Float64}(undef, length(θ_prev))
    for j in eachindex(θ_prev)
        wj = w_prev[j]
        if wj == 0.0
            terms[j] = -Inf   # log(0) = -Inf → excluded from log-sum-exp
            continue
        end
        diff = θ_prop .- θ_prev[j]
        logK = -0.5 * (diff' * Σ_inv * diff) - 0.5 * (k * log(2π) + logdetΣ)
        terms[j] = log(wj) + logK
    end
    logdenom = logsumexp(terms)
    return exp(logprior - logdenom)
end

function logsumexp_indep(x::AbstractVector{<:Real})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

@testset "Importance-weight oracle vs production" begin
    base_seed = 20260201
    Σ = Matrix{Float64}([2.0 0.0; 0.0 3.0])
    Σ_inv = inv(Σ)
    logdetΣ = logdet(Σ)
    Σ_full = Matrix{Float64}([2.0 0.9; 0.9 3.0])
    Σ_full_inv = inv(Σ_full)
    logdetΣ_full = logdet(Σ_full)
    # 1-D covariance inverse (cfg_gauss uses Σ_i1 = [1.0]).
    Σ_inv_1d = inv(Σ_i1)
    logdetΣ_1d = logdet(Σ_i1)

    # 2-D fixtures for the weight oracle.
    θ_prop2d = [0.5, -0.2]
    θ_prev2d = [[-0.3, 0.4], [0.1, -0.1], [0.8, 0.2]]
    sim_2d = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ 1.0 .* randn(rng, 2),
        summary = identity,
    )
    cfg_2d_diag = ABCSMCConfig(
        θ -> sum(logpdf(Normal(0.0, PRIOR_SD), x) for x in θ),
        rng -> PRIOR_SD .* randn(rng, 2),
        transform_identity, distance_absdiff,
        GaussianKernel(), Σ, sim_2d,
    )
    cfg_2d_full = ABCSMCConfig(
        θ -> sum(logpdf(Normal(0.0, PRIOR_SD), x) for x in θ),
        rng -> PRIOR_SD .* randn(rng, 2),
        transform_identity, distance_absdiff,
        GaussianKernel(), Σ_full, sim_2d,
    )

    # 1-D fixtures.
    θ_prev1 = [[-0.3], [0.1], [0.8]]
    w_prev1 = [0.2, 0.5, 0.3]   # sums to 1

    # Case 1: unequal previous weights, diagonal covariance (1-D), normalized weights.
    w_prod1 = importance_weight(cfg_gauss, [0.5], θ_prev1, w_prev1)
    w_ora1 = weight_oracle(prior_logpdf_1d, [0.5], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    @test isapprox(w_prod1, w_ora1; rtol=1e-10)
    w_prod1b = importance_weight(cfg_gauss, [1.2], θ_prev1, w_prev1)
    w_ora1b = weight_oracle(prior_logpdf_1d, [1.2], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    @test isapprox(w_prod1b, w_ora1b; rtol=1e-10)
    w_prod1c = importance_weight(cfg_gauss, [-0.8], θ_prev1, w_prev1)
    w_ora1c = weight_oracle(prior_logpdf_1d, [-0.8], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    @test isapprox(w_prod1c, w_ora1c; rtol=1e-10)

    # Case 2: duplicate ancestors (repeated θ values), diagonal covariance.
    θ_prev2 = [[0.1], [0.1], [0.8]]
    w_prev2 = [0.3, 0.2, 0.5]
    w_prod2 = importance_weight(cfg_gauss, [0.5], θ_prev2, w_prev2)
    w_ora2 = weight_oracle(prior_logpdf_1d, [0.5], θ_prev2, w_prev2, Σ_inv_1d, logdetΣ_1d)
    @test isapprox(w_prod2, w_ora2; rtol=1e-10)

    # Case 3: zero-weight ancestor (denominator term excluded), diagonal covariance.
    θ_prev3 = [[-0.3], [0.1], [0.8]]
    w_prev3 = [0.0, 0.5, 0.5]
    w_prod3 = importance_weight(cfg_gauss, [0.5], θ_prev3, w_prev3)
    w_ora3 = weight_oracle(prior_logpdf_1d, [0.5], θ_prev3, w_prev3, Σ_inv_1d, logdetΣ_1d)
    @test isapprox(w_prod3, w_ora3; rtol=1e-10)

    # Case 4: zero proposal-density denominator → documented degenerate (Inf).
    w_prev4 = [0.0, 0.0, 0.0]
    w_prod4 = importance_weight(cfg_gauss, [0.5], θ_prev3, w_prev4)
    @test isinf(w_prod4) && w_prod4 > 0   # degenerate: no usable finite weight

    # Case 5: diagonal covariance (independent dimensions), 2-D.
    w_prev5 = [0.25, 0.35, 0.40]
    w_prod5 = importance_weight(cfg_2d_diag, θ_prop2d, θ_prev2d, w_prev5)
    w_ora5 = weight_oracle(
        θ -> sum(logpdf(Normal(0.0, PRIOR_SD), x) for x in θ),
        θ_prop2d, θ_prev2d, w_prev5, Σ_inv, logdetΣ,
    )
    @test isapprox(w_prod5, w_ora5; rtol=1e-10)

    # Case 6: full covariance (off-diagonal nonzero), 2-D.
    w_prev6 = [0.4, 0.3, 0.3]
    w_prod6 = importance_weight(cfg_2d_full, θ_prop2d, θ_prev2d, w_prev6)
    w_ora6 = weight_oracle(
        θ -> sum(logpdf(Normal(0.0, PRIOR_SD), x) for x in θ),
        θ_prop2d, θ_prev2d, w_prev6, Σ_full_inv, logdetΣ_full,
    )
    @test isapprox(w_prod6, w_ora6; rtol=1e-10)

    # Case 7: bounded prior support — Uniform on [0, 10]; proposal outside support
    # has log prior −Inf → weight 0 (not Inf, not NaN).
    bounded_prior = θ -> θ[1] >= 0 && θ[1] <= 10 ? 0.0 : -Inf
    w_ora7_in = weight_oracle(bounded_prior, [5.0], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    w_ora7_out = weight_oracle(bounded_prior, [15.0], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    @test isfinite(w_ora7_in) && w_ora7_in > 0   # finite inside support
    @test w_ora7_out == 0.0                      # zero weight outside support

    # Case 8: extreme finite log densities (large negative prior).
    extreme_prior = θ -> -1e8
    w_ora_extreme = weight_oracle(extreme_prior, [0.5], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    @test w_ora_extreme == 0.0 || isfinite(w_ora_extreme)   # underflow to 0, not NaN

    # Case 9: nonfinite inputs produce NaN consistently in production and oracle
    # (clearly invalid, never silently accepted as a usable finite weight).
    w_prod_nan = importance_weight(cfg_gauss, [NaN], θ_prev1, w_prev1)
    w_ora_nan = weight_oracle(prior_logpdf_1d, [NaN], θ_prev1, w_prev1, Σ_inv_1d, logdetΣ_1d)
    @test isnan(w_prod_nan) && isnan(w_ora_nan)

    # Case 10: normalized vs unnormalized previous weights scale inversely.
    w_prev_norm = [0.2, 0.5, 0.3]
    w_prev_unnorm = [2.0, 5.0, 3.0]   # 10x the normalized ones
    w_prod_norm = importance_weight(cfg_gauss, [0.5], θ_prev1, w_prev_norm)
    w_prod_unnorm = importance_weight(cfg_gauss, [0.5], θ_prev1, w_prev_unnorm)
    @test isapprox(w_prod_norm / w_prod_unnorm, 10.0; rtol=1e-10)

    # Case 11: final population weights are finite, nonnegative, normalized.
    state = run_smc(cfg_gauss, S_OBS_1D; N=50, T=3, ε0=Inf,
                    base_seed=base_seed,
                    schedule=FixedSchedule([3.0, 1.5, 0.7]))
    w = state.w
    @test all(isfinite, w)
    @test all(>(0.0), w)
    @test isapprox(sum(w), 1.0; atol=1e-12)
end


@testset "Finite-tolerance Gaussian ABC benchmark (independent reference)" begin
    # theta ~ N(0, 5^2), summary = theta + N(0, 1), observed summary = 1.0.
    # Independent finite-tolerance ABC reference via numerical integration over theta.
    # Acceptance weight for tolerance eps:
    #   a(theta) = P(|theta + noise - 1| <= eps), noise ~ N(0,1)
    #            = Phi(eps - (theta - 1)) - Phi(-eps - (theta - 1))
    # ABC posterior (finite tolerance): p(theta) prop. prior(theta) * a(theta).
    base_seed = 20260201
    final_eps = 0.5
    std_normal = Normal(0.0, 1.0)

    acceptance(theta, eps) = cdf(std_normal, eps - (theta - 1.0)) -
                             cdf(std_normal, -eps - (theta - 1.0))

    # Integration domain and precision (prespecified).
    theta_lo, theta_hi = -12.0, 14.0
    dtheta = 0.0005
    thetas = theta_lo:dtheta:theta_hi

    # Reference (unnormalized then normalized).
    w_ref = Vector{Float64}(undef, length(thetas))
    for (k, th) in enumerate(thetas)
        w_ref[k] = pdf(Normal(0.0, PRIOR_SD), th) * acceptance(th, final_eps)
    end
    w_ref ./= sum(w_ref)

    ref_mean = sum(thetas .* w_ref)
    ref_var = sum(w_ref .* (thetas .- ref_mean).^2)
    ref_sd = sqrt(max(ref_var, 0.0))
    # Selected quantiles from the normalized reference.
    cdf_ref = cumsum(w_ref)
    function ref_quantile(q)
        idx = searchsortedfirst(cdf_ref, q)
        return thetas[min(idx, length(thetas))]
    end
    ref_q05 = ref_quantile(0.05)
    ref_q50 = ref_quantile(0.50)
    ref_q95 = ref_quantile(0.95)

    # Run production ABC-SMC with a prespecified schedule and budget.
    # Generation 1 uses eps0 = Inf (prior); generations 2..T use the schedule.
    state = run_smc(cfg_gauss, S_OBS_1D; N=200, T=3, ε0=Inf,
                    base_seed=base_seed,
                    schedule=FixedSchedule([1.0, final_eps, final_eps]))

    wn = state.w ./ sum(state.w)
    thetas_run = [θ[1] for θ in state.θ]
    est_mean = sum(wn .* thetas_run)
    est_var = sum(wn .* (thetas_run .- est_mean).^2)
    est_sd = sqrt(max(est_var, 0.0))

    # Separate ABC-reference and Monte-Carlo tolerances (prespecified).
    tol_mc_mean = 0.25
    tol_mc_sd = 0.35
    tol_mc_q = 0.30

    @test isapprox(state.gen, 3)
    @test abs(est_mean - ref_mean) < tol_mc_mean
    @test abs(est_sd - ref_sd) < tol_mc_sd

    # Quantile comparison (selected quantiles).
    sorted_idx = sortperm(thetas_run)
    sorted_vals = thetas_run[sorted_idx]
    sorted_w = wn[sorted_idx]
    cdf_run = cumsum(sorted_w)
    function run_quantile(q)
        idx = searchsortedfirst(cdf_run, q)
        return sorted_vals[min(idx, length(sorted_vals))]
    end
    @test abs(run_quantile(0.05) - ref_q05) < tol_mc_q
    @test abs(run_quantile(0.50) - ref_q50) < tol_mc_q
    @test abs(run_quantile(0.95) - ref_q95) < tol_mc_q

    # Reference values reported for the record.
    @test isfinite(ref_mean) && isfinite(ref_sd)
    @test isfinite(est_mean) && isfinite(est_sd)

    # Print for human review (captured in test logs).
    println("  [ref] mean=$(round(ref_mean; digits=4)) sd=$(round(ref_sd; digits=4)) q05=$(round(ref_q05; digits=4)) q50=$(round(ref_q50; digits=4)) q95=$(round(ref_q95; digits=4))")
    println("  [run] mean=$(round(est_mean; digits=4)) sd=$(round(est_sd; digits=4))")
end


@testset "Correlated-kernel benchmark (independent calculation)" begin
    # Σ2 = [4 1.8; 1.8 9], corr = 0.3.
    # Independent expectations:
    #   :diag -> sqrt.(diag) = [2.0, 3.0]
    #   :full -> diag of Cholesky L (conditional sds):
    #       L[1,1] = sqrt(4) = 2.0
    #       L[2,2] = sqrt(9 - 1.8^2/4) = sqrt(8.19)
    Σ2 = [4.0 1.8; 1.8 9.0]
    exp_diag = [2.0, 3.0]
    exp_full_2 = sqrt(9.0 - 1.8^2 / 4.0)
    exp_full = [2.0, exp_full_2]

    # Symmetry and positive definiteness (independent checks).
    @test Σ2 == permutedims(Σ2)                      # symmetric
    @test all(eigvals(Matrix{Float64}(Σ2)) .> 0)     # positive definite
    F = cholesky(Symmetric(Σ2))
    @test F isa Cholesky

    # Production bandwidths match the independent calculation.
    bw_diag = bandwidth_from_cov(Σ2; method=:diag)
    bw_full = bandwidth_from_cov(Σ2; method=:full)
    @test isapprox(bw_diag, exp_diag; atol=1e-12)
    @test isapprox(bw_full, exp_full; atol=1e-12)

    # Diagonal mode: zero off-diagonal covariance.
    Σ_diag = Matrix{Float64}(Diagonal([4.0, 9.0]))
    @test Σ_diag[1, 2] == 0.0 && Σ_diag[2, 1] == 0.0
    bw_diag2 = bandwidth_from_cov(Σ_diag; method=:diag)
    bw_diag_full = bandwidth_from_cov(Σ_diag; method=:full)
    @test isapprox(bw_diag2, bw_diag_full; atol=1e-12)   # reduces to :diag when diagonal

    # Full mode: nonzero off-diagonal covariance.
    Σ_full = [4.0 1.8; 1.8 9.0]
    @test Σ_full[1, 2] == 1.8 && Σ_full[2, 1] == 1.8   # nonzero off-diagonal

    # Proposal log densities differ at a prespecified off-axis point.
    # Independent Gaussian log-density:
    #   log K = -0.5 * diff' Σ_inv diff - 0.5 * (k*log(2π) + logdetΣ)
    θ_new = [1.0, 0.5]
    θ_old = [0.0, 0.0]
    k2 = 2
    function indep_logpdf(diff, Σ_inv, logdetΣ)
        return -0.5 * (diff' * Σ_inv * diff) - 0.5 * (k2 * log(2π) + logdetΣ)
    end

    Σ_diag_inv = inv(Σ_diag)
    logdet_diag = logdet(Σ_diag)
    Σ_full_inv = inv(Σ_full)
    logdet_full = logdet(Σ_full)

    diff = θ_new .- θ_old
    lp_diag_indep = indep_logpdf(diff, Σ_diag_inv, logdet_diag)
    lp_full_indep = indep_logpdf(diff, Σ_full_inv, logdet_full)

    # Production GaussianKernel log-density matches independent calculation.
    K = GaussianKernel()
    lp_diag_prod = K(θ_new, θ_old, Σ_diag_inv, logdet_diag)
    lp_full_prod = K(θ_new, θ_old, Σ_full_inv, logdet_full)
    @test isapprox(lp_diag_prod, lp_diag_indep; atol=1e-12)
    @test isapprox(lp_full_prod, lp_full_indep; atol=1e-12)

    # Full and diagonal proposal log densities differ at the off-axis point.
    @test lp_full_prod != lp_diag_prod
    @test lp_full_indep != lp_diag_indep

    # make_adaptive_kernel :diag vs :full give different scales on correlated pop.
    sim_2d = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ 1.0 .* randn(rng, 2),
        summary = identity,
    )
    cfg2 = ABCSMCConfig(
        θ -> sum(logpdf(Normal(0.0, PRIOR_SD), x) for x in θ),
        rng -> PRIOR_SD .* randn(rng, 2),
        transform_identity, distance_absdiff,
        GaussianKernel(), Σ2, sim_2d,
    )
    ker_diag = make_adaptive_kernel(cfg2; method=:diag, min_scale=1e-6)
    ker_full = make_adaptive_kernel(cfg2; method=:full, min_scale=1e-6)
    rng_pop = MersenneTwister(7)
    z = randn(rng_pop, 10)
    pop = [[z[i], 0.9 * z[i] + 0.1 * randn(rng_pop)] for i in 1:10]
    bw_kd = ker_diag(pop)
    bw_kf = ker_full(pop)
    @test !isapprox(bw_kd, bw_kf; atol=1e-10)
    @test length(bw_kd) == 2
    @test length(bw_kf) == 2
    @test all(isfinite, bw_kd) && all(isfinite, bw_kf)
    @test all(>(0.0), bw_kd) && all(>(0.0), bw_kf)
end


@testset "Bimodal benchmark (prespecified symmetric finite-tolerance target)" begin
    # theta ~ N(0, 4^2), s = theta^2 + N(0, 0.1), observed summary = 4.0
    # -> posterior mass near theta = +/- 2 (two symmetric modes).
    base_seed = 20260202
    min_mode_mass = 0.10
    mode_boundary = 0.0   # theta == 0 separates the two modes

    state_bi = run_smc(cfg_bi, S_OBS_BI; N=120, T=5, ε0=Inf,
                       base_seed=base_seed,
                       schedule=QuantileSchedule(0.3))

    θ_vals = [θ[1] for θ in state_bi.θ]
    w_bi = state_bi.w ./ sum(state_bi.w)

    @test isapprox(sum(w_bi), 1.0; atol=1e-12)      # total normalized weight ~ one
    @test all(isfinite, w_bi)                       # no nonfinite weights
    @test all(>(0.0), w_bi)                       # nonnegative

    mass_neg = sum(w_bi[θ_vals .< mode_boundary])   # negative mode
    mass_pos = sum(w_bi[θ_vals .> mode_boundary])   # positive mode
    mass_at_boundary = sum(w_bi[θ_vals .== mode_boundary])

    @test mass_neg >= min_mode_mass                 # both meaningful modes
    @test mass_pos >= min_mode_mass
    @test isapprox(mass_neg + mass_pos + mass_at_boundary, 1.0; atol=1e-12)

    # Stability under the fixed seed and budget: rerun, compare.
    state_bi2 = run_smc(cfg_bi, S_OBS_BI; N=120, T=5, ε0=Inf,
                        base_seed=base_seed,
                        schedule=QuantileSchedule(0.3))
    @test state_bi.θ == state_bi2.θ
    @test state_bi.w == state_bi2.w
    w_bi2 = state_bi2.w ./ sum(state_bi2.w)
    θ_vals2 = [θ[1] for θ in state_bi2.θ]
    mass_neg2 = sum(w_bi2[θ_vals2 .< mode_boundary])
    @test isapprox(mass_neg, mass_neg2; atol=1e-12)

    # Record mode treatment for the log.
    println("  [bimodal] mass_neg=$(round(mass_neg; digits=4)) mass_pos=$(round(mass_pos; digits=4)) mass_at_boundary=$(round(mass_at_boundary; digits=6))")
    println("  [bimodal] boundary=theta=0; particles at boundary treated as neither mode")
end


@testset "Transformed and bounded benchmark (independent reference on inference coordinate)" begin
    # phi ~ N(log(2), 0.5^2), theta = exp(phi), summary = theta + N(0, 0.1),
    # observed summary = 2.0. Transform applied exactly once at simulation time.
    true_theta = 2.0
    mu_phi = log(true_theta)
    sd_phi = 0.5
    noise_lt = 0.1
    S_OBS_LT = [true_theta]
    base_seed = 20260203

    prior_logpdf_lt = φ -> sum(logpdf(Normal(mu_phi, sd_phi), x) for x in φ)
    sample_prior_lt = rng -> mu_phi .+ sd_phi .* randn(rng, 1)
    exp_transform(φ) = exp.(φ)
    distance_lt = (s, s_obs) -> abs(s[1] - s_obs[1])
    sim_lt = ContinuousSimulator(
        model_builder = φ -> φ,
        evolve = (θ, rng) -> θ .+ noise_lt .* randn(rng, 1),
        summary = identity,
    )
    cfg_lt = ABCSMCConfig(
        prior_logpdf_lt, sample_prior_lt, exp_transform,
        distance_lt, GaussianKernel(), Matrix{Float64}(I, 1, 1), sim_lt,
    )

    # Independent finite-tolerance ABC reference on the phi (inference) coordinate.
    # Acceptance: P(|exp(phi) + noise - 2| <= eps), noise ~ N(0, 0.1)
    final_eps_lt = 0.3
    noise_dist = Normal(0.0, noise_lt)
    acceptance_lt(phi) = cdf(noise_dist, final_eps_lt - (true_theta - exp(phi[1]))) -
                         cdf(noise_dist, -final_eps_lt - (true_theta - exp(phi[1])))

    phi_lo, phi_hi = mu_phi - 4*sd_phi, mu_phi + 4*sd_phi
    dphi = 0.0005
    phis = phi_lo:dphi:phi_hi
    w_ref_lt = Vector{Float64}(undef, length(phis))
    for (k, ph) in enumerate(phis)
        w_ref_lt[k] = pdf(Normal(mu_phi, sd_phi), ph[1]) * acceptance_lt(ph)
    end
    w_ref_lt ./= sum(w_ref_lt)

    # Transform reference quantities to the theta (summary) coordinate.
    thetas_ref = exp.(phis)
    ref_mean_theta = sum(thetas_ref .* w_ref_lt)
    ref_var_theta = sum(w_ref_lt .* (thetas_ref .- ref_mean_theta).^2)
    ref_sd_theta = sqrt(max(ref_var_theta, 0.0))
    cdf_ref_lt = cumsum(w_ref_lt)
    function ref_quantile_lt(q)
        idx = searchsortedfirst(cdf_ref_lt, q)
        return thetas_ref[min(idx, length(thetas_ref))]
    end
    ref_q05_lt = ref_quantile_lt(0.05)
    ref_q50_lt = ref_quantile_lt(0.50)

    # Production run on the inference coordinate (transform applied once).
    # FixedSchedule needs T-1 entries for gens 2..T; final gen uses final_eps_lt.
    state_lt = run_smc(cfg_lt, S_OBS_LT; N=100, T=5, ε0=Inf,
                       base_seed=base_seed,
                       schedule=FixedSchedule([1.0, 0.5, final_eps_lt,
                                               final_eps_lt, final_eps_lt]))

    phi_vals = [φ[1] for φ in state_lt.θ]
    w_lt = state_lt.w ./ sum(state_lt.w)
    theta_est = sum(w_lt .* exp.(phi_vals))
    theta_vals = exp.(phi_vals)

    # Support: every transformed value lies in support (theta > 0).
    @test all(>(0.0), theta_vals)
    @test all(isfinite, w_lt)
    @test isapprox(sum(w_lt), 1.0; atol=1e-12)

    # Estimated transformed mean/quantiles against the reference.
    tol_mc_mean_lt = 0.4
    tol_mc_sd_lt = 0.4
    tol_mc_q_lt = 0.30
    @test abs(theta_est - ref_mean_theta) < tol_mc_mean_lt
    est_sd_lt = sqrt(max(sum(w_lt .* (theta_vals .- theta_est).^2), 0.0))
    @test abs(est_sd_lt - ref_sd_theta) < tol_mc_sd_lt

    sorted_idx_lt = sortperm(theta_vals)
    sorted_vals_lt = theta_vals[sorted_idx_lt]
    sorted_w_lt = w_lt[sorted_idx_lt]
    cdf_run_lt = cumsum(sorted_w_lt)
    function run_quantile_lt(q)
        idx = searchsortedfirst(cdf_run_lt, q)
        return sorted_vals_lt[min(idx, length(sorted_vals_lt))]
    end
    @test abs(run_quantile_lt(0.05) - ref_q05_lt) < tol_mc_q_lt
    @test abs(run_quantile_lt(0.50) - ref_q50_lt) < tol_mc_q_lt

    println("  [transformed] ref_mean_theta=$(round(ref_mean_theta; digits=4)) run_mean_theta=$(round(theta_est; digits=4))")
    println("  [transformed] support min theta=$(round(minimum(theta_vals); digits=6)) (all > 0)")
end


@testset "Systematic-resampling benchmark (independent oracle)" begin
    base_seed = 20260204

    # Fixed weight vector (some zero weights).
    w = [0.0, 0.1, 0.2, 0.0, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05]
    Nw = length(w)

    # Independent systematic-resampling oracle (standalone, not calling production).
    function systematic_resample_oracle(w::AbstractVector{<:Real}, rng::AbstractRNG)
        N = length(w)
        u = rand(rng) / N
        positions = u .+ (0:N-1) ./ N
        cumw = cumsum(Float64.(w))
        cumw[end] = 1.0
        idx = Vector{Int}(undef, N)
        j = 1
        for i in 1:N
            while j < N && positions[i] > cumw[j]
                j += 1
            end
            idx[i] = j
        end
        return idx
    end

    # Deterministic replay: same weights + same seed -> identical indices.
    idx_a = systematic_resample(w, MersenneTwister(base_seed))
    idx_b = systematic_resample(w, MersenneTwister(base_seed))
    @test idx_a == idx_b

    # Exactly N ancestor indices, all valid.
    @test length(idx_a) == Nw
    @test all(1 .<= idx_a .<= Nw)

    # Zero-weight ancestors never selected.
    zero_idx = Set(findall(==(0.0), w))
    @test isempty(intersect(Set(idx_a), zero_idx))

    # Consistency with the independent oracle (identical rng stream).
    idx_ora = systematic_resample_oracle(w, MersenneTwister(base_seed))
    @test idx_a == idx_ora

    # Bounded empirical discrepancy across a predefined set of seed identities.
    # Systematic resampling selects index i either floor(N*w_i) or ceil(N*w_i)
    # times per run, so long-run frequency is within ~1/N of w_i.
    seed_identities = [base_seed + k for k in 1:20]
    counts = zeros(Int, Nw)
    for s in seed_identities
        idx = systematic_resample(w, MersenneTwister(s))
        for j in idx
            counts[j] += 1
        end
    end
    total_draws = length(seed_identities) * Nw
    for j in 1:Nw
        freq = counts[j] / total_draws
        @test abs(freq - w[j]) <= 1.0 / Nw + 0.02
    end
end


@testset "Movement benchmark (proposal-covariance-relative)" begin
    # Assess movement relative to the configured proposal covariance, not an
    # arbitrary fraction of prior width.
    base_seed = 20260205
    d = 2
    Σ_move = Matrix{Float64}(I, d, d)   # unit proposal covariance
    K_move = GaussianKernel()
    n_anc = 20

    # Duplicate ancestors: all start at the same point.
    ancestor = [0.5, -0.3]
    ancestors = [copy(ancestor) for _ in 1:n_anc]

    descendants = Vector{Vector{Float64}}(undef, n_anc)
    for i in 1:n_anc
        rng = MersenneTwister(base_seed + i)
        descendants[i] = SequentialABC.perturb(K_move, ancestor, Σ_move, rng)
    end

    # Duplicate ancestors can generate distinct descendants.
    n_unique = length(unique(descendants))
    @test n_unique >= n_anc - 1   # essentially all distinct

    # Standardized displacement is nonzero and consistent with the kernel.
    displacements = [descendants[i] .- ancestor for i in 1:n_anc]
    std_displacements = [displacements[i] ./ sqrt.(diag(Σ_move)) for i in 1:n_anc]
    n_nonzero = count(d -> any(!=(0.0), d), std_displacements)
    @test n_nonzero == n_anc   # every descendant moved

    # Empirical descendant covariance is nondegenerate.
    X = reduce(hcat, descendants)'   # n_anc x d
    emp_cov = cov(X)
    @test all(eigvals(Matrix{Float64}(emp_cov)) .> 0)

    # Support handling: Gaussian kernel has unbounded support; every proposal
    # is finite and in support.
    @test all(isfinite, X)

    # No NaN/Inf proposals.
    @test all(all(isfinite, desc) for desc in descendants)

    # Production perturb matches an independent Gaussian draw:
    #   θ_new ~ N(θ_old, Σ)
    # Mean over many perturbations should be near the ancestor.
    n_draws = 500
    draws = [SequentialABC.perturb(K_move, ancestor, Σ_move, MersenneTwister(1000 + i)) for i in 1:n_draws]
    draw_mean = mean(reduce(hcat, draws)', dims=1)[:]
    @test isapprox(draw_mean[1], ancestor[1]; atol=0.15)
    @test isapprox(draw_mean[2], ancestor[2]; atol=0.15)
end


@testset "Adaptive-distance benchmark (independent verification)" begin
    # Heterogeneous 2-key summaries: :a low variance, :b high variance.
    base_seed = 20260206
    N_ad = 60
    key_order = [:a, :b]

    # Constructed summaries matrix: column :a tight, column :b diffuse.
    rng_ad = MersenneTwister(base_seed)
    X_ad = Matrix{Float64}(undef, N_ad, 2)
    for i in 1:N_ad
        X_ad[i, 1] = 0.0 + 0.1 * randn(rng_ad)     # :a tight
        X_ad[i, 2] = 0.0 + 5.0 * randn(rng_ad)     # :b diffuse
    end
    w_uniform = fill(1.0 / N_ad, N_ad)

    # Estimated scaling values (independent: std of each column).
    std_a = std(X_ad[:, 1])
    std_b = std(X_ad[:, 2])
    exp_w_a = 1.0 / std_a
    exp_w_b = 1.0 / std_b
    @test exp_w_a > exp_w_b   # :a tighter -> higher inverse-std weight

    # Production estimate_weights matches the independent scaling.
    cfg_ad = AdaptiveDistanceConfig(StandardDeviationScale())
    w_est = SequentialABC.estimate_weights(cfg_ad.strategy, X_ad, w_uniform,
                                           key_order, cfg_ad.policy)
    @test isapprox(w_est[1], exp_w_a; rtol=1e-12)
    @test isapprox(w_est[2], exp_w_b; rtol=1e-12)
    @test w_est[1] > w_est[2]

    # Numerical policy: constant (zero-variance) key with :clamp -> max_weight.
    X_const = [X_ad[:, 1] zeros(N_ad)]   # second key constant zero
    w_clamp = SequentialABC.estimate_weights(StandardDeviationScale(), X_const,
                                             w_uniform, key_order,
                                             NumericalPolicy(constant_policy=:clamp))
    @test w_clamp[2] == cfg_ad.policy.max_weight   # max_weight for constant key

    # Numerical policy: constant key with :error throws.
    @test_throws ArgumentError SequentialABC.estimate_weights(
        StandardDeviationScale(), X_const, w_uniform, key_order,
        NumericalPolicy(constant_policy=:error))

    # Transformed distances after scaling differ from non-adaptive on a
    # constructed heterogeneous summary.
    s_sim = (a = 1.0, b = 10.0)
    s_obs = (a = 0.0, b = 0.0)
    base_dist = (x, y) -> sqrt((x.a - y.a)^2 + (x.b - y.b)^2)
    nonadaptive_dist = base_dist(s_sim, s_obs)

    # Build an adaptive state with precomputed weights (11-arg constructor +
    # direct field assignment, since adaptive is not a positional arg).
    ad_state = SequentialABC.AdaptiveDistanceState(key_order, w_est, 1,
                                                   Float64[], 0.5)
    state_ad = ABCSMCState(cfg_ad, Vector{Float64}[], Vector{Float64}[],
                           NamedTuple[], Float64[], 0.5, 1, 0.0, base_seed, 0, 0)
    state_ad.adaptive = ad_state

    adaptive_d = SequentialABC.adaptive_distance(state_ad, base_dist, s_sim,
                                                 s_obs)
    # Weighted distance: sqrt((w_a*1.0)^2 + (w_b*10.0)^2) vs unweighted sqrt(1+100).
    @test adaptive_d != nonadaptive_dist
    @test isfinite(adaptive_d) && adaptive_d > 0

    # Checkpointed adaptive state reproduces the same subsequent calculation.
    # Run adaptive to gen 2, checkpoint, resume, continue to gen 3; compare
    # adaptive state to an uninterrupted run to gen 3.
    PRIOR_A = Normal(0.0, 2.5)
    PRIOR_B = Normal(0.0, 25.0)
    sample_prior_2key = rng -> [rand(rng, PRIOR_A), rand(rng, PRIOR_B)]
    prior_logpdf_2key = θ -> logpdf(PRIOR_A, θ[1]) + logpdf(PRIOR_B, θ[2])
    simulator_2key = ContinuousSimulator(
        model_builder = μ -> [μ[1], μ[2]],
        evolve = (θ, rng) -> θ .+ randn(rng, 2),
        summary = y -> (a = y[1], b = y[2]),
    )
    distance_2key = (s, so) -> sqrt((s.a - so.a)^2 + (s.b - so.b)^2)
    Σ_2key = [0.5 0.0; 0.0 0.5]
    cfg_2key = ABCSMCConfig(
        prior_logpdf_2key, sample_prior_2key, transform_identity,
        distance_2key, GaussianKernel(), Σ_2key, simulator_2key,
    )
    Y_OBS_2 = (a = 1.0, b = 0.0)

    dir_ad = mktempdir()
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    state_full = make_state(cfg_2key, Vector{Float64}, N_ad, 3.0, base_seed)
    run_smc!(state_full, Y_OBS_2; T = 3,
             schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = ad_config, checkpoint_dir = dir_ad)

    # Resume from gen-2 checkpoint and continue to gen 3.
    dir_part = mktempdir()
    state_part = make_state(cfg_2key, Vector{Float64}, N_ad, 3.0, base_seed)
    run_smc!(state_part, Y_OBS_2; T = 2,
             schedule = FixedSchedule([3.0, 1.5]),
             adaptive = ad_config, checkpoint_dir = dir_part)
    state_resumed = load_latest_state(dir_part, cfg_2key)
    run_smc!(state_resumed, Y_OBS_2; T = 3,
             schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = ad_config, checkpoint_dir = dir_part)

    @test state_resumed.gen == state_full.gen
    @test state_resumed.adaptive.gen_applied == state_full.adaptive.gen_applied
    @test state_resumed.adaptive.weights == state_full.adaptive.weights
    @test state_resumed.adaptive.epsilon == state_full.adaptive.epsilon
    @test state_resumed.adaptive.epsilon_selection_distances ==
        state_full.adaptive.epsilon_selection_distances
end


@testset "Failed and nonfinite simulation benchmark" begin
    # Software failures must never silently become ordinary finite discrepancies.
    base_seed = 20260207

    # 1. Simulator exception propagates (not converted to a numeric penalty).
    boom_sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (problem, rng) -> error("simulated simulator failure: boom"),
        summary = identity,
    )
    cfg_boom = ABCSMCConfig(
        prior_logpdf_1d, sample_prior_1d, transform_identity, distance_absdiff,
        GaussianKernel(), Σ_i1, boom_sim,
    )
    propagated = nothing
    try
        run_smc(cfg_boom, S_OBS_1D; N=10, T=2, base_seed=base_seed,
                schedule=FixedSchedule([3.0, 2.0, 1.0]))
    catch e
        propagated = e
    end
    @test propagated !== nothing
    @test occursin("simulated simulator failure: boom",
                   sprint(showerror, propagated))

    # 2. NaN summary cannot be silently accepted: distance is NaN, so
    #    d <= eps is false and the particle is never accepted (unreachable).
    nan_sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> [NaN],
        summary = identity,
    )
    cfg_nan = ABCSMCConfig(
        prior_logpdf_1d, sample_prior_1d, transform_identity, distance_absdiff,
        GaussianKernel(), Σ_i1, nan_sim,
    )
    nan_propagated = nothing
    try
        run_smc(cfg_nan, S_OBS_1D; N=5, T=1, base_seed=base_seed, ε0=1.0)
    catch e
        nan_propagated = e
    end
    @test nan_propagated !== nothing   # rejected, never silently accepted

    # 3. Inf summary similarly rejected.
    inf_sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> [Inf],
        summary = identity,
    )
    cfg_inf = ABCSMCConfig(
        prior_logpdf_1d, sample_prior_1d, transform_identity, distance_absdiff,
        GaussianKernel(), Σ_i1, inf_sim,
    )
    inf_propagated = nothing
    try
        run_smc(cfg_inf, S_OBS_1D; N=5, T=1, base_seed=base_seed, ε0=1.0)
    catch e
        inf_propagated = e
    end
    @test inf_propagated !== nothing   # rejected, never silently accepted

    # 4. A normal simulator still succeeds (sanity that the harness isn't broken).
    state_ok = run_smc(cfg_gauss, S_OBS_1D; N=20, T=2, ε0=Inf,
                       base_seed=base_seed,
                       schedule=FixedSchedule([3.0, 2.0]))
    @test state_ok.gen == 2
    @test all(isfinite, state_ok.dists)
end


@testset "RNG contract (make_particle_seed)" begin
    # generation, particle index, attempt, replicate/role, base seed.
    base_seed = 20260208
    tuples = [(1, 1, 1, 1, base_seed),
              (1, 2, 1, 1, base_seed),
              (2, 1, 1, 1, base_seed),
              (1, 1, 2, 1, base_seed),
              (1, 1, 1, 2, base_seed),
              (1, 1, 1, 1, base_seed + 1),
              (3, 7, 12, 3, base_seed)]

    # 1. Identical tuples produce identical seeds.
    seeds = [make_particle_seed(t[1], t[2], t[3], t[4], t[5]) for t in tuples]
    @test all(isfinite, seeds)
    @test length(unique(seeds)) == length(seeds)   # distinct tuples -> distinct seeds

    # 2. Deterministic replay: same tuple, separate calls -> same seed.
    for t in tuples
        @test make_particle_seed(t[1], t[2], t[3], t[4], t[5]) ==
              make_particle_seed(t[1], t[2], t[3], t[4], t[5])
    end

    # 3. Collision scan over a predefined representative campaign:
    #    no two distinct tuples in the campaign collide.
    campaign = [(g, i, a, m, b) for g in 1:3 for i in 1:5 for a in 1:4
                for m in 1:2 for b in [base_seed, base_seed + 1]]
    campaign_seeds = [make_particle_seed(c[1], c[2], c[3], c[4], c[5])
                      for c in campaign]
    @test length(unique(campaign_seeds)) == length(campaign_seeds)

    # 4. Serial and threaded calls map the same preassigned tuples to identical seeds.
    serial_seeds = [make_particle_seed(t[1], t[2], t[3], t[4], t[5]) for t in tuples]
    threaded_results = Vector{Any}(undef, length(tuples))
    for (k, t) in enumerate(tuples)
        threaded_results[k] = Threads.@spawn make_particle_seed(t[1], t[2], t[3],
                                                                 t[4], t[5])
    end
    threaded_seeds = [fetch(r) for r in threaded_results]
    @test threaded_seeds == serial_seeds

    # 5. Identical evaluation inputs produce identical summaries (make_particle_seed
    #    is the seed source for run_sim_once; same tuple -> same RNG stream).
    sim_det = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ 0.5 * randn(rng),
        summary = identity,
    )
    cfg_det = ABCSMCConfig(
        θ -> logpdf(Normal(0.0, 5.0), θ[1]),
        rng -> [5.0 * randn(rng)],
        transform_identity, distance_absdiff,
        GaussianKernel(), Σ_i1, sim_det,
    )
    θ_same = [0.5]
    s1 = SequentialABC.run_sim_once(cfg_det, 2, 3, 1, θ_same; M=1, base_seed=base_seed)[1]
    s2 = SequentialABC.run_sim_once(cfg_det, 2, 3, 1, θ_same; M=1, base_seed=base_seed)[1]
    @test s1 == s2

    # Different attempt -> different summary (seed differs).
    s3 = SequentialABC.run_sim_once(cfg_det, 2, 3, 2, θ_same; M=1, base_seed=base_seed)[1]
    @test s1 != s3
end


@testset "Checkpoint-resume equivalence" begin
    # Compare uninterrupted and resumed runs across every completed generation.
    # Exact agreement in the current persisted state.
    base_seed = 20260209
    N_ck = 60
    T_ck = 4
    dir_full = mktempdir()
    schedule_ck = FixedSchedule([3.0, 1.5, 0.7, 0.4])

    # Uninterrupted run.
    state_full = make_state(cfg_gauss, Vector{Float64}, N_ck, 3.0, base_seed)
    run_smc!(state_full, S_OBS_1D; T = T_ck,
             schedule = schedule_ck, checkpoint_dir = dir_full)

    @test state_full.gen == T_ck

    # Same run stopped after each generation, then resumed to T_ck.
    for stop_gen in 1:(T_ck - 1)
        dir_part = mktempdir()   # fresh directory per stop_gen
        state_part = make_state(cfg_gauss, Vector{Float64}, N_ck, 3.0, base_seed)
        run_smc!(state_part, S_OBS_1D; T = stop_gen,
                 schedule = schedule_ck, checkpoint_dir = dir_part)

        state_resumed = load_latest_state(dir_part, cfg_gauss)
        @test state_resumed.gen == stop_gen
        run_smc!(state_resumed, S_OBS_1D; T = T_ck,
                 schedule = schedule_ck, checkpoint_dir = dir_part)

        @test state_resumed.gen == state_full.gen
        @test state_resumed.θ == state_full.θ
        @test state_resumed.w == state_full.w
        @test state_resumed.dists == state_full.dists
        @test state_resumed.ε == state_full.ε
        @test state_resumed.ess == state_full.ess
        @test state_resumed.n_attempts == state_full.n_attempts
        @test state_resumed.n_resamples == state_full.n_resamples
    end

    # Full checkpoint round-trip at the final generation.
    state_ft = load_latest_state(dir_full, cfg_gauss)
    @test state_ft.gen == state_full.gen
    @test state_ft.θ == state_full.θ
    @test state_ft.w == state_full.w
end

