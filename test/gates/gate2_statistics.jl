# test/gates/gate2_statistics.jl
# Gate 2 — Statistical correctness.
#
# Analytic / synthetic targets with fixed seeds and prespecified tolerances.
# Independent weight oracle compared to production `importance_weight`.
#
# Run via: julia --project=. test/gates/run_gates.jl
# Standalone: include("test/gates/gate_common.jl"); run_gate2()

using LogExpFunctions

struct TestResult
    name::String
    passed::Bool
    detail::String
end

function run_gate2()
    r = GateResult("Gate 2 — statistical correctness")
    base_seed = 20260201

    # ------------------------------------------------------------------
    # 2a. 1D Gaussian posterior: θ ~ N(0, 5²), s = θ + N(0, 1), s_obs = 1.0
    #     Analytic posterior: mean = 1/1.04 ≈ 0.96154, var = 1/1.04 ≈ 0.96154
    # ------------------------------------------------------------------
    dim = 1
    prior_sd = 5.0
    noise = 1.0
    s_obs = [1.0]

    _prior_logpdf(θ) = sum(logpdf(Normal(0.0, prior_sd), x) for x in θ)
    _sample_prior(rng) = prior_sd .* randn(rng, dim)
    _distance(s, so) = sqrt(sum(abs2, s .- so))
    _transform(φ) = φ  # identity: raw == model for this test
    Σ = Matrix{Float64}(I, dim, dim)
    sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> θ .+ noise .* randn(rng, dim),
        summary = identity,
    )
    cfg_gauss = ABCSMCConfig(_prior_logpdf, _sample_prior, _transform,
                             _distance, GaussianKernel(), Σ, sim)

    state = run_smc(cfg_gauss, s_obs; N=100, T=4, ε0=Inf, base_seed=base_seed,
                    schedule=QuantileSchedule(0.5))

    # Weighted posterior mean
    w_norm = state.w ./ sum(state.w)
    mean_est = sum(w_norm .* [θ[1] for θ in state.θ])
    mean_analytic = 1.0 / 1.04   # (0/25 + 1/1) / (1/25 + 1/1)
    tol_mean = 0.35
    check!(r, "1D Gaussian posterior mean within tolerance",
           abs(mean_est - mean_analytic) < tol_mean;
           detail="est=$(round(mean_est, digits=4)) analytic=$(round(mean_analytic, digits=4)) tol=$tol_mean")

    # Weighted sd sanity (loose)
    var_est = sum(w_norm .* ([θ[1] for θ in state.θ] .- mean_est).^2)
    sd_est = sqrt(max(var_est, 0.0))
    sd_analytic = sqrt(1.0 / 1.04)
    check!(r, "1D Gaussian posterior sd within tolerance",
           abs(sd_est - sd_analytic) < 0.6;
           detail="est=$(round(sd_est, digits=4)) analytic=$(round(sd_analytic, digits=4))")

    # ------------------------------------------------------------------
    # 2b. Correlated 2D: :diag vs :full bandwidth semantics
    #     Σ = [4 1.8; 1.8 9], corr = 0.3
    #     :diag → [2.0, 3.0]; :full → [2.0, sqrt(9 - 1.8^2/4)] ≈ [2.0, 2.862]
    # ------------------------------------------------------------------
    Σ2 = [4.0 1.8; 1.8 9.0]
    bw_diag = bandwidth_from_cov(Σ2; method=:diag)
    bw_full = bandwidth_from_cov(Σ2; method=:full)

    check!(r, ":diag bandwidth[1] == 2.0", isapprox(bw_diag[1], 2.0; atol=1e-12);
           detail="bw_diag=$(bw_diag)")
    check!(r, ":diag bandwidth[2] == 3.0", isapprox(bw_diag[2], 3.0; atol=1e-12);
           detail="bw_diag=$(bw_diag)")
    check!(r, ":full bandwidth[1] == :diag bandwidth[1]",
           isapprox(bw_full[1], bw_diag[1]; atol=1e-12);
           detail="bw_full=$(bw_full) bw_diag=$(bw_diag)")
    check!(r, ":full bandwidth[2] < :diag bandwidth[2] (correlation shrinks conditional sd)",
           bw_full[2] < bw_diag[2];
           detail="full[2]=$(bw_full[2]) diag[2]=$(bw_diag[2])")

    # make_adaptive_kernel respects :diag vs :full
    cfg2 = _std_config(; dim=2, noise=0.1, prior_sd=5.0, transform=identity)
    ker_diag = make_adaptive_kernel(cfg2; method=:diag, min_scale=1e-6)
    ker_full = make_adaptive_kernel(cfg2; method=:full, min_scale=1e-6)
    # Correlated 2-D population so :full differs from :diag
    rng_pop = MersenneTwister(7)
    z = randn(rng_pop, 10)
    pop = [[z[i], 0.9 * z[i] + 0.1 * randn(rng_pop)] for i in 1:10]
    bw_kd = ker_diag(pop)
    bw_kf = ker_full(pop)
    check!(r, "make_adaptive_kernel :diag vs :full give different scales on correlated pop",
           !isapprox(bw_kd, bw_kf; atol=1e-10);
           detail="diag=$(bw_kd) full=$(bw_kf)")
    check!(r, "make_adaptive_kernel output length == dim", length(bw_kd) == 2;
           detail="length=$(length(bw_kd))")

    # ------------------------------------------------------------------
    # 2c. Bimodal target: θ ~ N(0, 4²), s = θ² + N(0, 0.1), s_obs = 4.0
    #     Posterior mass near θ = ±2 → two modes. Require min mass per mode.
    # ------------------------------------------------------------------
    prior_sd_bi = 4.0
    noise_bi = 0.1
    s_obs_bi = [4.0]

    _prior_bi(θ) = sum(logpdf(Normal(0.0, prior_sd_bi), x) for x in θ)
    _sample_bi(rng) = prior_sd_bi .* randn(rng, 1)
    _dist_bi(s, so) = abs(s[1] - so[1])
    sim_bi = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (θ, rng) -> [θ[1]^2 + noise_bi * randn(rng)],
        summary = identity,
    )
    cfg_bi = ABCSMCConfig(_prior_bi, _sample_bi, identity, _dist_bi,
                          GaussianKernel(), Matrix{Float64}(I,1,1), sim_bi)

    state_bi = run_smc(cfg_bi, s_obs_bi; N=120, T=5, ε0=Inf, base_seed=base_seed+1,
                       schedule=QuantileSchedule(0.3))

    θ_vals = [θ[1] for θ in state_bi.θ]
    w_bi = state_bi.w ./ sum(state_bi.w)
    mass_neg = sum(w_bi[θ_vals .< 0])
    mass_pos = sum(w_bi[θ_vals .> 0])
    min_mode_mass = 0.10
    check!(r, "Bimodal: negative mode mass ≥ threshold",
           mass_neg >= min_mode_mass; detail="mass_neg=$(round(mass_neg, digits=3))")
    check!(r, "Bimodal: positive mode mass ≥ threshold",
           mass_pos >= min_mode_mass; detail="mass_pos=$(round(mass_pos, digits=3))")

    # ------------------------------------------------------------------
    # 2d. Log-transformed parameter: φ ~ N(log(2), 0.5²), θ = exp(φ), s = θ + noise
    #     Posterior should concentrate near θ ≈ 2.0 (φ ≈ log(2))
    # ------------------------------------------------------------------
    true_theta = 2.0
    mu_phi = log(true_theta)
    sd_phi = 0.5
    noise_lt = 0.1
    s_obs_lt = [true_theta]

    _prior_lt(φ) = sum(logpdf(Normal(mu_phi, sd_phi), x) for x in φ)
    _sample_lt(rng) = mu_phi .+ sd_phi .* randn(rng, 1)
    _dist_lt(s, so) = abs(s[1] - so[1])
    _exp_transform(φ) = exp.(φ)
    sim_lt = ContinuousSimulator(
        model_builder = φ -> φ,
        evolve = (θ, rng) -> θ .+ noise_lt .* randn(rng, 1),
        summary = identity,
    )
    cfg_lt = ABCSMCConfig(_prior_lt, _sample_lt, _exp_transform, _dist_lt,
                          GaussianKernel(), Matrix{Float64}(I,1,1), sim_lt)

    state_lt = run_smc(cfg_lt, s_obs_lt; N=100, T=5, ε0=Inf, base_seed=base_seed+2,
                       schedule=QuantileSchedule(0.3))

    phi_vals = [φ[1] for φ in state_lt.θ]
    w_lt = state_lt.w ./ sum(state_lt.w)
    theta_est = sum(w_lt .* exp.(phi_vals))
    tol_lt = 0.4
    check!(r, "Log-transformed: posterior mean of θ near true value",
           abs(theta_est - true_theta) < tol_lt;
           detail="est=$(round(theta_est, digits=4)) true=$true_theta tol=$tol_lt")

    # ------------------------------------------------------------------
    # 2e. Independent weight oracle vs production importance_weight
    #     Toni: w ∝ π(θ_prop) / Σ_j w_prev[j] · K(θ_prop | θ_prev[j], Σ)
    #     Edge cases: zero weights (denominator terms), duplicated ancestors
    # ------------------------------------------------------------------
    function weight_oracle(prior_logpdf, θ_prop, θ_prev, w_prev, Σ_inv, logdetΣ)
        k = length(θ_prop)
        logprior = prior_logpdf(θ_prop)
        terms = Float64[]
        for j in eachindex(θ_prev)
            wj = w_prev[j]
            if wj == 0.0
                push!(terms, -Inf)   # log(0) = -Inf → excluded from log-sum-exp
                continue
            end
            diff = θ_prop .- θ_prev[j]
            logK = -0.5 * (diff' * Σ_inv * diff) - 0.5 * (k * log(2π) + logdetΣ)
            push!(terms, log(wj) + logK)
        end
        logdenom = logsumexp(terms)
        return exp(logprior - logdenom)
    end

    _kernel_logpdf(θ_prop, θ_prev, Σ_inv, logdetΣ) =
        -0.5 * ((θ_prop .- θ_prev)' * Σ_inv * (θ_prop .- θ_prev)) -
        0.5 * (length(θ_prop) * log(2π) + logdetΣ)

    Σ_inv = inv(Σ)
    logdetΣ = logdet(Σ)
    θ_prop = [0.5]

    # Case 1: normal, all weights positive
    θ_prev1 = [[-0.3], [0.1], [0.8]]
    w_prev1 = [0.2, 0.5, 0.3]
    w_prod1 = importance_weight(cfg_gauss, θ_prop, θ_prev1, w_prev1)
    w_oracle1 = weight_oracle(_prior_logpdf, θ_prop, θ_prev1, w_prev1, Σ_inv, logdetΣ)
    check!(r, "Weight oracle matches production (normal case)",
           isapprox(w_prod1, w_oracle1; rtol=1e-10);
           detail="prod=$(w_prod1) oracle=$(w_oracle1)")

    # Case 2: duplicated ancestors (repeated θ values)
    θ_prev2 = [[0.1], [0.1], [0.8]]  # first two duplicated
    w_prev2 = [0.3, 0.2, 0.5]
    w_prod2 = importance_weight(cfg_gauss, θ_prop, θ_prev2, w_prev2)
    w_oracle2 = weight_oracle(_prior_logpdf, θ_prop, θ_prev2, w_prev2, Σ_inv, logdetΣ)
    check!(r, "Weight oracle matches production (duplicated ancestors)",
           isapprox(w_prod2, w_oracle2; rtol=1e-10);
           detail="prod=$(w_prod2) oracle=$(w_oracle2)")

    # Case 3: zero-weight ancestor (denominator term excluded)
    θ_prev3 = [[-0.3], [0.1], [0.8]]
    w_prev3 = [0.0, 0.5, 0.5]   # first weight is zero
    w_prod3 = importance_weight(cfg_gauss, θ_prop, θ_prev3, w_prev3)
    w_oracle3 = weight_oracle(_prior_logpdf, θ_prop, θ_prev3, w_prev3, Σ_inv, logdetΣ)
    check!(r, "Weight oracle matches production (zero-weight ancestor)",
           isapprox(w_prod3, w_oracle3; rtol=1e-10);
           detail="prod=$(w_prod3) oracle=$(w_oracle3)")

    # Case 4: all weights zero → denominator is -Inf → weight = Inf (degenerate)
    θ_prev4 = θ_prev3
    w_prev4 = [0.0, 0.0, 0.0]
    w_prod4 = importance_weight(cfg_gauss, θ_prop, θ_prev4, w_prev4)
    check!(r, "Zero denominator → production returns Inf (degenerate, documented)",
           isinf(w_prod4) && w_prod4 > 0;
           detail="w=$(w_prod4)")

    return r
end
