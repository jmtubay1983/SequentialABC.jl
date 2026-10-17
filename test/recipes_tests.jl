# test/recipes_tests.jl
#
# Phase A3: the recipes module — recommended schedules, kernel scalings and
# summary presets. Everything here composes primitives that already have their
# own suites (schedules, adaptive kernels, summaries, distances); these tests
# pin the composition, the validation, and one end-to-end run per engine.

using Test
using Random
using Statistics
using LinearAlgebra
using Distributions

# Small deterministic 2-D reference problem, the same family as the other
# suites: θ ~ 𝒩(0, 2²I), s(θ) = θ + 𝒩(0, I), Euclidean distance. The summary is
# a preset; the distance is the merged `l2_distance`.
_prior_rec(θ) = sum(logpdf(Normal(0.0, 2.0), x) for x in θ)
_sample_prior_rec(rng) = 2.0 .* randn(rng, 2)
_preset_rec = preset_summaries(:vector; names = (:a, :b))
_simulator_rec = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = raw -> namedtuple_to_vector(compute_summary(_preset_rec, raw)),
)
OBS_REC = [1.0, -1.0]
SEED_REC = 20260119

# A pilot population for the kernel recipe, from its own stream so it cannot
# disturb the engine seeds.
_RNG_REC = MersenneTwister(20260119)
PILOT_REC = [_sample_prior_rec(_RNG_REC) for _ in 1:120]
CFG_REC = ABCSMCConfig(_prior_rec, _sample_prior_rec, identity, l2_distance,
                       GaussianKernel(), default_kernel_scaling(PILOT_REC),
                       _simulator_rec)

@testset "recipes" begin
    # ── default_schedule ─────────────────────────────────────────────────────
    @testset "adaptive engines and the geometric ladder" begin
        s_smc = default_schedule(:smc)
        @test s_smc isa QuantileSchedule
        @test s_smc.q == 0.5
        @test default_schedule(:pmc) isa QuantileSchedule
        @test default_schedule(:pmc).q == 0.5
        @test default_schedule(:smc; q = 0.9).q == 0.9
        @test_throws ArgumentError default_schedule(:smc; q = 0.0)
        @test_throws ArgumentError default_schedule(:smc; q = 1.5)

        # MCMC has no schedule; the error carries the calibration recipe
        err = try default_schedule(:mcmc); nothing catch e; e end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("fixed tolerance", msg)
        @test occursin("epsilon_quantile", msg)

        err = try default_schedule(:nope); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin(":nope", sprint(showerror, err))

        ladder = default_schedule(5; ε0 = 4.0, εT = 0.4)
        @test ladder isa FixedSchedule
        @test length(ladder.ε) == 5
        @test ladder.ε[1] == 4.0                     # endpoints exact
        @test ladder.ε[end] == 0.4
        @test issorted(ladder.ε; rev = true)
        @test all(diff(ladder.ε) .< 0)               # strictly tightening
        @test ladder.ε[2] ≈ 4.0 * 0.1^(1 / 4)        # geometric in between
        @test ladder.ε[3] ≈ 4.0 * 0.1^(2 / 4)
        @test default_schedule(2; ε0 = 2.0, εT = 1.0).ε == [2.0, 1.0]
        @test default_schedule(3; ε0 = 1.0, εT = 1.0).ε == ones(3)  # flat is fine

        @test_throws ArgumentError default_schedule(1; ε0 = 4.0, εT = 0.4)
        @test_throws ArgumentError default_schedule(5; ε0 = 0.4, εT = 4.0)
        @test_throws ArgumentError default_schedule(5; ε0 = 0.0, εT = 0.4)
        @test_throws ArgumentError default_schedule(5; ε0 = 4.0, εT = -0.1)
        @test_throws ArgumentError default_schedule(5; ε0 = 4.0, εT = Inf)
        @test_throws ArgumentError default_schedule(5; ε0 = NaN, εT = 1.0)
    end

    # ── default_kernel_scaling ───────────────────────────────────────────────
    @testset "engine rules from a unit prior and from a pilot" begin
        @test default_kernel_scaling(2; engine = :smc) ==
              2.0 .* Matrix{Float64}(I, 2, 2)
        @test default_kernel_scaling(2; engine = :pmc) ==
              2.0 .* Matrix{Float64}(I, 2, 2)
        @test default_kernel_scaling(2; engine = :mcmc) ≈
              (2.38^2 / 2) .* Matrix{Float64}(I, 2, 2)
        @test_throws ArgumentError default_kernel_scaling(0)
        @test_throws ArgumentError default_kernel_scaling(2; engine = :nope)

        Σ_smc = default_kernel_scaling(PILOT_REC; engine = :smc)
        @test size(Σ_smc) == (2, 2)
        @test Σ_smc ≈ 2.0 .* estimate_covariance(PILOT_REC)
        @test Σ_smc == adaptive_scale(estimate_covariance(PILOT_REC), 2;
                                      rule = :beaumont)
        @test Σ_smc ≈ Σ_smc'
        @test default_kernel_scaling(PILOT_REC; engine = :mcmc) ≈
              (2.38^2 / 2) .* estimate_covariance(PILOT_REC)

        # weights and shrink forward to estimate_covariance
        w = fill(1.0 / length(PILOT_REC), length(PILOT_REC))
        @test default_kernel_scaling(PILOT_REC; engine = :smc, weights = w) ≈ Σ_smc
        @test default_kernel_scaling(PILOT_REC; engine = :smc, shrink = 0.5) ≈
              2.0 .* estimate_covariance(PILOT_REC; shrink = 0.5)

        # a degenerate pilot stays positive definite (the shrink floors it)
        @test default_kernel_scaling([fill(1.0, 2), fill(1.0, 2)]; engine = :smc) ≈
              2e-6 .* Matrix{Float64}(I, 2, 2)
        @test_throws ArgumentError default_kernel_scaling(Vector{Float64}[])
    end

    # ── preset_summaries ─────────────────────────────────────────────────────
    @testset "preset_summaries: shapes, keys and values" begin
        # :scalar — one number under `name`
        sc = preset_summaries(:scalar)
        @test sc isa FunctionSummary
        @test summary_keys(sc) == [:value]
        @test compute_summary(sc, 3.5) == (value = 3.5,)
        @test summary_keys(preset_summaries(:scalar; name = :entropy)) == [:entropy]

        # :vector — one key per entry
        vc = preset_summaries(:vector; names = (:a, :b))
        @test vc isa CompositeSummary
        @test summary_keys(vc) == [:a, :b]
        @test compute_summary(vc, [10.0, 20.0]) == (a = 10.0, b = 20.0)
        @test namedtuple_to_vector(compute_summary(vc, [10.0, 20.0])) == [10.0, 20.0]
        vd = preset_summaries(:vector; d = 3)
        @test summary_keys(vd) == [:v1, :v2, :v3]
        @test compute_summary(vd, [7.0, 8.0, 9.0]) == (v1 = 7.0, v2 = 8.0, v3 = 9.0)
        @test summary_keys(preset_summaries(:vector; names = [:a, :b])) == [:a, :b]
        @test summary_keys(preset_summaries(:vector; names = (:a, :b), d = 2)) ==
              [:a, :b]

        # :moments — count/mean/sd over a collection
        mo = preset_summaries(:moments)
        @test summary_keys(mo) == [:count, :mean, :sd]
        nt = compute_summary(mo, [1.0, 2.0, 3.0, 4.0])
        @test nt.count == 4
        @test nt.mean == 2.5
        @test nt.sd ≈ std([1.0, 2.0, 3.0, 4.0])

        # :quantiles — named by percentage
        qu = preset_summaries(:quantiles)
        @test summary_keys(qu) == [:q10, :q50, :q90]
        v = [1.0, 2.0, 3.0, 4.0]
        qnt = compute_summary(qu, v)
        @test qnt.q10 ≈ quantile(v, 0.1)
        @test qnt.q50 == 2.5
        @test qnt.q90 ≈ quantile(v, 0.9)
        @test summary_keys(preset_summaries(:quantiles; probs = (0.25, 0.75))) ==
              [:q25, :q75]
        @test summary_keys(preset_summaries(:quantiles; probs = (0.975,))) == [:q97_5]
        @test summary_keys(preset_summaries(:quantiles; probs = (0.0, 0.05, 1.0))) ==
              [:q0, :q5, :q100]
        @test summary_keys(preset_summaries(:quantiles; probs = [0.5])) == [:q50]

        # `name` and the summaries framework carry through
        named = preset_summaries(:moments; name = :pop)
        @test named.name === :pop
        both = compose(vc, preset_summaries(:scalar; name = :extra))
        @test summary_keys(both) == [:a, :b, :extra]
        @test summary_keys(scale(named, Dict(:mean => 0.1))) == [:count, :mean, :sd]
    end

    @testset "preset_summaries: argument validation" begin
        err = try preset_summaries(:nope); nothing catch e; e end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin(":scalar", msg) && occursin(":quantiles", msg)

        @test_throws ArgumentError preset_summaries(:vector)             # names or d
        @test_throws ArgumentError preset_summaries(:vector; d = 0)
        @test_throws ArgumentError preset_summaries(:vector; names = Symbol[])
        @test_throws ArgumentError preset_summaries(:vector; names = (:a, :a))
        @test_throws ArgumentError preset_summaries(:vector; names = (:a, :b), d = 3)

        @test_throws ArgumentError preset_summaries(:scalar; names = (:a,))
        @test_throws ArgumentError preset_summaries(:scalar; probs = (0.5,))
        @test_throws ArgumentError preset_summaries(:moments; names = (:a,))
        @test_throws ArgumentError preset_summaries(:moments; d = 2)
        @test_throws ArgumentError preset_summaries(:moments; probs = (0.5,))
        @test_throws ArgumentError preset_summaries(:quantiles; names = (:a,))
        @test_throws ArgumentError preset_summaries(:quantiles; d = 2)

        @test_throws ArgumentError preset_summaries(:quantiles; probs = Float64[])
        @test_throws ArgumentError preset_summaries(:quantiles; probs = (0.9, 0.1))
        @test_throws ArgumentError preset_summaries(:quantiles; probs = (0.5, 0.5))
        @test_throws ArgumentError preset_summaries(:quantiles; probs = (-0.1,))
        @test_throws ArgumentError preset_summaries(:quantiles; probs = (1.5,))
        @test_throws ArgumentError preset_summaries(:quantiles; probs = (NaN,))
        @test_throws ArgumentError preset_summaries(:quantiles; probs = (Inf,))

        # the message names the offending keyword
        err = try preset_summaries(:moments; names = (:a,)); nothing catch e; e end
        @test occursin("names", sprint(showerror, err))
    end

    # ── integration ──────────────────────────────────────────────────────────
    @testset "recipes drive the engines end to end" begin
        # adaptive SMC: preset summary + l2_distance + recipe kernel and schedule
        st = run_inference(:smc, CFG_REC, OBS_REC; N = 40, T = 3,
                           schedule = default_schedule(:smc), base_seed = SEED_REC)
        @test st.gen == 3
        @test st.ε < Inf
        @test maximum(st.dists) <= st.ε
        @test all(θ -> all(isfinite, θ), st.θ)
        again = run_inference(:smc, CFG_REC, OBS_REC; N = 40, T = 3,
                              schedule = default_schedule(:smc), base_seed = SEED_REC)
        @test again.θ == st.θ

        # the geometric ladder is a prescribed schedule for both engines
        ladder = default_schedule(4; ε0 = 4.0, εT = 1.8)
        @test length(ladder.ε) == 4
        st_l = run_inference(:smc, CFG_REC, OBS_REC; N = 40, T = 4,
                             schedule = ladder, ε0 = 4.0, base_seed = SEED_REC)
        @test st_l.gen == 4
        @test st_l.ε == ladder.ε[4]
        @test maximum(st_l.dists) <= st_l.ε

        # PMC consumes the same ladder through its εs vector, which also sets ε₁
        st_p = run_inference(:pmc, CFG_REC, OBS_REC; N = 40, T = 4,
                             εs = ladder.ε, base_seed = SEED_REC)
        @test st_p.gen == 4
        @test st_p.ε == ladder.ε[4]
        @test maximum(st_p.dists) <= st_p.ε
        @test sum(st_p.w) ≈ 1.0

        # MCMC: calibrate ε from a short adaptive run — the :mcmc recipe
        st_cal = run_inference(:smc, CFG_REC, OBS_REC; N = 40, T = 3,
                               base_seed = SEED_REC)
        ε_cal = epsilon_quantile(st_cal, 0.5)
        @test 0 < ε_cal <= st_cal.ε
        chain = run_inference(:mcmc, CFG_REC, OBS_REC; n_samples = 20,
                              ε = ε_cal, base_seed = SEED_REC)
        @test length(chain) == 21
        @test all(θ -> all(isfinite, θ), chain.state.θ)

        # … and the :mcmc kernel recipe dropped into a config
        cfg_m = ABCSMCConfig(_prior_rec, _sample_prior_rec, identity, l2_distance,
                             GaussianKernel(),
                             default_kernel_scaling(PILOT_REC; engine = :mcmc),
                             _simulator_rec)
        chain2 = run_inference(:mcmc, cfg_m, OBS_REC; n_samples = 10, ε = 3.0,
                               base_seed = SEED_REC)
        @test length(chain2) == 11
    end
end
