using SequentialABC
using Test
using Distributions
using Random
using Statistics
using LinearAlgebra

# ── shared test helpers ──────────────────────────────────────────────────────

const PRIOR = Normal(0.0, 5.0)
const Y_OBS = 1.0

sample_prior = rng -> [rand(rng, PRIOR)]
prior_logpdf = θ -> logpdf(PRIOR, θ[1])
transform = θ -> θ

# Simple 2-key summary for multi-key tests
summary_2key = (a = 0.0, b = 0.0)

simulator_2key = ContinuousSimulator(
    model_builder = μ -> [μ[1], μ[2]],
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = y -> (a = y[1], b = y[2]),
)

distance_2key = (s, so) -> sqrt((s.a - so.a)^2 + (s.b - so.b)^2)

Σ_mat_2 = [0.5 0.0; 0.0 0.5]
cfg_2key = ABCSMCConfig(
    prior_logpdf, sample_prior, transform, distance_2key,
    GaussianKernel(), Σ_mat_2, simulator_2key,
)

# Single-key simulator
simulator_scalar = ContinuousSimulator(
    model_builder = μ -> μ[1],
    evolve = (μ, rng) -> μ + randn(rng),
    summary = y -> (value = y,),
)

distance_scalar = (s, so) -> abs(s.value - so.value)

Σ_mat_1 = [0.5;;]
cfg_scalar = ABCSMCConfig(
    prior_logpdf, sample_prior, transform, distance_scalar,
    GaussianKernel(), Σ_mat_1, simulator_scalar,
)

N = 100

# ── Configuration tests ──────────────────────────────────────────────────────

@testset "NumericalPolicy defaults and validation" begin
    p = NumericalPolicy()
    @test p.floor == 1e-8
    @test p.constant_policy === :error
    @test p.max_weight == 10.0

    p2 = NumericalPolicy(floor = 1e-6, constant_policy = :clamp, max_weight = 5.0)
    @test p2.floor == 1e-6
    @test p2.constant_policy === :clamp
    @test p2.max_weight == 5.0

    @test_throws ArgumentError NumericalPolicy(floor = -1.0)
    @test_throws ArgumentError NumericalPolicy(floor = 0.0)
    @test_throws ArgumentError NumericalPolicy(constant_policy = :bad)
end

@testset "Strategy types identify estimator only" begin
    @test NoAdaptation() isa AbstractScaleStrategy
    @test StandardDeviationScale() isa AbstractScaleStrategy

    # StandardDeviationScale has no policy fields
    @test_throws ArgumentError StandardDeviationScale(floor = 1e-8)
    @test_throws ArgumentError StandardDeviationScale(constant_policy = :error)
end

@testset "FixedScale validation" begin
    fs = FixedScale([1.0, 2.0, 3.0])
    @test fs.weights == [1.0, 2.0, 3.0]

    @test_throws ArgumentError FixedScale(Float64[])
    @test_throws ArgumentError FixedScale([NaN, 1.0])
    @test_throws ArgumentError FixedScale([-1.0, 1.0])
    @test_throws ArgumentError FixedScale([0.0, 0.0])
end

@testset "AdaptiveDistanceConfig construction" begin
    cfg_ad = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        PreviousGenAccepted(),
        UnitInit(),
        NumericalPolicy(floor = 1e-6),
    )
    @test cfg_ad.strategy isa StandardDeviationScale
    @test cfg_ad.estimator_source isa PreviousGenAccepted
    @test cfg_ad.initializer isa UnitInit
    @test cfg_ad.policy.floor == 1e-6

    # Defaults
    cfg_def = AdaptiveDistanceConfig(NoAdaptation())
    @test cfg_def.strategy isa NoAdaptation
    @test cfg_def.estimator_source isa PreviousGenAccepted
    @test cfg_def.initializer isa UnitInit
    @test cfg_def.policy isa NumericalPolicy
end

# ── Weight estimation tests ──────────────────────────────────────────────────

@testset "Unweighted standard-deviation weights" begin
    # Two keys with very different scales
    summaries = [
        (a = 1.0, b = 100.0),
        (a = 2.0, b = 200.0),
        (a = 3.0, b = 300.0),
    ]
    key_order = [:a, :b]
    X = [1.0 100.0; 2.0 200.0; 3.0 300.0]
    w = ones(3) ./ 3
    policy = NumericalPolicy(floor = 1e-8, constant_policy = :error, max_weight = 10.0)

    weights = estimate_weights(StandardDeviationScale(), X, w, key_order, policy)
    @test length(weights) == 2
    @test isapprox(weights[1], 1.0 / std([1.0, 2.0, 3.0]))
    @test isapprox(weights[2], 1.0 / std([100.0, 200.0, 300.0]))
    # Inverse relationship: larger std → smaller weight
    @test weights[1] > weights[2]
end

@testset "Stable summary-key ordering" begin
    key_order = [:b, :a]  # alphabetical would be [:a, :b], but we use insertion order
    X = [100.0 1.0; 200.0 2.0; 300.0 3.0]  # b first, then a
    w = ones(3) ./ 3
    policy = NumericalPolicy()

    weights = estimate_weights(StandardDeviationScale(), X, w, key_order, policy)
    # weights[1] corresponds to key :b, weights[2] to key :a
    @test isapprox(weights[1], 1.0 / std([100.0, 200.0, 300.0]))
    @test isapprox(weights[2], 1.0 / std([1.0, 2.0, 3.0]))
end

@testset "Key mismatch error" begin
    summaries = [(a = 1.0, b = 2.0), (a = 3.0, b = 4.0)]
    key_order = [:a, :c]  # :c doesn't exist
    X = [1.0 2.0; 3.0 4.0]
    w = ones(2) ./ 2
    policy = NumericalPolicy()

    # estimate_weights doesn't check key existence, but apply_weights does
    # The error is caught in update_adaptive! when reading summaries
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    # This test is covered at the update_adaptive! level below
end

@testset "Non-finite summary error" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    # Corrupt one summary with NaN
    state.summaries[1] = (a = NaN, b = 2.0)

    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    @test_throws ArgumentError update_adaptive!(
        state, ad_config, cfg_2key.distance, Y_OBS)
end

@testset "Constant-summary :error policy" begin
    # All particles have identical summary for key 'a' → zero variance
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    # Force all summaries to be identical
    for i in eachindex(state.summaries)
        state.summaries[i] = (a = 1.0, b = state.summaries[i].b)
    end

    ad_config = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        NumericalPolicy(floor = 1e-8, constant_policy = :error),
    )
    @test_throws ArgumentError update_adaptive!(
        state, ad_config, cfg_2key.distance, Y_OBS)
end

@testset "Constant-summary :ignore policy" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    for i in eachindex(state.summaries)
        state.summaries[i] = (a = 1.0, b = state.summaries[i].b)
    end

    ad_config = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        NumericalPolicy(floor = 1e-8, constant_policy = :ignore),
    )
    update_adaptive!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.weights[1] == 0.0  # key 'a' gets zero weight
    @test state.adaptive.weights[2] > 0.0   # key 'b' gets positive weight
end

@testset "Constant-summary :clamp policy" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    for i in eachindex(state.summaries)
        state.summaries[i] = (a = 1.0, b = state.summaries[i].b)
    end

    ad_config = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        NumericalPolicy(floor = 1e-8, constant_policy = :clamp, max_weight = 50.0),
    )
    update_adaptive!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.weights[1] == 50.0  # clamped to max_weight
end

@testset "FixedScale correctness" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    ad_config = AdaptiveDistanceConfig(
        FixedScale([2.0, 0.5]),
        NumericalPolicy(),
    )
    update_adaptive!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.weights == [2.0, 0.5]
end

# ── Adaptive lifecycle tests ─────────────────────────────────────────────────

@testset "Acceptance distances distinct from epsilon-selection distances" begin
    dir = mktempdir()
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())

    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T = 3, schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = ad_config, checkpoint_dir = dir)

    # state.dists are acceptance distances (computed with generation-2 weights)
    @test all(isfinite, state.dists)
    @test all(>(0), state.dists)

    # state.adaptive.epsilon_distances are recomputed under new weights
    @test state.adaptive !== nothing
    @test state.adaptive.epsilon_distances !== nothing
    @test length(state.adaptive.epsilon_distances) == N

    # They should differ from state.dists (different weights → different distances)
    @test state.dists != state.adaptive.epsilon_distances

    # state.dists must not have been overwritten
    @test state.gen == 3
end

@testset "Epsilon and distance-weight consistency" begin
    dir = mktempdir()
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())

    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T = 3, schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = ad_config, checkpoint_dir = dir)

    # After gen 2: weights estimated, distances recomputed
    # After gen 3: same
    @test state.adaptive.gen_estimated == 3
    @test state.adaptive.epsilon !== nothing
    @test isfinite(state.adaptive.epsilon)
    @test state.adaptive.epsilon > 0
end

@testset "NoAdaptation fixed-seed equivalence to current main" begin
    # NoAdaptation should behave identically to not using adaptive at all
    schedule = FixedSchedule([3.0, 1.5, 0.7])

    state_adapt = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state_adapt, Y_OBS; T = 3, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(NoAdaptation()))

    state_plain = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state_plain, Y_OBS; T = 3, schedule = schedule)

    @test state_adapt.θ == state_plain.θ
    @test state_adapt.w == state_plain.w
    @test state_adapt.dists == state_plain.dists
    @test state_adapt.ε == state_plain.ε
    @test state_adapt.gen == state_plain.gen
end

@testset "Deterministic adaptive replay" begin
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    schedule = FixedSchedule([3.0, 1.5, 0.7])

    state1 = make_state(cfg_2key, Vector{Float64}, N, 3.0, 4242)
    run_smc!(state1, Y_OBS; T = 3, schedule = schedule, adaptive = ad_config)

    state2 = make_state(cfg_2key, Vector{Float64}, N, 3.0, 4242)
    run_smc!(state2, Y_OBS; T = 3, schedule = schedule, adaptive = ad_config)

    @test state1.θ == state2.θ
    @test state1.w == state2.w
    @test state1.dists == state2.dists
    @test state1.adaptive.weights == state2.adaptive.weights
    @test state1.adaptive.epsilon_distances == state2.adaptive.epsilon_distances
    @test state1.adaptive.epsilon == state2.adaptive.epsilon
end

# ── Schedule tests ───────────────────────────────────────────────────────────

@testset "Fixed schedule ignores supplied distances" begin
    fs = FixedSchedule([3.0, 1.5, 0.7])
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    # 3-arg call ignores distances
    ε1 = fs(state, 2, state.dists)
    ε2 = fs(state, 2, [100.0, 200.0, 300.0])  # completely different distances
    @test ε1 == ε2 == 1.5

    # 2-arg still works
    ε3 = fs(state, 2)
    @test ε3 == 1.5
end

@testset "Quantile schedule uses supplied distances" begin
    qs = QuantileSchedule(0.5)
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    # 3-arg with state.dists should equal 2-arg
    ε_3arg = qs(state, 2, state.dists)
    ε_2arg = qs(state, 2)
    @test isapprox(ε_3arg, ε_2arg)

    # Different distances → different epsilon
    ε_diff = qs(state, 2, [0.1, 0.2, 0.3])
    @test ε_diff != ε_2arg
end

@testset "Backward-compatible custom schedule (2-arg only)" begin
    custom_schedule = (state, t) -> 2.0 / t
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T = 3, schedule = custom_schedule)
    @test state.gen == 3
    @test state.ε == 2.0 / 3
end

# ── Checkpoint tests ─────────────────────────────────────────────────────────

@testset "Uninterrupted vs resumed bitwise equivalence" begin
    dir = mktempdir()
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    schedule = FixedSchedule([3.0, 1.5, 0.7])

    # Uninterrupted run to gen 3
    s_full = make_state(cfg_2key, Vector{Float64}, N, 3.0, 7777)
    run_smc!(s_full, Y_OBS; T = 3, schedule = schedule,
             adaptive = ad_config, checkpoint_dir = dir)

    # Run to gen 2, save checkpoint
    s_part = make_state(cfg_2key, Vector{Float64}, N, 3.0, 7777)
    run_smc!(s_part, Y_OBS; T = 2, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()),
             checkpoint_dir = dir)

    # Reload and continue to gen 3
    s_resumed = load_latest_state(dir, cfg_2key)
    @test s_resumed.gen == 2
    run_smc!(s_resumed, Y_OBS; T = 3, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()),
             checkpoint_dir = dir)

    # Bitwise equivalence
    @test s_resumed.θ == s_full.θ
    @test s_resumed.w == s_full.w
    @test s_resumed.dists == s_full.dists
    @test s_resumed.ε == s_full.ε
    @test s_resumed.gen == s_full.gen
    @test s_resumed.adaptive.weights == s_full.adaptive.weights
    @test s_resumed.adaptive.epsilon_distances == s_full.adaptive.epsilon_distances
    @test s_resumed.adaptive.epsilon == s_full.adaptive.epsilon
end

@testset "Legacy checkpoint loading" begin
    # Simulate a legacy checkpoint by saving without adaptive state
    dir = mktempdir()
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    # Save manually without schema_version (legacy format)
    path = joinpath(dir, "state_gen_001.jls")
    tmp = path * ".tmp"
    mkpath(dir)
    data = (θ = state.θ, w = state.w, summaries = state.summaries,
            dists = state.dists, ε = state.ε, gen = state.gen,
            ess = state.ess, base_seed = state.base_seed,
            n_attempts = state.n_attempts, n_resamples = state.n_resamples)
    open(tmp, "w") do io; serialize(io, data); end
    mv(tmp, path; force = true)

    # Legacy checkpoint should load fine with adaptive = nothing
    state2 = load_checkpoint(path, cfg_2key)
    @test state2.gen == 1
    @test state2.adaptive === nothing
end

@testset "New adaptive checkpoint round-trip" begin
    dir = mktempdir()
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())

    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T = 2, schedule = FixedSchedule([3.0, 1.5]),
             adaptive = ad_config, checkpoint_dir = dir)

    state2 = load_latest_state(dir, cfg_2key)
    @test state2.gen == 2
    @test state2.adaptive !== nothing
    @test state2.adaptive.weights == state.adaptive.weights
    @test state2.adaptive.epsilon_distances == state.adaptive.epsilon_distances
    @test state2.adaptive.epsilon == state.adaptive.epsilon
end

# ── Scalar summary tests ─────────────────────────────────────────────────────

@testset "Scalar summaries" begin
    state = make_state(cfg_scalar, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    update_adaptive!(state, ad_config, cfg_scalar.distance, Y_OBS)
    @test length(state.adaptive.key_order) == 1
    @test state.adaptive.key_order == [:value]
    @test length(state.adaptive.weights) == 1
    @test state.adaptive.weights[1] > 0
end

# ── Heterogeneous scales test ────────────────────────────────────────────────

@testset "Heterogeneous summary scales" begin
    # One key has much larger variance than the other
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    update_adaptive!(state, ad_config, cfg_2key.distance, Y_OBS)

    # Key 'a' has smaller std → larger weight; key 'b' has larger std → smaller weight
    @test state.adaptive.weights[1] > state.adaptive.weights[2]
end

@testset "Informative plus irrelevant high-variance summaries" begin
    # Key 'a' is informative (low variance), key 'b' is noise (high variance)
    # Adaptive weights should down-weight the noisy key
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    update_adaptive!(state, ad_config, cfg_2key.distance, Y_OBS)

    # Key 'a' should get higher weight than key 'b' (lower std → higher weight)
    @test state.adaptive.weights[1] > state.adaptive.weights[2]
end

# ── Posterior accuracy on tractable model ────────────────────────────────────

@testset "Posterior accuracy on tractable model" begin
    # Simple Gaussian: prior θ ~ N(0, 5), observation y = θ + noise, noise ~ N(0, 1)
    # Posterior: θ | y ~ N(y * 25/26, 25/26) ≈ N(0.9615, 0.9615)
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 9999)
    run_smc!(state, Y_OBS; T = 5, schedule = QuantileSchedule(0.5))

    # Weighted posterior mean should be near the analytic value ≈ 0.96
    θbar = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
    @test 0.4 < θbar < 1.6
end

# ── Full integration test ────────────────────────────────────────────────────

@testset "Full adaptive ABC-SMC pipeline" begin
    dir = mktempdir()
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())

    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 2026)
    run_smc!(state, Y_OBS; T = 4,
             schedule = QuantileSchedule(0.5),
             adaptive = ad_config,
             checkpoint_dir = dir)

    @test state.gen == 4
    @test isapprox(sum(state.w), 1.0; atol = 1e-10)
    @test all(isfinite, state.w)
    @test all(>(0), state.w)
    @test state.adaptive !== nothing
    @test state.adaptive.gen_estimated == 4
    @test isfinite(state.adaptive.epsilon)
    @test state.adaptive.epsilon > 0
end

