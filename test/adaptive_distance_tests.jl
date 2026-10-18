using SequentialABC
using Test
using Distributions
using Random
using Statistics
using LinearAlgebra
using Serialization

# ── shared test helpers ──────────────────────────────────────────────────────
# Plain globals (no `const`): the test suite shares Main and earlier files
# (test_checkpoint.jl, test_smc.jl) already declared these names as plain
# globals — a later `const` redeclaration errors at load time.

PRIOR = Normal(0.0, 5.0)
Y_OBS = (a = 1.0, b = 0.0)      # observed summary for the 2-key model
SCALAR_OBS = (value = 1.0,)   # observed summary for the scalar model

# Scalar model: θ ~ N(0, 5), simulated summary = θ + N(0, 1).
# Posterior for θ given summary ≈ 1: N(25/26, 25/26), mean ≈ 0.9615.
sample_prior = rng -> [rand(rng, PRIOR)]
prior_logpdf = θ -> logpdf(PRIOR, θ[1])
transform = θ -> θ

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

# 2-key model with deliberately heterogeneous summary scales:
#   key :a — informative, low variance   (θ₁ ~ N(0, 2.5) + N(0,1) noise)
#   key :b — irrelevant, high variance   (θ₂ ~ N(0, 25)  + N(0,1) noise)
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
Σ_mat_2 = [0.5 0.0; 0.0 0.5]
cfg_2key = ABCSMCConfig(
    prior_logpdf_2key, sample_prior_2key, transform, distance_2key,
    GaussianKernel(), Σ_mat_2, simulator_2key,
)

N = 100

# Internal helpers under test — AdaptiveDistanceState and the lifecycle
# helpers stay unexported (constraint: configuration types only).
const estimate_w = SequentialABC.estimate_weights
const update_ad! = SequentialABC.update_adaptive!

# ── Configuration tests ──────────────────────────────────────────────────────

@testset "NumericalPolicy defaults and validation" begin
    p = NumericalPolicy()
    @test p.floor == 1e-8
    @test p.constant_policy === :error
    @test p.max_weight == 1.0e6

    p2 = NumericalPolicy(floor = 1e-6, constant_policy = :clamp, max_weight = 5.0)
    @test p2.floor == 1e-6
    @test p2.constant_policy === :clamp
    @test p2.max_weight == 5.0

    @test_throws ArgumentError NumericalPolicy(floor = -1.0)
    @test_throws ArgumentError NumericalPolicy(floor = 0.0)
    @test_throws ArgumentError NumericalPolicy(constant_policy = :bad)
    @test_throws ArgumentError NumericalPolicy(max_weight = 0.0)
end

@testset "Strategy types identify estimator only" begin
    @test NoAdaptation() isa AbstractScaleStrategy
    @test StandardDeviationScale() isa AbstractScaleStrategy
    @test FixedScale([1.0]) isa AbstractScaleStrategy

    # Strategy types carry no policy fields — NumericalPolicy is the single
    # source of truth (passing policy kwargs is rejected).
    @test_throws ArgumentError StandardDeviationScale(floor = 1e-8)
    @test_throws ArgumentError StandardDeviationScale(constant_policy = :error)
    @test_throws ArgumentError NoAdaptation(max_weight = 5.0)
    @test !hasproperty(StandardDeviationScale(), :floor)
    @test !hasproperty(StandardDeviationScale(), :constant_policy)
    @test !hasproperty(StandardDeviationScale(), :max_weight)
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

    # Defaults: previous-gen estimator, unit initializer, default policy.
    cfg_def = AdaptiveDistanceConfig(NoAdaptation())
    @test cfg_def.strategy isa NoAdaptation
    @test cfg_def.estimator_source isa PreviousGenAccepted
    @test cfg_def.initializer isa UnitInit
    @test cfg_def.policy isa NumericalPolicy

    # (strategy, policy) convenience.
    cfg_pol = AdaptiveDistanceConfig(
        StandardDeviationScale(), NumericalPolicy(constant_policy = :clamp))
    @test cfg_pol.policy.constant_policy === :clamp
    @test cfg_pol.estimator_source isa PreviousGenAccepted
    @test cfg_pol.initializer isa UnitInit

    # Out-of-scope estimator sources / initializers are rejected.
    @test_throws ArgumentError AdaptiveDistanceConfig(
        StandardDeviationScale(), :all_simulations, UnitInit(), NumericalPolicy())
    @test_throws ArgumentError AdaptiveDistanceConfig(
        StandardDeviationScale(), PreviousGenAccepted(), :prior_predictive,
        NumericalPolicy())
end

# ── Weight estimation tests ──────────────────────────────────────────────────

@testset "Unweighted standard-deviation weights" begin
    X = [1.0 100.0; 2.0 200.0; 3.0 300.0]
    key_order = [:a, :b]
    w = ones(3) ./ 3
    policy = NumericalPolicy(floor = 1e-8, constant_policy = :error, max_weight = 10.0)

    weights = estimate_w(StandardDeviationScale(), X, w, key_order, policy)
    @test length(weights) == 2
    @test isapprox(weights[1], 1.0 / std([1.0, 2.0, 3.0]))
    @test isapprox(weights[2], 1.0 / std([100.0, 200.0, 300.0]))
    # Inverse relationship: larger std → smaller weight.
    @test weights[1] > weights[2]

    # NoAdaptation: all unit weights regardless of scales.
    w_none = estimate_w(NoAdaptation(), X, w, key_order, policy)
    @test w_none == ones(2)
end

@testset "Stable summary-key ordering" begin
    # Columns follow the declared key order, not alphabetical order.
    key_order = [:b, :a]
    X = [100.0 1.0; 200.0 2.0; 300.0 3.0]
    w = ones(3) ./ 3
    weights = estimate_w(StandardDeviationScale(), X, w, key_order, NumericalPolicy())
    @test isapprox(weights[1], 1.0 / std([100.0, 200.0, 300.0]))
    @test isapprox(weights[2], 1.0 / std([1.0, 2.0, 3.0]))

    # A run pins key order to the observed summary's declaration order.
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    update_ad!(state, AdaptiveDistanceConfig(StandardDeviationScale()),
               cfg_2key.distance, Y_OBS)
    @test state.adaptive.key_order == [:a, :b]
    @test length(state.adaptive.weights) == 2
end

@testset "Key mismatch error" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.key_order == [:a, :b]

    # Drop :b from one accepted summary → mismatch with the frozen key order.
    state.summaries[1] = (a = 1.0,)
    @test_throws ArgumentError update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
end

@testset "Non-finite summary error" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    state.summaries[1] = (a = NaN, b = 2.0)

    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    @test_throws ArgumentError update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
end

# ── Constant-summary policies ────────────────────────────────────────────────

# Force key :a to be constant across the accepted population.
function force_constant_a!(state)
    for i in eachindex(state.summaries)
        state.summaries[i] = (a = 1.0, b = state.summaries[i].b)
    end
    return state
end

@testset "Constant-summary :error policy" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    force_constant_a!(state)

    ad_config = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        NumericalPolicy(floor = 1e-8, constant_policy = :error),
    )
    @test_throws ArgumentError update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
end

@testset "Constant-summary :ignore policy" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    force_constant_a!(state)

    ad_config = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        NumericalPolicy(floor = 1e-8, constant_policy = :ignore),
    )
    update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.weights[1] == 0.0   # constant key :a muted
    @test state.adaptive.weights[2] > 0.0    # varying key :b still weighted
end

@testset "Constant-summary :clamp policy" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    force_constant_a!(state)

    ad_config = AdaptiveDistanceConfig(
        StandardDeviationScale(),
        NumericalPolicy(floor = 1e-8, constant_policy = :clamp, max_weight = 50.0),
    )
    update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.weights[1] == 50.0  # clamped to max_weight
    @test state.adaptive.weights[2] > 0.0
end

@testset "FixedScale correctness" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    ad_config = AdaptiveDistanceConfig(FixedScale([2.0, 0.5]), NumericalPolicy())
    update_ad!(state, ad_config, cfg_2key.distance, Y_OBS)
    @test state.adaptive.weights == [2.0, 0.5]

    # Epsilon-selection distances equal the base distance on pre-scaled
    # summaries (adaptive weighted L2 for this L2-style base distance).
    s1 = state.summaries[1]
    expected = cfg_2key.distance((a = 2.0 * s1.a, b = 0.5 * s1.b),
                                 (a = 2.0 * Y_OBS.a, b = 0.5 * Y_OBS.b))
    @test state.adaptive.epsilon_distances[1] == expected
end

# ── Adaptive lifecycle tests ─────────────────────────────────────────────────

@testset "Acceptance distances distinct from epsilon-selection distances" begin
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T = 3, schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = ad_config)

    # state.dists: acceptance distances of the final generation, computed with
    # the weights that governed that generation.
    @test all(isfinite, state.dists)
    @test all(>(0), state.dists)
    @test length(state.dists) == N

    # state.adaptive.epsilon_distances: the same population recomputed at the
    # boundary under the newly estimated weights.
    @test state.adaptive !== nothing
    @test length(state.adaptive.epsilon_distances) == N

    # Different weight vectors → different distance vectors; acceptance
    # distances were not overwritten by the recomputation.
    @test state.dists != state.adaptive.epsilon_distances
    @test state.gen == 3
end

@testset "Epsilon and distance-weight consistency" begin
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    run_smc!(state, Y_OBS; T = 3, schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = ad_config)

    # Weights were re-estimated at the boundary of the final generation.
    @test state.adaptive.gen_estimated == 3

    # The epsilon frozen for the last executed generation is the FixedSchedule
    # entry for t = 3 — finite, positive, and consistent with state.ε.
    @test state.adaptive.epsilon !== nothing
    @test isfinite(state.adaptive.epsilon)
    @test state.adaptive.epsilon > 0
    @test state.adaptive.epsilon == 0.7
    @test state.ε == 0.7

    # Weights match 1/std of the generation they were estimated from.
    for (k, key) in enumerate([:a, :b])
        s_std = std([s[key] for s in state.summaries])
        @test isapprox(state.adaptive.weights[k], 1.0 / s_std)
    end
end

@testset "NoAdaptation fixed-seed equivalence to current main" begin
    # The plain run in this branch is pinned to main by the unchanged legacy
    # suite; NoAdaptation must reproduce it exactly.
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
    @test state_adapt.ess == state_plain.ess
    # Unit weights at every boundary under NoAdaptation.
    @test all(==(1.0), state_adapt.adaptive.weights)
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
    @test state1.ε == state2.ε
    @test state1.adaptive.key_order == state2.adaptive.key_order
    @test state1.adaptive.weights == state2.adaptive.weights
    @test state1.adaptive.epsilon_distances == state2.adaptive.epsilon_distances
    @test state1.adaptive.epsilon == state2.adaptive.epsilon
    @test state1.adaptive.gen_estimated == state2.adaptive.gen_estimated
end

# ── Schedule tests ───────────────────────────────────────────────────────────

@testset "Fixed schedule ignores supplied distances" begin
    fs = FixedSchedule([3.0, 1.5, 0.7])
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    ε1 = fs(state, 2, state.dists)
    ε2 = fs(state, 2, [100.0, 200.0, 300.0])   # completely different distances
    @test ε1 == ε2 == 1.5

    # 2-arg still works.
    @test fs(state, 2) == 1.5
end

@testset "Quantile schedule uses supplied distances" begin
    qs = QuantileSchedule(0.5)
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    # 3-arg with state.dists agrees with the 2-arg form.
    ε_3arg = qs(state, 2, state.dists)
    ε_2arg = qs(state, 2)
    @test isapprox(ε_3arg, ε_2arg)

    # Different distances → different epsilon.
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
    dir_full = mktempdir()
    dir_part = mktempdir()
    schedule = FixedSchedule([3.0, 1.5, 0.7])

    # Uninterrupted run to generation 3.
    s_full = make_state(cfg_2key, Vector{Float64}, N, 3.0, 7777)
    run_smc!(s_full, Y_OBS; T = 3, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()),
             checkpoint_dir = dir_full)

    # Same run stopped after generation 2.
    s_part = make_state(cfg_2key, Vector{Float64}, N, 3.0, 7777)
    run_smc!(s_part, Y_OBS; T = 2, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()),
             checkpoint_dir = dir_part)

    # Resume from the checkpoint and continue to generation 3.
    s_resumed = load_latest_state(dir_part, cfg_2key)
    @test s_resumed.gen == 2
    @test s_resumed.adaptive !== nothing
    run_smc!(s_resumed, Y_OBS; T = 3, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()),
             checkpoint_dir = dir_part)

    # Exact (bitwise) numerical equivalence of the two trajectories.
    @test s_resumed.θ == s_full.θ
    @test s_resumed.w == s_full.w
    @test s_resumed.dists == s_full.dists
    @test s_resumed.ε == s_full.ε
    @test s_resumed.gen == s_full.gen
    @test s_resumed.ess == s_full.ess
    @test s_resumed.n_attempts == s_full.n_attempts
    @test s_resumed.n_resamples == s_full.n_resamples
    @test s_resumed.adaptive.key_order == s_full.adaptive.key_order
    @test s_resumed.adaptive.weights == s_full.adaptive.weights
    @test s_resumed.adaptive.epsilon_distances == s_full.adaptive.epsilon_distances
    @test s_resumed.adaptive.epsilon == s_full.adaptive.epsilon
    @test s_resumed.adaptive.gen_estimated == s_full.adaptive.gen_estimated
end

@testset "Legacy checkpoint loading" begin
    # Legacy pre-adaptive checkpoint: no schema_version tag, no adaptive block.
    dir = mktempdir()
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)

    path = joinpath(dir, "state_gen_001.jls")
    tmp = path * ".tmp"
    data = (θ = state.θ, w = state.w, summaries = state.summaries,
            dists = state.dists, ε = state.ε, gen = state.gen,
            ess = state.ess, base_seed = state.base_seed,
            n_attempts = state.n_attempts, n_resamples = state.n_resamples)
    open(io -> serialize(io, data), tmp, "w")
    mv(tmp, path)

    # New code loads it with adaptive = nothing …
    state2 = load_checkpoint(path, cfg_2key)
    @test state2.gen == 1
    @test state2.adaptive === nothing
    @test state2.θ == state.θ
    @test state2.dists == state.dists

    # … and continues as a plain non-adaptive run.
    run_smc!(state2, Y_OBS; T = 2, schedule = FixedSchedule([3.0, 1.5]))
    @test state2.gen == 2
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
    @test state2.adaptive.key_order == state.adaptive.key_order
    @test state2.adaptive.weights == state.adaptive.weights
    @test state2.adaptive.epsilon_distances == state.adaptive.epsilon_distances
    @test state2.adaptive.epsilon == state.adaptive.epsilon
    @test state2.adaptive.gen_estimated == state.adaptive.gen_estimated
end

# ── Scalar summary tests ─────────────────────────────────────────────────────

@testset "Scalar summaries" begin
    state = make_state(cfg_scalar, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, SCALAR_OBS)

    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    update_ad!(state, ad_config, cfg_scalar.distance, SCALAR_OBS)
    @test state.adaptive.key_order == [:value]
    @test length(state.adaptive.weights) == 1
    @test state.adaptive.weights[1] > 0
    @test length(state.adaptive.epsilon_distances) == N
    @test all(isfinite, state.adaptive.epsilon_distances)
end

# ── Heterogeneous / informative summary tests ───────────────────────────────

@testset "Heterogeneous summary scales" begin
    # Fixture: key :a has much smaller spread than key :b.
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    update_ad!(state, AdaptiveDistanceConfig(StandardDeviationScale()),
               cfg_2key.distance, Y_OBS)

    a_std = std([s.a for s in state.summaries])
    b_std = std([s.b for s in state.summaries])
    @test b_std > a_std                       # the fixture really is heterogeneous
    @test isapprox(state.adaptive.weights[1], 1.0 / a_std)
    @test isapprox(state.adaptive.weights[2], 1.0 / b_std)
    @test state.adaptive.weights[1] > state.adaptive.weights[2]
end

@testset "Informative plus irrelevant high-variance summaries" begin
    # Key :a is informative (low variance), key :b is noise (high variance):
    # the noise key must be down-weighted.
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    update_ad!(state, AdaptiveDistanceConfig(StandardDeviationScale()),
               cfg_2key.distance, Y_OBS)

    @test state.adaptive.weights[1] > state.adaptive.weights[2]

    # And a full adaptive run still completes with sane state.
    state_run = make_state(cfg_2key, Vector{Float64}, N, 3.0, 5555)
    run_smc!(state_run, Y_OBS; T = 3, schedule = FixedSchedule([3.0, 1.5, 0.7]),
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()))
    @test state_run.gen == 3
    @test all(isfinite, state_run.w)
    @test state_run.adaptive.weights[1] > state_run.adaptive.weights[2]
end

# ── Posterior accuracy on a tractable model ──────────────────────────────────

@testset "Posterior accuracy on tractable model" begin
    # θ ~ N(0,5), summary = θ + N(0,1), observed summary 1.0:
    # posterior θ | s ≈ N(25/26, 25/26), mean ≈ 0.9615.
    state = make_state(cfg_scalar, Vector{Float64}, N, 3.0, 9999)
    run_smc!(state, SCALAR_OBS; T = 5, schedule = QuantileSchedule(0.5))
    θbar = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
    @test 0.4 < θbar < 1.6

    # Same accuracy with adaptive distance scaling active.
    state_ad = make_state(cfg_scalar, Vector{Float64}, N, 3.0, 9999)
    run_smc!(state_ad, SCALAR_OBS; T = 5, schedule = QuantileSchedule(0.5),
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()))
    θbar_ad = sum(state_ad.θ[i][1] * state_ad.w[i] for i in eachindex(state_ad.w))
    @test 0.4 < θbar_ad < 1.6
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
    @test state.adaptive.key_order == [:a, :b]
    @test state.adaptive.gen_estimated == 4
    @test isfinite(state.adaptive.epsilon)
    @test state.adaptive.epsilon > 0

    # The final checkpoint round-trips the restart-critical adaptive state.
    st2 = load_latest_state(dir, cfg_2key)
    @test st2.gen == 4
    @test st2.adaptive.weights == state.adaptive.weights
    @test st2.adaptive.epsilon == state.adaptive.epsilon
end
