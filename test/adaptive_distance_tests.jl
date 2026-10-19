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
#   key :a — low variance, tight prior    (θ₁ ~ N(0, 2.5) + N(0,1) noise)
#   key :b — diffuse, high variance       (θ₂ ~ N(0, 25)  + N(0,1) noise)
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
const adaptive_dist = SequentialABC.adaptive_distance
const mean_summary = SequentialABC.mean_summary

function weighted_distance_2key(base_distance, s, s_obs, weights)
    return base_distance(
        (a = weights[1] * s.a, b = weights[2] * s.b),
        (a = weights[1] * s_obs.a, b = weights[2] * s_obs.b),
    )
end

function write_checkpoint_payload(dir, name, payload)
    path = joinpath(dir, name)
    open(io -> serialize(io, payload), path, "w")
    return path
end

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

@testset "ABCSMCState eleven-argument compatibility constructor" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    compatible = ABCSMCState(
        state.cfg, state.θ, state.w, state.summaries, state.dists, state.ε,
        state.gen, state.ess, state.base_seed, state.n_attempts,
        state.n_resamples,
    )
    @test compatible.adaptive === nothing
    @test compatible.θ === state.θ
    @test compatible.summaries === state.summaries
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
    @test state.adaptive.epsilon_selection_distances[1] == expected
end

@testset "FixedScale governs generation 1" begin
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    fixed = [2.0, 0.5]
    run_smc!(state, Y_OBS; T = 1, schedule = (_, _) -> error("schedule called"),
             adaptive = AdaptiveDistanceConfig(FixedScale(fixed)))
    @test state.adaptive.weights == fixed
    @test state.adaptive.gen_applied == 1
    @test isempty(state.adaptive.epsilon_selection_distances)
    @test state.dists == [weighted_distance_2key(
        cfg_2key.distance, s, Y_OBS, fixed) for s in state.summaries]
end

@testset "Deterministic p=2 scale-weight reference check" begin
    # Fixed fixture: 5 accepted summaries, 2 keys, no constant coordinates and
    # unit initial weights. Every expected value below is derived from first
    # principles — arithmetic means, the (n-1) sample-variance formula, inverse
    # scale weights, scaled coordinate differences, the weighted Euclidean
    # distance and a sorted median — never from the production helpers under
    # test.
    a_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    b_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    obs_ref = (a = 3.0, b = 30.0)
    n_ref = 5

    a_mean = sum(a_vals) / n_ref
    b_mean = sum(b_vals) / n_ref
    a_var = sum(abs2, a_vals .- a_mean) / (n_ref - 1)
    b_var = sum(abs2, b_vals .- b_mean) / (n_ref - 1)
    @test a_var == 2.5                    # known closed-form variances
    @test b_var == 250.0
    expected_weights = [1.0 / sqrt(a_var), 1.0 / sqrt(b_var)]

    state = make_state(cfg_2key, Vector{Float64}, n_ref, 3.0, 7)
    state.summaries = [(a = a_vals[i], b = b_vals[i]) for i in 1:n_ref]
    state.gen = 1
    update_ad!(state, AdaptiveDistanceConfig(StandardDeviationScale()),
               cfg_2key.distance, obs_ref)

    # Estimated inverse standard-deviation weights.
    @test state.adaptive.weights == expected_weights

    # Scaled coordinate differences and the weighted Euclidean distance of
    # every accepted summary, evaluated exactly as the p = 2 workflow does:
    # scale both summaries first, then take the Euclidean distance.
    w1, w2 = expected_weights
    expected_distances = [sqrt(((a_vals[i] * w1) - (obs_ref.a * w1))^2 +
                               ((b_vals[i] * w2) - (obs_ref.b * w2))^2)
                          for i in 1:n_ref]
    @test state.adaptive.epsilon_selection_distances == expected_distances

    # Quantile epsilon: with n = 5 the q = 0.5 quantile is exactly the middle
    # of the sorted expected distances — obtained here by sorting, not by any
    # production helper.
    expected_epsilon = sort(expected_distances)[3]
    @test SequentialABC.epsilon_quantile(
        state.adaptive.epsilon_selection_distances, 0.5) == expected_epsilon
    @test QuantileSchedule(0.5)(state, 2,
                                state.adaptive.epsilon_selection_distances) ==
          expected_epsilon
end

# ── Adaptive lifecycle tests ─────────────────────────────────────────────────

@testset "T=3 adaptive lifecycle and generation provenance" begin
    ad_config = AdaptiveDistanceConfig(StandardDeviationScale())
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    preparation_trace = Any[]
    epsilon_ladder = [3.0, 1.5, 0.7]

    instrumented_schedule = function (current_state, t, distances)
        # The schedule is called immediately after update_adaptive!, while the
        # accepted summaries still belong to generation t-1.
        weights = copy(current_state.adaptive.weights)
        summaries = copy(current_state.summaries)
        expected = [weighted_distance_2key(cfg_2key.distance, s, Y_OBS, weights)
                    for s in summaries]
        push!(preparation_trace, (
            requested_generation = t,
            completed_generation = current_state.gen,
            gen_applied = current_state.adaptive.gen_applied,
            weights = weights,
            summaries = summaries,
            distances = copy(distances),
        ))
        @test distances == expected
        return epsilon_ladder[t]
    end

    run_smc!(state, Y_OBS; T = 3, schedule = instrumented_schedule,
             adaptive = ad_config)

    # Only generations 2 and 3 are prepared: once each, never generation 4.
    @test [entry.requested_generation for entry in preparation_trace] == [2, 3]
    @test [entry.completed_generation for entry in preparation_trace] == [1, 2]
    @test [entry.gen_applied for entry in preparation_trace] == [2, 3]

    # The final frozen selection vector was computed from generation 2 and used
    # to choose epsilon_3. It does not describe the generation-3 population.
    @test state.adaptive !== nothing
    @test state.adaptive.gen_applied == 3
    @test state.adaptive.weights == preparation_trace[end].weights
    @test state.adaptive.epsilon_selection_distances ==
          preparation_trace[end].distances
    @test state.adaptive.epsilon == epsilon_ladder[3]

    # state.dists is independently tied to the accepted generation-3 summaries,
    # evaluated under the same frozen weights_3.
    final_acceptance_distances = [
        weighted_distance_2key(cfg_2key.distance, s, Y_OBS,
                               state.adaptive.weights)
        for s in state.summaries
    ]
    @test state.dists == final_acceptance_distances
    @test state.gen == 3
    @test state.ε == epsilon_ladder[3]
end

@testset "NoAdaptation fixed-seed equivalence to current main" begin
    # The plain run in this branch is pinned to main by the unchanged legacy
    # suite; NoAdaptation must reproduce it exactly.
    schedule = FixedSchedule([3.0, 1.5, 0.7])

    state_adapt = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    dir_adapt = mktempdir()
    run_smc!(state_adapt, Y_OBS; T = 3, schedule = schedule,
             adaptive = AdaptiveDistanceConfig(NoAdaptation()),
             checkpoint_dir = dir_adapt)

    state_plain = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    dir_plain = mktempdir()
    run_smc!(state_plain, Y_OBS; T = 3, schedule = schedule,
             checkpoint_dir = dir_plain)

    @test state_adapt.θ == state_plain.θ
    @test state_adapt.w == state_plain.w
    @test state_adapt.dists == state_plain.dists
    @test state_adapt.ε == state_plain.ε
    @test state_adapt.gen == state_plain.gen
    @test state_adapt.ess == state_plain.ess
    @test state_adapt.n_attempts == state_plain.n_attempts
    @test state_adapt.n_resamples == state_plain.n_resamples
    # NoAdaptation normalizes to the plain non-adaptive path: no adaptive state.
    @test state_adapt.adaptive === nothing

    payload_adapt = open(deserialize, last(list_checkpoints(dir_adapt))[2])
    payload_plain = open(deserialize, last(list_checkpoints(dir_plain))[2])
    @test payload_adapt == payload_plain
    @test payload_adapt.schema_version == 1
    @test !haskey(payload_adapt, :adaptive)
end

@testset "Callable distances on plain and adaptive paths" begin
    plain = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    simulated = (a = 2.0, b = 4.0)
    observed = (a = 1.0, b = 0.0)
    ordinary(s, so) = sqrt((s.a - so.a)^2 + (s.b - so.b)^2)
    closure_scale = 2.0
    closure = (s, so) -> closure_scale * ordinary(s, so)
    callable = weighted_l2((a = 4.0, b = 0.25))

    @test adaptive_dist(plain, ordinary, simulated, observed) ==
          ordinary(simulated, observed)
    @test adaptive_dist(plain, closure, simulated, observed) ==
          closure(simulated, observed)
    @test adaptive_dist(plain, callable, simulated, observed) ==
          callable(simulated, observed)

    initialize!(plain, Y_OBS)
    update_ad!(plain, AdaptiveDistanceConfig(FixedScale([2.0, 0.5])),
               cfg_2key.distance, Y_OBS)
    scaled_simulated = (a = 2.0 * simulated.a, b = 0.5 * simulated.b)
    scaled_observed = (a = 2.0 * observed.a, b = 0.5 * observed.b)
    @test adaptive_dist(plain, ordinary, simulated, observed) ==
          ordinary(scaled_simulated, scaled_observed)
    @test adaptive_dist(plain, closure, simulated, observed) ==
          closure(scaled_simulated, scaled_observed)
    @test adaptive_dist(plain, callable, simulated, observed) ==
          callable(scaled_simulated, scaled_observed)
end

@testset "mean_summary preserves legacy and named-summary behavior" begin
    @test mean_summary([2.0], 1) == 2.0
    @test mean_summary([2.0, 4.0], 2) == 3.0
    # M = 1 returns the sole replicate unchanged (identity, no arithmetic).
    v_m1 = [2.0, 4.0]
    @test mean_summary([v_m1], 1) === v_m1
    @test mean_summary([[2.0, 4.0], [4.0, 8.0]], 2) == [3.0, 6.0]
    named = (a = 2.0, b = 4.0)
    @test mean_summary([named], 1) === named
    @test mean_summary([(a = 2.0, b = 4.0), (a = 4.0, b = 8.0)], 2) ==
          (a = 3.0, b = 6.0)
    @test_throws ArgumentError mean_summary(
        [(a = 1.0, b = 2.0), (a = 3.0, c = 4.0)], 2)
    @test_throws ArgumentError mean_summary(
        [(a = 1.0, b = "bad"), (a = 2.0, b = "worse")], 2)
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
    @test state1.adaptive.epsilon_selection_distances == state2.adaptive.epsilon_selection_distances
    @test state1.adaptive.epsilon == state2.adaptive.epsilon
    @test state1.adaptive.gen_applied == state2.adaptive.gen_applied
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
    resume_trace = Any[]
    resume_schedule = function (current_state, t, distances)
        push!(resume_trace, (
            requested_generation = t,
            completed_generation = current_state.gen,
            gen_applied = current_state.adaptive.gen_applied,
        ))
        return schedule(current_state, t, distances)
    end
    run_smc!(s_resumed, Y_OBS; T = 3, schedule = resume_schedule,
             adaptive = AdaptiveDistanceConfig(StandardDeviationScale()),
             checkpoint_dir = dir_part)

    @test resume_trace == [(
        requested_generation = 3,
        completed_generation = 2,
        gen_applied = 3,
    )]

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
    @test s_resumed.adaptive.epsilon_selection_distances == s_full.adaptive.epsilon_selection_distances
    @test s_resumed.adaptive.epsilon == s_full.adaptive.epsilon
    @test s_resumed.adaptive.gen_applied == s_full.adaptive.gen_applied
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
    payload = open(deserialize, last(list_checkpoints(dir))[2])
    @test payload.schema_version == 2
    @test keys(payload.adaptive) == (
        :key_order,
        :weights,
        :gen_applied,
        :epsilon_selection_distances,
        :epsilon,
    )
    @test state2.gen == 2
    @test state2.adaptive !== nothing
    @test state2.adaptive.key_order == state.adaptive.key_order
    @test state2.adaptive.weights == state.adaptive.weights
    @test state2.adaptive.epsilon_selection_distances == state.adaptive.epsilon_selection_distances
    @test state2.adaptive.epsilon == state.adaptive.epsilon
    @test state2.adaptive.gen_applied == state.adaptive.gen_applied
end


@testset "Checkpoint schema compatibility matrix" begin
    dir = mktempdir()
    state = make_state(cfg_2key, Vector{Float64}, N, 3.0, 1234)
    initialize!(state, Y_OBS)
    base = (
        θ = state.θ,
        w = state.w,
        summaries = state.summaries,
        dists = state.dists,
        ε = state.ε,
        gen = state.gen,
        ess = state.ess,
        base_seed = state.base_seed,
        n_attempts = state.n_attempts,
        n_resamples = state.n_resamples,
    )
    adaptive_block = (
        key_order = [:a, :b],
        weights = [1.0, 1.0],
        gen_applied = 1,
        epsilon_selection_distances = Float64[],
        epsilon = state.ε,
    )

    # Schema 0 is an untagged legacy payload; schema 1 is tagged and plain.
    schema0 = write_checkpoint_payload(dir, "schema0.jls", base)
    schema1 = write_checkpoint_payload(
        dir, "schema1.jls", merge((schema_version = 1,), base))
    @test load_checkpoint(schema0, cfg_2key).adaptive === nothing
    @test load_checkpoint(schema1, cfg_2key).adaptive === nothing

    # Declared schema and payload shape must agree exactly.
    schema0_adaptive = write_checkpoint_payload(
        dir, "schema0-adaptive.jls", merge(base, (adaptive = adaptive_block,)))
    schema1_adaptive = write_checkpoint_payload(
        dir, "schema1-adaptive.jls",
        merge((schema_version = 1,), base, (adaptive = adaptive_block,)))
    schema2_plain = write_checkpoint_payload(
        dir, "schema2-plain.jls", merge((schema_version = 2,), base))
    future_schema = write_checkpoint_payload(
        dir, "schema3.jls", merge((schema_version = 3,), base))
    malformed_version = write_checkpoint_payload(
        dir, "schema-string.jls", merge((schema_version = "2",), base))
    # malformed_weight_type
    mw_t = write_checkpoint_payload(
        dir, "mw-t.jls",
        merge((schema_version = 2,), base,
              (adaptive = merge(adaptive_block, (weights = "bad",)),)))
    @test_throws ArgumentError load_checkpoint(mw_t, cfg_2key)
    # inconsistent lengths
    mw_l = write_checkpoint_payload(
        dir, "mw-l.jls",
        merge((schema_version = 2,), base,
              (adaptive = merge(adaptive_block, (weights = [1.0],)),)))
    @test_throws ArgumentError load_checkpoint(mw_l, cfg_2key)
    # missing field
    m_missing = write_checkpoint_payload(
        dir, "m-missing.jls",
        merge((schema_version = 2,), base,
              (adaptive = (key_order = [:a,:b], weights = [1.0,1.0],
                           epsilon_selection_distances = Float64[], epsilon = state.ε),)))
    @test_throws ArgumentError load_checkpoint(m_missing, cfg_2key)
    # bad gen_applied
    m_badgen = write_checkpoint_payload(
        dir, "m-badgen.jls",
        merge((schema_version = 2,), base,
              (adaptive = merge(adaptive_block, (gen_applied = 5,)),)))
    @test_throws ArgumentError load_checkpoint(m_badgen, cfg_2key)
    # bad distances length (poplen N)
    m_badlen = write_checkpoint_payload(
        dir, "m-badlen.jls",
        merge((schema_version = 2,), base,
              (adaptive = merge(adaptive_block, (epsilon_selection_distances = [0.1],)),)))
    @test_throws ArgumentError load_checkpoint(m_badlen, cfg_2key)
    # bad epsilon (nonfinite/neg)
    m_bad_eps = write_checkpoint_payload(
        dir, "m-badeps.jls",
        merge((schema_version = 2,), base,
              (adaptive = merge(adaptive_block, (epsilon = NaN,)),)))
    @test_throws ArgumentError load_checkpoint(m_bad_eps, cfg_2key)
    m_bad_eps2 = write_checkpoint_payload(
        dir, "m-badeps2.jls",
        merge((schema_version = 2,), base,
              (adaptive = merge(adaptive_block, (epsilon = -1.0,)),)))
    @test_throws ArgumentError load_checkpoint(m_bad_eps2, cfg_2key)
    # schema0_adaptive / schema1_adaptive / schema2_plain
    schema0_adaptive = write_checkpoint_payload(
        dir, "schema0-adaptive.jls", merge(base, (adaptive = adaptive_block,)))
    schema1_adaptive = write_checkpoint_payload(
        dir, "schema1-adaptive.jls",
        merge((schema_version = 1,), base, (adaptive = adaptive_block,)))
    schema2_plain = write_checkpoint_payload(
        dir, "schema2-plain.jls", merge((schema_version = 2,), base))
    future_schema = write_checkpoint_payload(
        dir, "schema3.jls", merge((schema_version = 3,), base))
    malformed_version = write_checkpoint_payload(
        dir, "schema-string.jls", merge((schema_version = "2",), base))
    @test_throws ArgumentError load_checkpoint(schema0_adaptive, cfg_2key)
    @test_throws ArgumentError load_checkpoint(schema1_adaptive, cfg_2key)
    @test_throws ArgumentError load_checkpoint(schema2_plain, cfg_2key)
    @test_throws ArgumentError load_checkpoint(future_schema, cfg_2key)
    @test_throws ArgumentError load_checkpoint(malformed_version, cfg_2key)
    @test_throws ArgumentError load_checkpoint(mw_t, cfg_2key)
    @test_throws ArgumentError load_checkpoint(mw_l, cfg_2key)
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
    @test length(state.adaptive.epsilon_selection_distances) == N
    @test all(isfinite, state.adaptive.epsilon_selection_distances)
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

@testset "Heterogeneous-scale informative summaries" begin
    # Key :a has low spread (tight prior), key :b has high spread (diffuse
    # prior): the diffuse key must be down-weighted.
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

@testset "Posterior mean sanity check" begin
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
    @test state.adaptive.gen_applied == 4
    @test isfinite(state.adaptive.epsilon)
    @test state.adaptive.epsilon > 0

    # The final checkpoint round-trips the restart-critical adaptive state.
    st2 = load_latest_state(dir, cfg_2key)
    @test st2.gen == 4
    @test st2.adaptive.weights == state.adaptive.weights
    @test st2.adaptive.epsilon == state.adaptive.epsilon
end
