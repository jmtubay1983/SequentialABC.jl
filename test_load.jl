using SequentialABC
using Distributions
using LinearAlgebra
using Random
using Test

println("=== SequentialABC.jl Phase‑1 smoke test (Julia ", VERSION, ") ===")

# ---------------------------------------------------------------------------
# Toy model: Gaussian mean estimation
#   μ ~ N(0, 5²)  (prior)
#   y | μ ~ N(μ, 1), observed y = 1.0
#   summary = simulated y; distance = |y_sim − y_obs|
# Analytic posterior: μ | y=1 ~ N(25/26, 25/26) ≈ N(0.9615, 0.9615)
# ---------------------------------------------------------------------------
const PRIOR = Normal(0.0, 5.0)
const Y_OBS = 1.0

sample_prior = rng -> [rand(rng, PRIOR)]
prior_logpdf = θ -> logpdf(PRIOR, θ[1])          # log-density on raw space (vector θ)
transform    = θ -> θ                             # identity for this toy
distance     = (s, s_obs) -> abs(s - s_obs)

simulator = ContinuousSimulator(
    model_builder = μ -> μ[1],
    evolve        = (μ, rng) -> μ + randn(rng),
    summary       = y -> y,
)

N     = 100
Σ_mat = [0.5;;]
cfg   = ABCSMCConfig(prior_logpdf, sample_prior, transform, distance,
                     GaussianKernel(), Σ_mat, simulator)

schedule = (state, t) -> [3.0, 1.0, 0.3][t]

# ---------------------------------------------------------------------------
# Test 1 — kernel log-density matches Distributions.logpdf(MvNormal)
# ---------------------------------------------------------------------------
let K = GaussianKernel(), Σ = [0.8 0.1; 0.1 0.6]
    Σf   = Matrix{Float64}(Σ)
    Σinv = inv(Σf)
    ld   = logdet(Σf)
    for (x, μ) in [([1.2, -0.4], [0.7, 0.3]), ([0.0, 0.0], [0.0, 0.0])]
        ours = K(Vector{Float64}(x), Vector{Float64}(μ), Σinv, ld)
        ref  = logpdf(MvNormal(Vector{Float64}(μ), Σf), Vector{Float64}(x))
        @assert isapprox(ours, ref; atol=1e-12) "kernel logpdf mismatch: $ours vs $ref"
    end
    println("✓ Test 1: kernel log-density == logpdf(MvNormal)")
end

# ---------------------------------------------------------------------------
# Test 2 — full run: weights, ESS, sanity
# ---------------------------------------------------------------------------
state = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
run_smc!(state, Y_OBS; T=3, schedule=schedule)

@assert state.gen == 3
@assert isapprox(sum(state.w), 1.0; atol=1e-10) "weights do not sum to 1"
@assert all(isfinite, state.w) && all(θ -> all(isfinite, θ), state.θ) "non-finite values"
@assert all(>(0), state.w) "zero-weight particles present"
@assert 1.0 <= state.ess <= N + 1e-9 "ESS out of range: $(state.ess)"

θbar = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
@assert 0.4 < θbar < 1.6 "posterior mean far from analytic value 0.96: $θbar"
println("✓ Test 2: 3 generations run; Σw=1, ESS=$(round(state.ess, digits=1)), weighted mean=$(round(θbar, digits=3)) (analytic ≈ 0.96)")

# ---------------------------------------------------------------------------
# Test 3 — determinism: same base_seed ⇒ bit-identical populations
# ---------------------------------------------------------------------------
state2 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
run_smc!(state2, Y_OBS; T=3, schedule=schedule)

@assert state.θ == state2.θ "populations differ under identical seeds"
@assert state.w == state2.w "weights differ under identical seeds"
println("✓ Test 3: determinism — two runs bit-identical")

state3 = make_state(cfg, Vector{Float64}, N, 3.0, 9999)
run_smc!(state3, Y_OBS; T=3, schedule=schedule)
@assert state.θ != state3.θ "different seeds produced identical populations (suspicious)"
println("✓ Test 3b: different base_seed ⇒ different population")

# ---------------------------------------------------------------------------
# Test 4 — restartability: interrupted (2 then deepcopy + 1) == uninterrupted (3)
# ---------------------------------------------------------------------------
stateA = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
run_smc!(stateA, Y_OBS; T=2, schedule=schedule)
stateB = deepcopy(stateA)                       # simulates checkpoint reload
run_smc!(stateB, Y_OBS; T=3, schedule=schedule)

@assert stateB.θ == state.θ "restart produced different particles"
@assert stateB.w == state.w "restart produced different weights"
println("✓ Test 4: restart after gen 2 == uninterrupted run")

# ---------------------------------------------------------------------------
# Test 5 — gen‑1: uniform weights, ESS = N
# ---------------------------------------------------------------------------
stateG1 = make_state(cfg, Vector{Float64}, N, 3.0, 1234)
initialize!(stateG1, Y_OBS)
@assert stateG1.gen == 1
@assert all(==(1.0 / N), stateG1.w) "gen-1 weights not uniform"
@assert stateG1.ess == Float64(N) "gen-1 ESS should be N, got $(stateG1.ess)"
println("✓ Test 5: gen-1 uniform weights, ESS = N")

# ---------------------------------------------------------------------------
# Test 6 — systematic_resample unit tests
# ---------------------------------------------------------------------------
let w = [0.1, 0.2, 0.3, 0.4]
    # 6a. Determinism: same seed ⇒ same indices
    idx1 = systematic_resample(w, MersenneTwister(42))
    idx2 = systematic_resample(w, MersenneTwister(42))
    @assert idx1 == idx2 "systematic_resample not deterministic under same seed"

    # 6b. Valid indices
    @assert all(i -> 1 <= i <= 4, idx1) "systematic_resample returned invalid indices"

    # 6c. Degenerate weights: all mass on particle 1 ⇒ all indices = 1
    @assert all(==(1), systematic_resample([1.0, 0.0, 0.0], MersenneTwister(1))) "degenerate weights mishandled"

    # 6d. Fairness: long-run frequencies ≈ weights
    rng = MersenneTwister(7)
    counts = zeros(Int, 4)
    for _ in 1:10_000
        for k in systematic_resample(w, rng)
            counts[k] += 1
        end
    end
    freqs = counts ./ (10_000 * 4)   # each call draws N=4 indices
    @assert all(abs.(freqs .- w) .< 0.02) "resampling frequencies deviate from weights: $freqs"
    println("✓ Test 6: systematic_resample — deterministic, valid, fair")
end

# ---------------------------------------------------------------------------
# Test 7 — resampling branch fires when ESS collapses
#   (prior σ=0.05 ≪ kernel sd ≈ 0.71 ⇒ gen‑2 proposals land deep in the prior
#    tails where π(θ) ≈ 0 ⇒ importance weights concentrate on a handful of
#    particles ⇒ ESS < 0.5N ⇒ systematic resampling fires)
# ---------------------------------------------------------------------------
let narrow = Normal(0.0, 0.05)
    cfg_p = ABCSMCConfig(θ -> logpdf(narrow, θ[1]),
                         rng -> [rand(rng, narrow)],
                         transform, distance,
                         GaussianKernel(), [0.5;;], simulator)
    sched_p = (state, t) -> [3.0, 1.0][t]

    sA = make_state(cfg_p, Vector{Float64}, N, 3.0, 4242)
    run_smc!(sA, Y_OBS; T=2, schedule=sched_p)

    @assert sA.n_resamples >= 1 "ESS collapse did not trigger resampling"
    @assert isapprox(sum(sA.w), 1.0; atol=1e-10) "weights not uniform after resampling"
    @assert sA.ess == Float64(N) "ESS not reset to N after resampling"

    # Resampling path is deterministic too
    sB = make_state(cfg_p, Vector{Float64}, N, 3.0, 4242)
    run_smc!(sB, Y_OBS; T=2, schedule=sched_p)
    @assert sA.θ == sB.θ && sA.w == sB.w "resampling path not deterministic"
    println("✓ Test 7: ESS < 0.5N triggers deterministic resampling; weights reset to 1/N")
end

println("=== ALL SMOKE TESTS PASSED ===")
