# test/gates/gate4_external_simulator.jl
# Gate 4 — Generic external-simulator example.
#
# Synthetic stochastic model implementing the production callback contract with
# explicit replicate seeds, plus a reduced SMC run via the public API
# (50 particles, 9 replicates, >= 3 generations).
#
# Returns ABM_INTEGRATION_NOT_RUN when no downstream adapter is supplied to
# this repository (model-agnostic: no swIAV or biology code lives here).
# Missing production features (EvaluationContext / EvaluationResult /
# replicate_seeds / provenance) are recorded as capability gaps, not shims.
#
# Run via: julia --project=. test/gates/run_gates.jl
# Standalone: include("test/gates/gate_common.jl"); run_gate4()

function run_gate4()
    r = GateResult("Gate 4 — external simulator example")

    # ------------------------------------------------------------------
    # Production features this gate needs but the engine does not provide.
    # Recorded as gaps (never shimmed).
    # ------------------------------------------------------------------
    capability_gap!(r, "EvaluationContext")
    capability_gap!(r, "EvaluationResult")
    capability_gap!(r, "replicate_seeds")
    capability_gap!(r, "provenance")

    # ------------------------------------------------------------------
    # Generic synthetic stochastic model: consumes an explicit replicate seed.
    # Model-agnostic — no domain terms.
    # ------------------------------------------------------------------
    function fake_stochastic_model(theta::AbstractVector, seed::UInt)
        rng = MersenneTwister(seed)
        return [theta[1] + 0.1 * randn(rng),
                theta[1]^2 + 0.1 * randn(rng)]
    end

    s_a = fake_stochastic_model([1.0], UInt(42))
    s_b = fake_stochastic_model([1.0], UInt(42))
    s_c = fake_stochastic_model([1.0], UInt(43))
    check!(r, "fake_stochastic_model deterministic given (theta, seed)",
           s_a == s_b; detail="determinism failed")
    check!(r, "fake_stochastic_model differs across replicate seeds",
           s_a != s_c; detail="seed has no effect")

    # ------------------------------------------------------------------
    # Wire into the production callback contract (ContinuousSimulator).
    # evolve receives production's seeded rng and derives an explicit seed
    # for the stochastic model — demonstrating how an external-process
    # adapter would map production seeds into an explicit-seed model.
    # ------------------------------------------------------------------
    sim = ContinuousSimulator(
        model_builder = theta -> theta,
        evolve = (problem, rng) -> fake_stochastic_model(problem, rand(rng, UInt)),
        summary = identity,
    )
    cfg = ABCSMCConfig(
        theta -> sum(logpdf(Normal(0.0, 5.0), x) for x in theta),
        rng -> 5.0 .* randn(rng, 1),
        identity,
        (s, s_obs) -> sqrt(sum(abs2, s .- s_obs)),
        GaussianKernel(), Matrix{Float64}(I, 1, 1), sim,
    )

    # ------------------------------------------------------------------
    # Reduced integration run via public API:
    #   N = 50 particles, M = 9 replicates, T = 3 generations
    # ------------------------------------------------------------------
    st = run_smc(cfg, [1.0, 1.0]; N=50, T=3, M=9, base_seed=20260501,
                 schedule=QuantileSchedule(0.5))
    check!(r, "Reduced SMC completes T=3 generations", st.gen == 3;
           detail="gen=$(st.gen)")
    check!(r, "Reduced SMC weights normalize to 1",
           isapprox(sum(st.w), 1.0; atol=1e-12); detail="sum(w)=$(sum(st.w))")
    check!(r, "Reduced SMC ESS > 0", st.ess > 0.0; detail="ess=$(st.ess)")

    # ------------------------------------------------------------------
    # No downstream adapter is supplied to this repository → NOT_RUN
    # (not a pass; the downstream repo wires its own adapter + context).
    # ------------------------------------------------------------------
    r.not_run_reason = "ABM_INTEGRATION_NOT_RUN: no downstream adapter supplied"
    return r
end
