# test/gates/gate1_api.jl
# Gate 1 — Production API probes.
#
# Exercises production code only (entrypoints, seed derivation, checkpoint,
# importance weights). Records CAPABILITY_NOT_IMPLEMENTED for capabilities the
# engine does not provide; never installs a test-local stand-in.
#
# Run via: julia --project=. test/gates/run_gates.jl
# Standalone: include("test/gates/gate_common.jl"); run_gate1()

function run_gate1()
    r = GateResult("Gate 1 — production API")
    dim = 1
    cfg = _std_config(; dim=dim, noise=1.0, prior_sd=5.0)
    s_obs = zeros(dim)
    base_seed = 20260101

    # ---- 1. Seed derivation: deterministic + 10_000-block collision scan ----
    s1 = make_particle_seed(1, 2, 3, 0, base_seed)
    s2 = make_particle_seed(1, 2, 3, 0, base_seed)
    check!(r, "make_particle_seed deterministic (fixed inputs → same output)",
           s1 == s2; detail="$(s1) != $(s2)")

    # Enumerate 10_000 distinct (gen, i, attempt, m) tuples and verify the
    # production seed derivation is injective (no collisions).
    seen = Set{UInt64}()
    nblocks = 10_000
    count = 0
    for gen in 1:10, i in 1:100, attempt in 1:10, m in (0, -1, -2, 1)
        count >= nblocks && break
        push!(seen, make_particle_seed(gen, i, attempt, m, base_seed))
        count += 1
    end
    check!(r, "10_000-block seed collision scan (all unique)",
           count == nblocks && length(seen) == nblocks;
           detail="$(count) blocks, $(length(seen)) unique seeds")

    # Role separation: same (gen, i, attempt, base), different m → different seed.
    roles = (ancestor = 0, perturbation = -1, resample = -2, replicate = 3)
    role_seeds = [make_particle_seed(1, 1, 1, Int(m), base_seed) for m in values(roles)]
    check!(r, "Seed role separation (distinct m → distinct seed)",
           length(Set(role_seeds)) == length(roles);
           detail="role seeds=$(role_seeds)")

    # ---- 2. Checkpoint write / read round-trip ----
    cdir = mktempdir()
    state = run_smc(cfg, s_obs; N=12, T=2, base_seed=base_seed,
                    checkpoint_dir=cdir, schedule=QuantileSchedule(0.5))
    check!(r, "run_smc completes T generations", state.gen == 2;
           detail="gen=$(state.gen)")
    cps = list_checkpoints(cdir)
    check!(r, "checkpoints written for gen 1 and 2",
           length(cps) == 2 && [c[1] for c in cps] == [1, 2];
           detail="gens=$( [c[1] for c in cps] )")
    loaded = load_latest_state(cdir, cfg)
    check!(r, "loaded checkpoint matches final state (θ)", loaded.θ == state.θ)
    check!(r, "loaded checkpoint matches final state (w)", loaded.w == state.w)
    check!(r, "loaded checkpoint matches final state (dists)", loaded.dists == state.dists)
    check!(r, "loaded checkpoint matches final state (gen)", loaded.gen == state.gen)
    check!(r, "loaded checkpoint matches final state (ess)", loaded.ess == state.ess)
    check!(r, "loaded checkpoint matches final state (base_seed)",
           loaded.base_seed == state.base_seed)
    check!(r, "loaded checkpoint matches final state (n_attempts)",
           loaded.n_attempts == state.n_attempts)
    check!(r, "loaded checkpoint matches final state (n_resamples)",
           loaded.n_resamples == state.n_resamples)

    # ---- 3. Importance weights finite and well-formed ----
    θ_prev = [[randn(MersenneTwister(1))] for _ in 1:5]
    w_prev = fill(1.0 / 5, 5)
    θ_prop = [randn(MersenneTwister(2))]
    w = importance_weight(cfg, θ_prop, θ_prev, w_prev)
    check!(r, "importance_weight finite", isfinite(w); detail="w=$w")
    check!(r, "importance_weight non-negative", w >= 0.0; detail="w=$w")

    # ---- Capability gaps: production does not provide these ----
    capability_gap!(r, "EvaluationContext")
    capability_gap!(r, "proposal_id / evaluation_id")
    capability_gap!(r, "ancestry")
    capability_gap!(r, "cache of retained particles")
    capability_gap!(r, "mid-generation checkpointing")

    return r
end
