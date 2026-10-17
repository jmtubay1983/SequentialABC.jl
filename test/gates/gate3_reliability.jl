# test/gates/gate3_reliability.jl
# Gate 3 — Execution reliability.
#
# Bit-identical rerun, exact restart-from-checkpoint equivalence, corrupt
# checkpoint rejection, simulator-failure propagation, serial vs Threads.@spawn
# identity.
#
# Run via: julia --project=. test/gates/run_gates.jl
# Standalone: include("test/gates/gate_common.jl"); run_gate3()

using Serialization

function run_gate3()
    r = GateResult("Gate 3 — execution reliability")
    base_seed = 20260401
    T = 4
    dim = 1
    cfg = _std_config(; dim=dim, noise=0.5, prior_sd=4.0)
    s_obs = [0.5]

    # ------------------------------------------------------------------
    # 3a. Bit-identical rerun: same config + same base_seed → identical state
    # ------------------------------------------------------------------
    a = run_smc(cfg, s_obs; N=80, T=4, base_seed=base_seed,
                schedule=QuantileSchedule(0.5))
    b = run_smc(cfg, s_obs; N=80, T=4, base_seed=base_seed,
                schedule=QuantileSchedule(0.5))
    check!(r, "Bit-identical rerun: θ", a.θ == b.θ; detail="θ differs")
    check!(r, "Bit-identical rerun: w", a.w == b.w; detail="w differs")
    check!(r, "Bit-identical rerun: dists", a.dists == b.dists; detail="dists differs")
    check!(r, "Bit-identical rerun: ε", a.ε == b.ε; detail="ε differs")
    check!(r, "Bit-identical rerun: ess", a.ess == b.ess; detail="ess differs")
    check!(r, "Bit-identical rerun: n_attempts", a.n_attempts == b.n_attempts;
           detail="n_attempts differs")
    check!(r, "Bit-identical rerun: n_resamples", a.n_resamples == b.n_resamples;
           detail="n_resamples differs")

    # ------------------------------------------------------------------
    # 3b. Restart from each generation checkpoint == uninterrupted run
    #     Compare: θ, w, dists, ε, ESS, n_attempts, n_resamples
    # ------------------------------------------------------------------
    cdir = mktempdir()
    full = run_smc(cfg, s_obs; N=60, T=4, base_seed=base_seed,
                   checkpoint_dir=cdir, schedule=QuantileSchedule(0.5))
    check!(r, "Uninterrupted run completes T=4", full.gen == 4; detail="gen=$(full.gen)")
    cps = list_checkpoints(cdir)
    gens = [c[1] for c in cps]
    check!(r, "Checkpoints exist for gens 1..T",
           length(cps) == T && gens == collect(1:T);
           detail="gens=$gens expected=$(collect(1:T))")

    for (g, path) in cps
        restarted = load_checkpoint(path, cfg)
        run_smc!(restarted, s_obs; T=4, schedule=QuantileSchedule(0.5),
                 checkpoint_dir=nothing)
        tag = "gen $g restart"
        check!(r, "$tag: θ equals uninterrupted", restarted.θ == full.θ;
               detail="gen $g θ differs")
        check!(r, "$tag: w equals uninterrupted", restarted.w == full.w;
               detail="gen $g w differs")
        check!(r, "$tag: dists equals uninterrupted", restarted.dists == full.dists;
               detail="gen $g dists differs")
        check!(r, "$tag: ε equals uninterrupted", restarted.ε == full.ε;
               detail="gen $g ε differs")
        check!(r, "$tag: ESS equals uninterrupted", restarted.ess == full.ess;
               detail="gen $g ess differs")
        check!(r, "$tag: n_attempts equals uninterrupted",
               restarted.n_attempts == full.n_attempts;
               detail="gen $g n_attempts differs")
        check!(r, "$tag: n_resamples equals uninterrupted",
               restarted.n_resamples == full.n_resamples;
               detail="gen $g n_resamples differs")
    end

    # ------------------------------------------------------------------
    # 3c. Corrupt checkpoint is rejected (throws on load)
    # ------------------------------------------------------------------
    corrupt_dir = mktempdir()
    # Write garbage bytes to a checkpoint-like filename
    corrupt_path = joinpath(corrupt_dir, "state_gen_001.jls")
    write(corrupt_path, b"this is not a valid serialization payload at all")
    corrupted = false
    try
        load_checkpoint(corrupt_path, cfg)
    catch
        corrupted = true
    end
    check!(r, "Corrupt checkpoint rejected (load throws)", corrupted;
           detail="load_checkpoint did not throw")

    # ------------------------------------------------------------------
    # 3d. Simulator exception propagates (not converted to numeric penalty)
    # ------------------------------------------------------------------
    boom_sim = ContinuousSimulator(
        model_builder = θ -> θ,
        evolve = (problem, rng) -> error("simulated simulator failure: boom"),
        summary = identity,
    )
    cfg_boom = ABCSMCConfig(cfg.prior, cfg.sample_prior, identity, cfg.distance,
                            GaussianKernel(), cfg.Σ, boom_sim)
    propagated = nothing
    try
        run_smc(cfg_boom, s_obs; N=10, T=2, base_seed=base_seed,
                schedule=FixedSchedule([2.0, 1.0]))
    catch e
        propagated = e
    end
    check!(r, "Simulator exception propagates", propagated !== nothing;
           detail="no exception thrown")
    if propagated !== nothing
        check!(r, "Propagated error is the simulator's own error",
           occursin("simulated simulator failure: boom", sprint(showerror, propagated));
           detail="message=$(sprint(showerror, propagated))")
    end

    # ------------------------------------------------------------------
    # 3e. Serial vs Threads.@spawn → identical results
    # ------------------------------------------------------------------
    serial_result = run_smc(cfg, s_obs; N=50, T=3, base_seed=base_seed,
                            schedule=FixedSchedule([2.0, 1.0, 0.5]))
    if Threads.nthreads() > 1
        spawned_result = nothing
        try
            t = Threads.@spawn run_smc(cfg, s_obs; N=50, T=3, base_seed=base_seed,
                                       schedule=FixedSchedule([2.0, 1.0, 0.5]))
            spawned_result = wait(t)
        catch e
            spawned_result = e
        end
        check!(r, "Threads.@spawn run completes (no exception)",
               spawned_result isa ABCSMCState; detail="spawn failed")
        if spawned_result isa ABCSMCState
            check!(r, "Serial vs spawned: θ identical", serial_result.θ == spawned_result.θ)
            check!(r, "Serial vs spawned: w identical", serial_result.w == spawned_result.w)
            check!(r, "Serial vs spawned: dists identical", serial_result.dists == spawned_result.dists)
            check!(r, "Serial vs spawned: ε identical", serial_result.ε == spawned_result.ε)
            check!(r, "Serial vs spawned: gen identical", serial_result.gen == spawned_result.gen)
        end
    else
        # Single-threaded: spawn test not applicable (no true parallelism possible)
    end

    return r
end
