using Random

"""
    make_particle_seed(gen::Int, i::Int, attempt::Int, m::Int, base::Int) -> Int

Deterministic seed for particle simulation.

The `attempt` counter is essential: without it, redraw‑until‑accepted would
replay the identical rejected proposal forever with a deterministic RNG.
The seed is a pure function of `(gen, i, attempt, m, base)` → full replay
across Slurm nodes and restarts.
"""
@inline function make_particle_seed(gen::Int, i::Int, attempt::Int, m::Int, base::Int)
    return hash((gen, i, attempt, m, base))
end

"""
    run_sim_once(cfg::ABCSMCConfig, gen::Int, i::Int, attempt::Int,
                 θ_raw; M::Int=1, base_seed::Int) -> Vector{Any}

Run M deterministic simulations for a single particle.

Each replicate m gets its own RNG seeded from `make_particle_seed`.
The transform is applied only at simulation time; the prior lives on the raw space.
"""
function run_sim_once(cfg::ABCSMCConfig, gen::Int, i::Int, attempt::Int,
                      θ_raw; M::Int=1, base_seed::Int)
    θ = cfg.transform(θ_raw)
    return [run_sim_once(cfg.simulator, θ,
                         MersenneTwister(make_particle_seed(gen, i, attempt, m, base_seed)))
            for m in 1:M]
end