using Random

"""
    SEED_SCHEMA_VERSION

Schema version for `make_particle_seed`. Bumped whenever the seed-derivation
mixing function changes, so that checkpoints saved under a different seed
schema can be detected and rejected on load.
"""
const SEED_SCHEMA_VERSION::UInt64 = 0x0000000000000001

"""
    _splitmix64_finalize(x::UInt64) -> UInt64

SplitMix64 finalizer (a.k.a. MurmurHash3 finalizer): avalanche an integer so
that small input changes produce large, well-distributed output changes.
All arithmetic is modulo 2^64 (UInt64 wraparound semantics).
"""
@inline function _splitmix64_finalize(x::UInt64)
    z = x ⊻ (x >> 30)
    z = (z * 0xbf58476d1ce4e5b9) % UInt64
    z = z ⊻ (z >> 27)
    z = (z * 0x94d049bb133111eb) % UInt64
    z = z ⊻ (z >> 31)
    return z
end

"""
    make_particle_seed(gen::Int, i::Int, attempt::Int, m::Int, base::Int) -> UInt64

Deterministic seed for particle simulation.

Uses a SplitMix64-style integer mixing function instead of Julia's `hash()`,
which is not guaranteed stable across Julia versions. The `SEED_SCHEMA_VERSION`
constant is mixed in so that checkpoints created under a different seed schema
can be detected and rejected.

The `attempt` counter is essential: without it, redraw‑until‑accepted would
replay the identical rejected proposal forever with a deterministic RNG.
The seed is a pure function of `(gen, i, attempt, m, base, SEED_SCHEMA_VERSION)`
→ full replay across Slurm nodes and restarts.
"""
@inline function make_particle_seed(gen::Int, i::Int, attempt::Int, m::Int, base::Int)
    z = SEED_SCHEMA_VERSION
    z = _splitmix64_finalize(z ⊻ reinterpret(UInt64, gen))
    z = _splitmix64_finalize(z ⊻ reinterpret(UInt64, i))
    z = _splitmix64_finalize(z ⊻ reinterpret(UInt64, attempt))
    z = _splitmix64_finalize(z ⊻ reinterpret(UInt64, m))
    z = _splitmix64_finalize(z ⊻ reinterpret(UInt64, base))
    return z
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
