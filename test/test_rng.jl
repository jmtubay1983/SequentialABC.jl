using SequentialABC
using Test
using Random

const SEED_SCHEMA_VERSION = SequentialABC.SEED_SCHEMA_VERSION

@testset "Deterministic RNG" begin
    base = 1234

    # Same arguments → identical seed
    @test make_particle_seed(1, 2, 3, 4, base) == make_particle_seed(1, 2, 3, 4, base)

    # Different base → different seed
    @test make_particle_seed(1, 2, 3, 4, base) != make_particle_seed(1, 2, 3, 4, base + 1)

    # Different attempt → different seed (essential for redraw-until-accepted)
    @test make_particle_seed(1, 1, 1, 0, base) != make_particle_seed(1, 1, 2, 0, base)

    # Different m role → different seed (no collision across roles)
    @test make_particle_seed(1, 1, 1, 0, base) != make_particle_seed(1, 1, 1, -1, base)
    @test make_particle_seed(1, 1, 1, 0, base) != make_particle_seed(1, 1, 1, 1, base)
    @test make_particle_seed(1, 1, 1, 0, base) != make_particle_seed(1, 0, 0, -2, base)

    # Seed is an Integer (SplitMix64 returns UInt64 on 64-bit)
    s = make_particle_seed(1, 1, 1, 0, base)
    @test s isa Integer
end

@testset "Version-stable seed derivation (pinned literals)" begin
    # These literal UInt64 values were recorded with Julia 1.13.2 and
    # SEED_SCHEMA_VERSION = 1. Any change to the seed-derivation mixing
    # function will change these values and fail this test — which is the
    # point: the version guard in load_checkpoint will then reject old
    # checkpoints rather than silently replaying a different RNG stream.
    @test make_particle_seed(1, 2, 3, 4, 1234) === 0xd94d5be9919a6e00
    @test make_particle_seed(1, 1, 1, 0, 1234) === 0x56befc4a4e90e2af
    @test make_particle_seed(1, 1, 2, 0, 1234) === 0xc5028cce36b36a98
    @test make_particle_seed(1, 1, 1, -1, 1234) === 0x728eead1d96420e6
    @test make_particle_seed(1, 1, 1, -2, 1234) === 0x7013c3e2dbc5ff2d
    @test make_particle_seed(1, 1, 1,  1, 1234) === 0xa14823dff8b36701
    @test make_particle_seed(1, 0, 0, -2, 1234) === 0xbe78c6ca93153948
    @test make_particle_seed(2, 5, 1,  0, 9999) === 0x8d0cdc4f8a388e9b
    @test make_particle_seed(3, 10, 3,  5,   42) === 0xcbe4d4259a0a4ca6
end

@testset "Seed derivation: 10_000-block collision scan" begin
    base_seed = 20260101
    seen = Set{UInt64}()
    nblocks = 10_000
    count = 0
    for gen in 1:10, i in 1:100, attempt in 1:10, m in (0, -1, -2, 1)
        count >= nblocks && break
        push!(seen, make_particle_seed(gen, i, attempt, m, base_seed))
        count += 1
    end
    @test count == nblocks
    @test length(seen) == nblocks   # zero collisions
end

@testset "Seed derivation: role separation" begin
    base_seed = 20260101
    roles = (ancestor = 0, perturbation = -1, resample = -2, replicate = 3)
    role_seeds = [make_particle_seed(1, 1, 1, Int(m), base_seed) for m in values(roles)]
    @test length(Set(role_seeds)) == length(roles)
end

@testset "RNG output from a given seed (Julia 1.13.2)" begin
    # Pinned output of MersenneTwister(make_particle_seed(...)) — recorded on
    # Julia 1.13.2 with SEED_SCHEMA_VERSION = 1. Changing the seed derivation
    # changes these values and this test fails loudly.
    base_seed = 20260101
    seed = make_particle_seed(1, 1, 1, 0, base_seed)
    @test seed === 0x23d18732e67ad845

    rng = MersenneTwister(seed)
    expected_randn = [-0.5973149668669369,
                      -0.2837698275837855,
                      -1.334872592144628,
                       1.2226570913931873,
                       1.9222069908655817]
    for (k, ev) in enumerate(expected_randn)
        @test isapprox(randn(rng), ev; atol=1e-12)
    end

    rng2 = MersenneTwister(seed)
    expected_rand = [0.34637767549220455,
                     0.16808586938038705,
                     0.5314947129822831,
                     0.9552020349270096,
                     0.7540368512186331]
    for (k, ev) in enumerate(expected_rand)
        @test isapprox(rand(rng2), ev; atol=1e-12)
    end
end
