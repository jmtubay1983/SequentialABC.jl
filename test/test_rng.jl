using SequentialABC
using Test
using Random

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

    # Seed is an Integer (hash returns UInt64 on 64-bit)
    s = make_particle_seed(1, 1, 1, 0, base)
    @test s isa Integer
end
