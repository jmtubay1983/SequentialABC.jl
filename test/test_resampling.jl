using SequentialABC
using Test
using Random

@testset "Systematic resampling" begin
    rng = MersenneTwister(1234)
    w = [0.7, 0.2, 0.1]
    N = length(w)

    idx = systematic_resample(w, rng)

    # Indices are valid
    @test length(idx) == N
    @test all(1 .<= idx .<= N)
    @test eltype(idx) == Int

    # Reproducible with the same seed
    rng2 = MersenneTwister(1234)
    @test systematic_resample(w, rng2) == idx

    # Different seed may give different indices (not guaranteed, but very likely)
    rng3 = MersenneTwister(9999)
    idx3 = systematic_resample(w, rng3)
    @test length(idx3) == N
    @test all(1 .<= idx3 .<= N)

    # Degenerate: point mass → always index 1
    w_point = [1.0, 0.0, 0.0]
    for k in 1:20
        @test all(systematic_resample(w_point, MersenneTwister(k)) .== 1)
    end

    # Uniform weights → each index selected roughly once (systematic is low-variance)
    w_uniform = fill(1.0 / 10, 10)
    for seed in 1:50
        idx_u = systematic_resample(w_uniform, MersenneTwister(seed))
        @test sort(idx_u) == collect(1:10)
    end
end
