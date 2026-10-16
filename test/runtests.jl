using Test

@testset "SequentialABC.jl" begin
    include("test_rng.jl")
    include("test_kernel.jl")
    include("test_weights.jl")
    include("test_resampling.jl")
    include("test_checkpoint.jl")
    include("test_smc.jl")
    include("regression_adjustment_tests.jl")
    include("test_examples.jl")
end
