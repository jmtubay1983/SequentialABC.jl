using SequentialABC
using Test

@testset "Example models run end-to-end" begin
    # Continuous logistic-growth example (no external deps beyond the package)
    @testset "continuous logistic" begin
        include(joinpath(@__DIR__, "..", "examples", "continuous_logistic.jl"))
    end

    # Agents.jl Schelling example — requires Agents.jl, which is not a
    # dependency of SequentialABC. Run only when available.
    @testset "agents Schelling (optional)" begin
        agents_path = joinpath(@__DIR__, "..", "examples", "agents_schelling.jl")
        if isfile(agents_path)
            agents_available = try
                using Agents
                true
            catch
                false
            end
            if agents_available
                include(agents_path)
            else
                @test true  # Agents.jl not installed — agents example skipped
            end
        else
            @test true  # agents_schelling.jl not found — agents example skipped
        end
    end
end
