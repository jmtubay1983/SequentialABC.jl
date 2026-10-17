using SequentialABC
using Test
using Aqua

@testset "Aqua" begin
    @testset "Ambiguities" begin
        Aqua.test_ambiguities(SequentialABC)
    end

    @testset "Piracies" begin
        Aqua.test_piracies(SequentialABC)
    end

    @testset "Dependencies compatibility" begin
        Aqua.test_deps_compat(Base.PkgId(Base.UUID("9ec30f26-5bd5-453c-9fb9-4a291198e4b8"), "SequentialABC"))
    end

    @testset "Julia compatibility" begin
        Aqua.test_julia_compat(Base.PkgId(Base.UUID("9ec30f26-5bd5-453c-9fb9-4a291198e4b8"), "SequentialABC"))
    end

    @testset "Project extras" begin
        Aqua.test_project_extras(SequentialABC)
    end

    @testset "Stale deps" begin
        Aqua.test_stale_deps(SequentialABC)
    end

    @testset "Undefined exports" begin
        Aqua.test_undefined_exports(SequentialABC)
    end

    @testset "Undocumented names" begin
        Aqua.test_undocumented_names(SequentialABC)
    end
end
