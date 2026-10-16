using Documenter
using SequentialABC

makedocs(
    sitename = "SequentialABC.jl",
    modules = [SequentialABC],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "Design Notes" => "design.md",
        "HPC / Slurm" => "hpc.md",
    ],
    repo = "https://github.com/tubay001/SequentialABC.jl.git",
)

deploydocs(
    repo = "github.com/tubay001/SequentialABC.jl.git",
    devbranch = "main",
)