using Documenter
using SequentialABC

makedocs(
    sitename = "SequentialABC.jl",
    modules = [SequentialABC],
    format = Documenter.HTML(edit_link = "main"),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "Plotting" => "plots.md",
        "Design Notes" => "design.md",
        "HPC / Slurm" => "hpc.md",
    ],
    repo = "https://github.com/jmtubay1983/SequentialABC.jl.git",
)

deploydocs(
    repo = "github.com/jmtubay1983/SequentialABC.jl.git",
    devbranch = "main",
)