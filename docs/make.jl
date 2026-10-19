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
        "Evaluation contracts" => "evaluation.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "ABC-MCMC" => "mcmc.md",
        "Population Monte Carlo" => "pmc.md",
        "Adaptive kernels" => "adaptive_kernels.md",
        "Kernel tuning" => "kernel_tuning.md",
        "Summary statistics" => "summaries.md",
        "Distances" => "distances.md",
        "Recipes" => "recipes.md",
        "Benchmarks" => "benchmarks.md",
        "Regression adjustment" => "regression_adjustment.md",
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