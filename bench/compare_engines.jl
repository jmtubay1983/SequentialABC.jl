# bench/compare_engines.jl
#
# Controlled comparison of the ABC-SMC and ABC-MCMC engines against the exact
# posterior of a Gaussian reference problem:
#
#   julia --project=. bench/compare_engines.jl                      # standard run
#   julia --project=. bench/compare_engines.jl --mode tiny          # CI-sized smoke run
#   julia --project=. bench/compare_engines.jl --mode full --out bench/results
#
# Rows
#   SMC  fixed (Σ0)              — badly scaled, hand-picked
#   SMC  fixed (oracle 2Σ*)      — Σ matched to the *exact* posterior: the best a
#                                  hand-tuned fixed kernel can do
#   SMC  adaptive (batch)        — Σ re-estimated from the population, one call
#   SMC  adaptive (streaming)    — … one particle at a time, committed per generation
#   MCMC fixed (Σ0), MCMC fixed (oracle), MCMC adaptive
#
# Every SMC row is run at the *same* prescribed tolerance ladder, and every MCMC
# row at the same ε, so the rows are scored at identical ABC targets and differ
# only in their kernel. Seeds are pinned: statistics are identical across
# repetitions and only the wall clock varies (the minimum is reported).

include(joinpath(@__DIR__, "bench_utils.jl"))

# ── modes ────────────────────────────────────────────────────────────────────
#
# N     — SMC population size            n_mcmc — ABC-MCMC snapshots
# T     — SMC generations                reps   — repetitions (timing only)

const MODES = Dict(
    "tiny"     => (N = 150,  T = 4, q = 0.6, ε0 = 4.0, n_mcmc = 400,    reps = 1),
    "standard" => (N = 400,  T = 6, q = 0.6, ε0 = 4.0, n_mcmc = 2_000,  reps = 1),
    "full"     => (N = 800,  T = 8, q = 0.6, ε0 = 4.0, n_mcmc = 10_000, reps = 3),
)

function parse_args(args)
    mode = "standard"
    out = nothing
    reps_override = nothing
    d = 2
    seed = 42

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--mode"
            mode = args[i + 1]; i += 2
        elseif a == "--tiny"
            mode = "tiny"; i += 1
        elseif a == "--full"
            mode = "full"; i += 1
        elseif a == "--out"
            out = args[i + 1]; i += 2
        elseif a == "--reps"
            reps_override = parse(Int, args[i + 1]); i += 2
        elseif a == "--d"
            d = parse(Int, args[i + 1]); i += 2
        elseif a == "--seed"
            seed = parse(Int, args[i + 1]); i += 2
        else
            error("unknown argument `$a`; supported: --mode tiny|standard|full, " *
                  "--out DIR, --reps N, --d N, --seed N")
        end
    end

    haskey(MODES, mode) || error("unknown mode `$mode`; use tiny, standard or full")
    bench = merge(MODES[mode], (seed = seed,))
    reps_override === nothing || (bench = merge(bench, (reps = reps_override,)))
    return (mode = mode, out = out, d = d, seed = seed, bench = bench)
end

# ── result rows ──────────────────────────────────────────────────────────────

struct BenchRow
    name::String
    sims::Int
    seconds::Float64
    ess::Float64
    resamples::Int
    ε::Float64
    acceptance::Float64        # NaN for ABC-SMC rows
    rmse::Float64
    mc_se::Float64
    cov_err::Float64
    sigma11::Float64
    mean::Vector{Float64}
end

const COLUMNS = ["engine", "sims", "wall (s)", "ESS", "resamp", "ε", "accept",
                 "RMSE", "mc se", "cov err", "Σ₁₁"]

row_cells(r::BenchRow) = [r.name, string(r.sims), @sprintf("%.2f", r.seconds),
                          @sprintf("%.1f", r.ess), string(r.resamples),
                          @sprintf("%.3f", r.ε),
                          isnan(r.acceptance) ? "—" : @sprintf("%.3f", r.acceptance),
                          @sprintf("%.4f", r.rmse), @sprintf("%.4f", r.mc_se),
                          @sprintf("%.3f", r.cov_err), @sprintf("%.4f", r.sigma11)]

function print_table(rows::Vector{BenchRow})
    println(@sprintf("%-30s %8s %8s %8s %7s %7s %7s %8s %8s %8s %9s", COLUMNS...))
    println("─"^112)
    for r in rows
        println(@sprintf("%-30s %8d %8.2f %8.1f %7d %7.3f %7s %8.4f %8.4f %8.3f %9.4f",
                         r.name, r.sims, r.seconds, r.ess, r.resamples, r.ε,
                         isnan(r.acceptance) ? "—" : @sprintf("%.3f", r.acceptance),
                         r.rmse, r.mc_se, r.cov_err, r.sigma11))
    end
    println("─"^112)
    println("legend: sims = simulator calls, wall (s) = min over reps, " *
            "ESS = final population (SMC) / Geyer min-over-dims (MCMC),")
    println("        RMSE = posterior-mean error vs the exact posterior, " *
            "mc se = √(mean diag Σ̂ / ESS) = the noise floor of that RMSE,")
    println("        cov err = relative Frobenius error of the posterior covariance, " *
            "Σ₁₁ = the kernel this row actually used")
    return nothing
end

# ── running a row ────────────────────────────────────────────────────────────

function smc_row(name, p::GaussianProblem, ref, Σ, bench, ladder; adaptive::Bool = false,
                 streaming::Bool = false, kernel_rule = :beaumont)
    cfg = problem_config(p, Σ)
    if adaptive
        out, secs = repeat_min(
            () -> run_smc_adaptive(cfg, p; N = bench.N, T = bench.T, q = bench.q,
                                   ε0 = bench.ε0, seed = bench.seed, ladder = ladder,
                                   streaming = streaming, kernel_rule = kernel_rule),
            bench.reps)
        st, kernel = out.state, out.kernel
    else
        out, secs = repeat_min(
            () -> run_smc_fixed(cfg, p; N = bench.N, T = bench.T, q = bench.q,
                                ε0 = bench.ε0, seed = bench.seed, ladder = ladder),
            bench.reps)
        st, kernel = out.state, nothing
    end

    m = posterior_metrics(st, ref; sims = st.n_attempts, resamples = st.n_resamples,
                          seconds = secs)
    row = BenchRow(name, m.sims, m.seconds, m.ess, m.resamples, m.ε, NaN,
                   m.rmse, m.mc_se, m.cov_err, Σ[1, 1], m.mean)
    return row, st, kernel
end

function mcmc_row(name, p::GaussianProblem, ref, Σ, ε; n_samples::Int, seed::Int,
                  reps::Int)
    cfg = problem_config(p, Σ)
    out, secs = repeat_min(
        () -> run_mcmc_bench(cfg, p; ε = ε, n_samples = n_samples, seed = seed), reps)
    chain = out.chain
    m = posterior_metrics(chain.state, ref; sims = chain.n_simulations, resamples = 0,
                          seconds = secs, ess = chain_ess(chain))
    return BenchRow(name, m.sims, m.seconds, m.ess, 0, chain.ε,
                    acceptance_rate(chain), m.rmse, m.mc_se, m.cov_err, Σ[1, 1], m.mean)
end

# ── main ─────────────────────────────────────────────────────────────────────

function main()
    cli = parse_args(ARGS)
    bench = cli.bench
    p = GaussianProblem(; d = cli.d)
    ref = reference_posterior(p)
    d = p.d

    # Baseline covariances: a badly scaled hand guess, and the oracle derived
    # from the *exact* posterior (SMC perturbation = 2·Σ*, MCMC proposal =
    # 2.38²/d · Σ*).
    Σ0 = 4.0 .* Matrix{Float64}(I, d, d)
    Σ_oracle_smc = adaptive_scale(Matrix(ref.Σ), d; rule = :beaumont)
    Σ_oracle_mcmc = adaptive_scale(Matrix(ref.Σ), d; rule = :rosenthal)

    # Calibration: the tolerance ladder an adaptive run reaches at *twice* the
    # benchmark population (so it is well mixed and tight). Every SMC row below
    # then runs this same ladder — without it each row would pick its own ε and
    # the rows would not be scored at the same ABC targets.
    calib = run_smc_adaptive(problem_config(p, Σ0), p; N = 2 * bench.N, T = bench.T,
                             q = bench.q, ε0 = bench.ε0, seed = cli.seed,
                             streaming = false)
    ladder = calib.ladder

    params = @sprintf("N = %d, T = %d, q = %.2f, ε0 = %.1f, n_mcmc = %d, reps = %d, seed = %d",
                      bench.N, bench.T, bench.q, bench.ε0, bench.n_mcmc, bench.reps,
                      cli.seed)
    println(run_header(cli.mode, p, params))
    println("baselines   Σ0 = $(Σ0[1, 1])·I   " *
            "Σ_oracle_smc = $(round(Σ_oracle_smc[1, 1]; digits = 4))·I   " *
            "Σ_oracle_mcmc = $(round(Σ_oracle_mcmc[1, 1]; digits = 4))·I")
    println("ladder      ", join(round.(ladder; digits = 4), " → "),
            "  (calibrated by an adaptive run at N = $(2 * bench.N))")
    println()

    rows = BenchRow[]

    t0 = time_ns()
    r, st_fixed, _ = smc_row("SMC fixed (Σ0)", p, ref, Σ0, bench, ladder)
    push!(rows, r)
    r, _, _ = smc_row("SMC fixed (oracle 2Σ*)", p, ref, Σ_oracle_smc, bench, ladder)
    push!(rows, r)
    r, st_batch, k_batch = smc_row("SMC adaptive (batch)", p, ref, Σ0, bench, ladder;
                                   adaptive = true, streaming = false)
    push!(rows, r)
    r, st_stream, k_stream = smc_row("SMC adaptive (streaming)", p, ref, Σ0, bench, ladder;
                                     adaptive = true, streaming = true)
    push!(rows, r)

    # MCMC rows share the tolerance the SMC ladder ends at.
    ε_mcmc = ladder[end]

    # Pre-run adaptation of the chain's proposal: 2.38²/d · Cov(ABC population),
    # estimated once and frozen — the form that keeps the MH chain valid.
    k_prop = AdaptiveKernel(Σ0; rule = :rosenthal)
    update!(k_prop, st_batch.θ, st_batch.w)

    push!(rows, mcmc_row("MCMC fixed (Σ0)", p, ref, Σ0, ε_mcmc;
                         n_samples = bench.n_mcmc, seed = cli.seed, reps = bench.reps))
    push!(rows, mcmc_row("MCMC fixed (oracle)", p, ref, Σ_oracle_mcmc, ε_mcmc;
                         n_samples = bench.n_mcmc, seed = cli.seed, reps = bench.reps))
    push!(rows, mcmc_row("MCMC adaptive (pre-run)", p, ref, Matrix(k_prop.Σ), ε_mcmc;
                         n_samples = bench.n_mcmc, seed = cli.seed, reps = bench.reps))
    println("wall clock: $(round((time_ns() - t0) / 1e9; digits = 2)) s " *
            "(mode $(cli.mode), MCMC at ε = $(round(ε_mcmc; digits = 4)), " *
            "adaptive proposal Σ₁₁ = $(round(k_prop.Σ[1, 1]; digits = 4)))")
    println()

    print_table(rows)

    # The two adaptation paths must agree: same seeds, same population, same Σ.
    b, s = rows[3], rows[4]
    parity_dmean = maximum(abs.(b.mean .- s.mean))
    parity_ok = b.sims == s.sims && parity_dmean < 1e-6
    println()
    println("streaming ≡ batch: ", parity_ok ? "yes" : "NO",
            "  (sims $(b.sims) vs $(s.sims), max |Δ posterior mean| = ",
            parity_dmean, ")")
    println("kernel updates: batch $(k_batch.n_updates), streaming $(k_stream.n_updates); " *
            "streaming accumulator holds $(k_stream.est.n) particles")

    # What the kernel choice actually bought. Accuracy is deliberately *not* one
    # of these: every SMC row ran the same ladder, and the MCMC rows share ε, so
    # the rows target the same posterior and their RMSEs differ only by
    # Monte-Carlo noise — compare each RMSE against its `mc se`.
    println()
    println("derived")
    println(@sprintf("  SMC  adaptive vs Σ0        %5.2f× fewer simulations   (%d vs %d), same accuracy band",
                     rows[1].sims / rows[3].sims, rows[3].sims, rows[1].sims))
    println(@sprintf("  SMC  adaptive vs oracle    %5.2f× the oracle's cost     (%d vs %d)",
                     rows[3].sims / rows[2].sims, rows[3].sims, rows[2].sims))
    println(@sprintf("  MCMC adaptive vs Σ0        %5.2f× the ESS               (%.1f vs %.1f) at identical simulation cost (%d)",
                     rows[7].ess / rows[5].ess, rows[7].ess, rows[5].ess, rows[5].sims))
    println(@sprintf("  MCMC oracle-Σ* vs Σ0       %5.2f× the ESS               (%.1f vs %.1f)",
                     rows[6].ess / rows[5].ess, rows[6].ess, rows[5].ess))
    println("  (ABC-MCMC simulates exactly once per Metropolis step, so all MCMC rows " *
            "spend $(rows[5].sims) simulations; the kernel shows up in mixing, not in the count.)")

    # Soundness check: every engine's posterior mean must agree with the exact
    # posterior to within its own Monte-Carlo error. This is the statistically
    # meaningful accuracy claim — a *tighter* cross-row comparison is not
    # available at these sample sizes (see the mc se discussion above).
    scale = [r.rmse / max(r.mc_se, eps()) for r in rows]
    println()
    println("consistency: max RMSE / mc se over rows = ",
            round(maximum(scale); digits = 2), "  (must be ≤ 5)")

    if cli.out !== nothing
        mkpath(cli.out)
        md = IOBuffer()
        println(md, "### `bench/compare_engines.jl` — mode `$(cli.mode)`\n")
        println(md, "```")
        println(md, run_header(cli.mode, p, params))
        println(md, "baselines   Σ0 = $(Σ0[1, 1])·I   " *
                    "Σ_oracle_smc = $(round(Σ_oracle_smc[1, 1]; digits = 4))·I   " *
                    "Σ_oracle_mcmc = $(round(Σ_oracle_mcmc[1, 1]; digits = 4))·I")
        println(md, "ladder      ", join(round.(ladder; digits = 4), " → "))
        println(md, "```\n")
        println(md, markdown_table(COLUMNS, [row_cells(r) for r in rows]))
        println(md, "\n`sims` = simulator calls; `wall (s)` = minimum over $(bench.reps) " *
                    "repetition(s); `ESS` = final population (SMC) or Geyer " *
                    "minimum-over-dimensions (MCMC). All SMC rows ran the ladder above; " *
                    "MCMC rows share ε = $(round(ε_mcmc; digits = 4)).")
        println(md, "\nStreaming and batch adaptation agreed: **$(parity_ok ? "yes" : "NO")** " *
                    "(max |Δ posterior mean| = $(parity_dmean)).")
        write(joinpath(cli.out, "compare_engines.md"), String(take!(md)))

        csv = IOBuffer()
        println(csv, "engine,sims,wall_seconds,ess,resamples,epsilon,acceptance,rmse,mc_se,cov_err,sigma11")
        for r in rows
            println(csv, join((r.name, r.sims, r.seconds, r.ess, r.resamples, r.ε,
                               isnan(r.acceptance) ? "" : r.acceptance, r.rmse,
                               r.mc_se, r.cov_err, r.sigma11), ","))
        end
        write(joinpath(cli.out, "compare_engines.csv"), String(take!(csv)))
        println("artifacts written to $(cli.out)/compare_engines.{md,csv}")
    end

    # Hard checks: structural invariants, the kernel-efficiency claims, and
    # consistency of every row's mean with the exact posterior.
    @assert all(r -> isfinite(r.rmse) && isfinite(r.cov_err), rows)
    @assert parity_ok "streaming and batch adaptation diverged"
    @assert k_stream.n_updates == bench.T - 1 "streaming committed the wrong number of times"
    @assert k_stream.est.n == 0 "streaming accumulator was not reset by commit!"
    @assert all(r -> r.ε == ladder[end], rows[1:4]) "SMC rows did not run the prescribed ladder"
    @assert rows[3].sims < rows[1].sims "adaptive SMC needed no fewer simulations than Σ0"
    @assert rows[4].sims < rows[1].sims "streaming SMC needed no fewer simulations than Σ0"
    @assert rows[7].acceptance > rows[5].acceptance "adaptive MCMC proposal did not beat Σ0"
    @assert maximum(scale) <= 5.0 "a row's posterior mean is inconsistent with the exact posterior"
    println("✓ benchmark completed successfully")
    return nothing
end

main()
