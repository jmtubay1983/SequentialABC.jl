using SequentialABC
using Test
using Distributions
using Random
using Statistics
using LinearAlgebra

# ── tiny deterministic reference problem (same 2-D family as the other suites) ─
# θ ~ N(0, 2²I), s(θ) = θ + N(0, I), distance = Euclidean norm. Names carry a
# `_sum` suffix because every test file shares one Main namespace.
_prior_sum(θ) = sum(logpdf(Normal(0.0, 2.0), x) for x in θ)
_sample_prior_sum(rng) = 2.0 .* randn(rng, 2)
_distance_sum(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
_simulator_sum = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve = (θ, rng) -> θ .+ randn(rng, 2),
    summary = identity,
)
CFG_SUM = ABCSMCConfig(_prior_sum, _sample_prior_sum, identity, _distance_sum,
                       GaussianKernel(), [1.0 0.0; 0.0 1.0], _simulator_sum)
OBS_SUM = [1.0, -1.0]
SEED_SUM = 20260117

# A three-part "population" summary: counts, mean age, spatial variance. It takes
# a Vector{Float64} of ages, so it can be applied to the simulator output above.
_count_sum = FunctionSummary(pop -> length(pop), :count)
_mean_sum = FunctionSummary(pop -> mean(pop), :mean_age)
_var_sum = FunctionSummary(pop -> var(pop), :spatial_var)
POP_SUM = CompositeSummary([_count_sum, _mean_sum, _var_sum], :population)

# A summary that declares two keys but returns one, used to pin the contract
# that declared keys are enforced before values are combined or projected.
struct _WrongKeysSummary <: AbstractSummary end
SequentialABC.summary_keys(::_WrongKeysSummary) = [:a, :b]
SequentialABC.compute_summary(::_WrongKeysSummary, out) = (a = 1.0,)

@testset "summary statistics framework" begin

    # ── FunctionSummary ──────────────────────────────────────────────────────
    @testset "FunctionSummary wraps a function under its name" begin
        s = FunctionSummary(x -> 2x, :double)
        @test summary_keys(s) == [:double]
        @test compute_summary(s, 3.0) == (double = 6.0,)
        @test compute_summary(s, 3.0) isa NamedTuple{(:double,),Tuple{Float64}}
        @test s.fn === s.fn            # the wrapped function is stored as-is
        @test s.name === :double

        # a NamedTuple result is kept structured until a composite flattens it
        nested = FunctionSummary(x -> (mean = x, sd = 0.5), :age)
        @test compute_summary(nested, 1.0) == (age = (mean = 1.0, sd = 0.5),)

        @test_throws ArgumentError FunctionSummary(x -> x, Symbol(""))
    end

    # ── CompositeSummary ─────────────────────────────────────────────────────
    @testset "CompositeSummary flattens in part order" begin
        @test summary_keys(POP_SUM) == [:count, :mean_age, :spatial_var]
        pop = [1.0, 2.0, 4.0, 8.0]
        nt = compute_summary(POP_SUM, pop)
        @test collect(keys(nt)) == [:count, :mean_age, :spatial_var]   # order preserved
        @test nt.count == 4
        @test nt.mean_age == 3.75
        @test nt.spatial_var ≈ var(pop)
        @test length(POP_SUM.summaries) == 3
        @test POP_SUM.summaries[1] === _count_sum

        # a Vector{FunctionSummary} literal widens to Vector{AbstractSummary}
        @test POP_SUM.summaries isa Vector{AbstractSummary}

        # nested keys are prefixed with the part's name when flattened
        mixed = CompositeSummary([FunctionSummary(x -> (mean = 2.0, sd = 1.0), :age),
                                  FunctionSummary(x -> 1.0, :mass)], :mixed)
        @test summary_keys(mixed) == [:age, :mass]      # declared keys are the part names
        @test compute_summary(mixed, nothing) == (age_mean = 2.0, age_sd = 1.0, mass = 1.0)

        # duplicate declared keys are rejected when the composite is built
        @test_throws ArgumentError CompositeSummary([_count_sum, FunctionSummary(x -> x, :count)],
                                                    :dup)
        # …and collisions that only appear after flattening are caught at compute time
        late_clash = CompositeSummary([FunctionSummary(x -> (mean = 1.0,), :age),
                                       FunctionSummary(x -> 0.0, :age_mean)], :late)
        @test summary_keys(late_clash) == [:age, :age_mean]     # statically distinct
        @test_throws ArgumentError compute_summary(late_clash, nothing)

        @test_throws ArgumentError CompositeSummary(AbstractSummary[], :empty)
        @test_throws ArgumentError CompositeSummary([_count_sum], Symbol(""))

        # summaries are scalars in broadcasting, so dotted calls map over outputs
        @test Base.broadcastable(POP_SUM) isa Ref
        dotted = compute_summary.(POP_SUM, [pop, 2 .* pop])
        @test dotted isa Vector{<:NamedTuple}
        @test dotted[1] == compute_summary(POP_SUM, pop)
        @test dotted[2].count == 4
    end

    @testset "compose splices composites" begin
        c = compose(_count_sum, _mean_sum)
        @test c isa CompositeSummary
        @test c.name === :composed
        @test summary_keys(c) == [:count, :mean_age]

        # right-nested and left-nested compositions give the same flat key order
        left = compose(compose(_count_sum, _mean_sum), _var_sum)
        right = compose(_count_sum, compose(_mean_sum, _var_sum))
        @test summary_keys(left) == summary_keys(right) ==
              [:count, :mean_age, :spatial_var]
        @test length(left.summaries) == 3
        @test length(right.summaries) == 3

        named = compose(_count_sum, _mean_sum; name = :sizes)
        @test named.name === :sizes
        @test_throws ArgumentError compose(_count_sum, _count_sum)   # duplicate :count
    end

    # ── scale ────────────────────────────────────────────────────────────────
    @testset "scale multiplies selected keys" begin
        pop = [1.0, 2.0, 4.0, 8.0]
        base = compute_summary(POP_SUM, pop)

        sc = scale(POP_SUM, Dict(:count => 1e-3))
        @test sc isa ScaledSummary
        @test sc.name === :population_scaled
        @test summary_keys(sc) == summary_keys(POP_SUM)       # keys unchanged
        scaled = compute_summary(sc, pop)
        @test scaled.count ≈ 1e-3 * base.count
        @test scaled.mean_age == base.mean_age                # unlisted key untouched
        @test scaled.spatial_var == base.spatial_var

        # several keys, integer factors, and a subset all work
        both = compute_summary(scale(POP_SUM, Dict(:count => 2, :mean_age => 0.5)), pop)
        @test both.count == 2 * base.count
        @test both.mean_age == 0.5 * base.mean_age

        # unknown keys are a caller-owned error, not a silent no-op
        @test_throws ArgumentError scale(POP_SUM, Dict(:coutn => 1.0))

        # an empty set of factors is valid and changes nothing
        unscaled = compute_summary(
            ScaledSummary(_count_sum, Dict{Symbol,Float64}(), :count_passthrough), pop)
        @test unscaled == (count = length(pop),)

        # scaled summaries compose with the rest of the framework
        comp = CompositeSummary([scale(_count_sum, Dict(:count => 1e-3)), _mean_sum], :scaled_comp)
        @test summary_keys(comp) == [:count, :mean_age]
        @test compute_summary(comp, pop).count ≈ 1e-3 * base.count
    end

    # ── PCA ──────────────────────────────────────────────────────────────────
    @testset "pca_reduce recovers principal components" begin
        rng = MersenneTwister(20260117)
        d1 = [3.0, 4.0] ./ 5.0                 # dominant direction, unit length
        d2 = [-4.0, 3.0] ./ 5.0                # orthogonal, unit length
        n = 200
        # 10:1 scale ratio along d1:d2 (100:1 in variance), mean shifted
        Z = [(10.0 * randn(rng)) .* d1 .+ randn(rng) .* d2 .+ [2.0, -1.0] for _ in 1:n]
        X = reduce(vcat, permutedims.(Z))      # n × 2 matrix of summary vectors

        base2 = CompositeSummary([FunctionSummary(v -> v[1], :a),
                                  FunctionSummary(v -> v[2], :b)], :base2)
        proj = pca_reduce(base2, X, 2)
        @test proj isa PCAProjection
        @test summary_keys(proj) == [:pc1, :pc2]
        @test size(proj.components) == (2, 2)
        @test proj.name === :pca

        # components align with the true axes (up to the canonical sign)
        @test abs(dot(proj.components[1, :], d1)) ≈ 1.0 atol = 1e-3
        @test abs(dot(proj.components[2, :], d2)) ≈ 1.0 atol = 1e-3
        @test proj.explained[1] > 0.97
        @test proj.explained[1] > proj.explained[2]
        @test sum(proj.explained) ≈ 1.0
        @test proj.mean ≈ vec(sum(X; dims = 1)) ./ n

        # canonical sign: the largest-|value| entry of every component is positive
        for j in 1:2
            v = proj.components[j, :]
            @test v[argmax(abs.(v))] > 0
        end

        # a refit on the same data is bit-identical (deterministic, no RNG)
        @test pca_reduce(base2, X, 2).components == proj.components

        # the projection is callable and transform is the same map, by name
        @test !(:transform in names(SequentialABC))   # deliberately unexported
        S = proj(X)
        @test S ≈ SequentialABC.transform(proj, X)
        @test size(S) == (n, 2)
        @test S ≈ (X .- proj.mean') * proj.components'
        for i in (1, 5, 200)
            @test namedtuple_to_vector(compute_summary(proj, Z[i])) ≈ S[i, :]
        end

        # k < p keeps the leading components only
        proj1 = pca_reduce(base2, X, 1)
        @test summary_keys(proj1) == [:pc1]
        @test proj1.components ≈ proj.components[1:1, :]
        @test proj1.explained[1] ≈ proj.explained[1]
        @test size(proj1(X)) == (n, 1)

        # center = false leaves the origin fixed
        raw = pca_reduce(base2, X, 2; center = false)
        @test raw.mean == zeros(2)
        @test raw(X) ≈ X * raw.components'

        # PCA is a summary: it composes and feeds the same compute/summary_keys API
        comp = CompositeSummary([proj, FunctionSummary(v -> 1.0, :extra)], :with_pca)
        @test summary_keys(comp) == [:pc1, :pc2, :extra]
        @test compute_summary(comp, Z[1]).pc1 ≈ S[1, 1]

        # rank-deficient data: the second component just captures nothing
        collinear = hcat(collect(1.0:10.0), 2 .* collect(1.0:10.0))
        lowrank = pca_reduce(base2, collinear, 2)
        @test lowrank.explained[1] ≈ 1.0
        @test lowrank.explained[2] ≤ 1e-12
        @test all(isfinite, lowrank.components)

        # constant data: no variance to explain, but no NaNs either
        flat = pca_reduce(base2, ones(5, 2), 2)
        @test flat.explained == zeros(2)
        @test all(isfinite, flat.components)
    end

    @testset "pca_reduce validates its inputs" begin
        base2 = CompositeSummary([FunctionSummary(v -> v[1], :a),
                                  FunctionSummary(v -> v[2], :b)], :base2)
        X = [1.0 0.0; 2.0 1.0; 3.0 2.0]

        @test_throws ArgumentError pca_reduce(base2, X, 4)        # k > min(n, p)
        @test_throws ArgumentError pca_reduce(base2, X, 0)
        @test_throws ArgumentError pca_reduce(base2, ones(3, 3), 1)   # 3 columns, 2 keys
        @test_throws ArgumentError pca_reduce(POP_SUM, X, 1)      # 2 columns, 3 keys

        proj = pca_reduce(base2, X, 1)
        @test_throws ArgumentError proj(ones(4, 3))
        @test_throws ArgumentError SequentialABC.transform(proj, hcat(X, X))
    end

    # ── aggregate over replicates ────────────────────────────────────────────
    @testset "aggregate summarises replicates" begin
        pop = [1.0, 2.0, 4.0, 8.0]
        reps = [pop, 2 .* pop, 3 .* pop]

        agg = aggregate(reps, POP_SUM)
        @test agg isa NamedTuple
        # aggregator-major key order: all means, then all standard deviations
        @test collect(keys(agg)) == [:mean_count, :mean_mean_age, :mean_spatial_var,
                                     :sd_count, :sd_mean_age, :sd_spatial_var]
        @test agg.mean_count == 4.0
        @test agg.sd_count == 0.0
        @test agg.mean_mean_age ≈ (3.75 + 7.5 + 11.25) / 3
        @test agg.mean_spatial_var ≈ (var(pop) + var(2 .* pop) + var(3 .* pop)) / 3
        @test agg.sd_mean_age ≈ std([3.75, 7.5, 11.25])

        # mean+sd of each key is what a distance function then sees as a vector
        v = namedtuple_to_vector(agg)
        @test length(v) == 6
        @test v[1:3] == [agg.mean_count, agg.mean_mean_age, agg.mean_spatial_var]
        @test v[4:6] == [agg.sd_count, agg.sd_mean_age, agg.sd_spatial_var]

        # custom aggregators, including a single one and quantiles
        only_mean = aggregate(reps, POP_SUM; aggs = (mean = mean,))
        @test collect(keys(only_mean)) == [:mean_count, :mean_mean_age, :mean_spatial_var]

        ranged = aggregate(reps, POP_SUM; aggs = (lo = minimum, hi = maximum))
        @test ranged.lo_count == 4.0
        @test ranged.hi_mean_age ≈ 11.25

        @test_throws ArgumentError aggregate(Vector{Vector{Float64}}(), POP_SUM)
        @test_throws ArgumentError aggregate(reps, POP_SUM; aggs = NamedTuple())
    end

    @testset "declared keys are enforced" begin
        # a summary that declares two keys but returns one breaks the contract
        wrong = _WrongKeysSummary()
        @test summary_keys(wrong) == [:a, :b]
        @test_throws ArgumentError aggregate([1.0], wrong)
        @test_throws ArgumentError compute_summary(pca_reduce(wrong, ones(3, 2), 1), 1.0)

        # out-of-order keys are a mismatch too: the vector layout must be fixed
        @test_throws ArgumentError validate_summary_keys((b = 1.0, a = 2.0), [:a, :b])
        @test_throws ArgumentError validate_summary_keys((a = 1.0,), POP_SUM)
        @test validate_summary_keys((count = 1.0, mean_age = 2.0, spatial_var = 3.0),
                                    POP_SUM) == [:count, :mean_age, :spatial_var]
        @test validate_summary_keys(compute_summary(POP_SUM, [1.0, 2.0]), POP_SUM) ==
              [:count, :mean_age, :spatial_var]
    end

    # ── helpers ──────────────────────────────────────────────────────────────
    @testset "namedtuple helpers" begin
        @test flatten_namedtuple((a = 1, b = (x = 2, y = 3))) == (a = 1, b_x = 2, b_y = 3)
        @test flatten_namedtuple((a = (b = (c = 4,),), d = 5.0)) == (a_b_c = 4, d = 5.0)
        @test flatten_namedtuple((a = 1.0,)) == (a = 1.0,)
        # a flattened prefix that collides with an existing key is an error
        @test_throws ArgumentError flatten_namedtuple((b = (x = 2,), b_x = 1))

        @test namedtuple_to_vector((a = 1, b = 2.5)) == [1.0, 2.5]
        @test namedtuple_to_vector((a = 1, b = 2.5)) isa Vector{Float64}
        @test namedtuple_to_vector(NamedTuple()) == Float64[]
        err = try
            namedtuple_to_vector((x = [1.0, 2.0],))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("x", sprint(showerror, err))          # the message names the key
        @test occursin("Real", sprint(showerror, err))

        @test vector_to_namedtuple([1.0, 2.0], [:a, :b]) == (a = 1.0, b = 2.0)
        @test vector_to_namedtuple([1.0, 2.0], [:a, :b]) isa
              NamedTuple{(:a, :b),Tuple{Float64,Float64}}
        @test namedtuple_to_vector(vector_to_namedtuple([3.0, 4.0], [:p, :q])) == [3.0, 4.0]
        @test_throws ArgumentError vector_to_namedtuple([1.0], [:a, :b])
    end

    # ── integration with the samplers ────────────────────────────────────────
    @testset "composite summary drives run_inference end-to-end" begin
        # The simulator returns R = 3 replicate populations per θ; the summary
        # aggregates them, so replicate mean *and* sd take part in the distance.
        R = 3
        reps_sim = ContinuousSimulator(
            model_builder = θ -> θ,
            evolve = (θ, rng) -> [θ[1] .+ θ[2] .* randn(rng, 20) for _ in 1:R],
            summary = reps -> namedtuple_to_vector(aggregate(reps, POP_SUM)),
        )
        cfg = ABCSMCConfig(_prior_sum, _sample_prior_sum, identity, _distance_sum,
                           GaussianKernel(), [1.0 0.0; 0.0 1.0], reps_sim)

        obs_reps = [[1.0, 2.0, 1.5, 1.2], [0.9, 2.1, 1.4, 1.3], [1.1, 1.9, 1.6, 1.1]]
        obs_nt = aggregate(obs_reps, POP_SUM)
        @test collect(keys(obs_nt)) == [:mean_count, :mean_mean_age, :mean_spatial_var,
                                       :sd_count, :sd_mean_age, :sd_spatial_var]
        s_obs = namedtuple_to_vector(obs_nt)
        @test length(s_obs) == 2 * length(summary_keys(POP_SUM))

        st = run_inference(:smc, cfg, s_obs; N = 40, T = 3, base_seed = SEED_SUM)
        @test st.gen == 3
        @test st.ε < Inf
        @test sum(st.w) ≈ 1.0
        @test all(θ -> all(isfinite, θ), st.θ)
        @test maximum(st.dists) <= st.ε            # every particle passed the last ε

        # deterministic: same seed, same particles and weights
        again = run_inference(:smc, cfg, s_obs; N = 40, T = 3, base_seed = SEED_SUM)
        @test again.θ == st.θ
        @test again.w == st.w
        @test again.dists == st.dists
    end

    @testset "PCA-reduced summary drives run_inference end-to-end" begin
        base2 = CompositeSummary([FunctionSummary(v -> v[1], :a),
                                  FunctionSummary(v -> v[2], :b)], :base2)

        # pilot ensemble on the prior predictive distribution (fixed seed):
        # one raw simulator output per prior draw, one summary vector per row
        rng = MersenneTwister(4242)
        pilot = [_simulator_sum.evolve(_simulator_sum.model_builder(_sample_prior_sum(rng)), rng)
                 for _ in 1:60]
        X = reduce(vcat, permutedims.(namedtuple_to_vector.(compute_summary.(base2, pilot))))
        proj = pca_reduce(base2, X, 2)
        @test size(proj(X)) == (60, 2)

        reduced = ContinuousSimulator(model_builder = θ -> θ,
                                      evolve = _simulator_sum.evolve,
                                      summary = raw -> namedtuple_to_vector(compute_summary(proj, raw)))
        cfg = ABCSMCConfig(_prior_sum, _sample_prior_sum, identity, _distance_sum,
                           GaussianKernel(), [1.0 0.0; 0.0 1.0], reduced)
        s_obs = namedtuple_to_vector(compute_summary(proj, OBS_SUM))

        st = run_inference(:smc, cfg, s_obs; N = 40, T = 3, base_seed = SEED_SUM)
        @test st.gen == 3
        @test all(θ -> all(isfinite, θ), st.θ)
        @test maximum(st.dists) <= st.ε

        # the same composite works with the other engines
        chain = run_inference(:mcmc, cfg, s_obs; n_samples = 20, ε = 3.0,
                              base_seed = SEED_SUM)
        @test length(chain) == 21                     # n_samples + the initial state
        @test all(θ -> all(isfinite, θ), chain.state.θ)

        pmc = run_inference(:pmc, cfg, s_obs; N = 40, T = 3, εs = [4.0, 3.0, 2.0],
                            base_seed = SEED_SUM)
        @test pmc.gen == 3
        @test maximum(pmc.dists) <= pmc.ε
    end
end
