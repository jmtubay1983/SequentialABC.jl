# examples/summaries_example.jl
#
# Composable, named summary statistics end to end:
#
#   1. a composite summary over a "population" simulator output
#   2. replicate aggregation (mean + sd) for the observed summary
#   3. PCA compression of a high-dimensional summary, fitted on a pilot ensemble
#   4. all three engines driven by the same summary through run_inference
#
# Everything is deterministic: fixed seeds, no global RNG, no files written.

using SequentialABC
using Distributions
using Printf
using Random
using Statistics

const SEED = 20260117

# --- 1. Named summaries of one simulator output ---------------------------------
# The simulator returns a population (`Vector{Float64}` of ages); each summary
# pulls one interpretable number out of it.
pop_summary = CompositeSummary([
    FunctionSummary(pop -> length(pop), :count),        # population size
    FunctionSummary(pop -> mean(pop),   :mean_age),     # average age
    FunctionSummary(pop -> var(pop),    :spatial_var),  # spatial spread
], :population)

@printf("composite summary keys = %s\n", Tuple(summary_keys(pop_summary)))

pop = [1.0, 2.0, 4.0, 8.0]
named = compute_summary(pop_summary, pop)
@printf("compute_summary(pop)   = count %d, mean_age %.3f, spatial_var %.3f\n",
        named.count, named.mean_age, named.spatial_var)
@printf("distance vector order  = %s\n\n", namedtuple_to_vector(named))

# --- 2. Replicates: aggregates with mean *and* sd --------------------------------
reps = [pop, 2 .* pop, 3 .* pop]                     # replicate runs of one θ
agg = aggregate(reps, pop_summary)
@printf("aggregate keys         = %s\n", Tuple(keys(agg)))
@printf("replicate means        = %.3f, %.3f, %.3f\n",
        agg.mean_count, agg.mean_mean_age, agg.mean_spatial_var)
@printf("replicate sds          = %.3f, %.3f, %.3f\n\n",
        agg.sd_count, agg.sd_mean_age, agg.sd_spatial_var)

# The sampler's own M replicates average internally; a simulator that returns a
# vector of replicate populations lets the summary compare sd as well.
rep_simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve        = (θ, rng) -> [θ[1] .+ θ[2] .* randn(rng, 20) for _ in 1:3],
    summary       = reps -> namedtuple_to_vector(aggregate(reps, pop_summary)),
)

# Synthetic observations: three replicate populations from a "true" θ, on their
# own seed (the trick examples/agents_schelling.jl uses). :count is always 20
# here, so sd_count = 0 says out loud that it carries no information.
obs_rng   = MersenneTwister(7)
TRUE_θ    = [1.0, 0.5]
obs_reps  = [TRUE_θ[1] .+ TRUE_θ[2] .* randn(obs_rng, 20) for _ in 1:3]
s_obs_rep = namedtuple_to_vector(aggregate(obs_reps, pop_summary))
@printf("observed summary (6-D) = %s\n\n", round.(s_obs_rep; digits = 4))

# --- 3. PCA compression of the same summary --------------------------------------
# A pilot ensemble on the prior predictive distribution gives the data PCA needs.
base = CompositeSummary([
    FunctionSummary(v -> v[1], :a),
    FunctionSummary(v -> v[2], :b),
], :two_keys)

pilot_sim = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve        = (θ, rng) -> θ .+ randn(rng, 2),
    summary       = identity,
)
pilot_rng = MersenneTwister(SEED)
sample_prior(rng) = 2.0 .* randn(rng, 2)
pilot = [pilot_sim.evolve(pilot_sim.model_builder(sample_prior(pilot_rng)), pilot_rng)
         for _ in 1:200]
X = reduce(vcat, permutedims.(namedtuple_to_vector.(compute_summary.(base, pilot))))

proj = pca_reduce(base, X, 2)
@printf("PCA: components = %s\n", summary_keys(proj))
@printf("PCA: explained  = %.4f, %.4f\n", proj.explained[1], proj.explained[2])
# callable form; SequentialABC.transform(proj, X) is the same map by name
@printf("PCA: proj(X)    = %d rows x %d cols\n\n", size(proj(X))...)

# --- 4. Drive all three engines from one summary ---------------------------------
prior(θ) = sum(logpdf(Normal(0.0, 2.0), x) for x in θ)
distance(s, s_obs) = sqrt(sum(abs2, s .- s_obs))
cfg = ABCSMCConfig(prior, sample_prior, identity, distance,
                   GaussianKernel(), [1.0 0.0; 0.0 1.0], rep_simulator)

st = run_inference(:smc, cfg, s_obs_rep; N = 50, T = 3, base_seed = SEED)
@printf("run_inference(:smc)  → gen %d, ε = %.3f, ESS = %.1f, resamples = %d\n",
        st.gen, st.ε, st.ess, st.n_resamples)

# The same composite inside a PCA-reduced pipeline, on the 2-D problem
pca_simulator = ContinuousSimulator(
    model_builder = θ -> θ,
    evolve        = pilot_sim.evolve,
    summary       = raw -> namedtuple_to_vector(compute_summary(proj, raw)),
)
cfg_pca = ABCSMCConfig(prior, sample_prior, identity, distance,
                       GaussianKernel(), [1.0 0.0; 0.0 1.0], pca_simulator)
s_obs_pca = namedtuple_to_vector(compute_summary(proj, [1.0, -1.0]))

st_pca = run_inference(:pmc, cfg_pca, s_obs_pca; N = 50, T = 3,
                       εs = [4.0, 3.0, 2.0], base_seed = SEED)
@printf("run_inference(:pmc)  → gen %d, ε = %.3f, ESS = %.1f, resamples = %d\n",
        st_pca.gen, st_pca.ε, st_pca.ess, st_pca.n_resamples)

chain = run_inference(:mcmc, cfg_pca, s_obs_pca; n_samples = 200, ε = 3.0,
                      base_seed = SEED)
@printf("run_inference(:mcmc) → %d samples, ε = %.3f, accept rate = %.3f\n",
        length(chain), chain.ε, acceptance_rate(chain))

# Declared keys are a contract: the key order fixes the layout of the distance
# vector, so two parts of a composite may not claim the same key.
try
    CompositeSummary([FunctionSummary(length, :count),
                      FunctionSummary(sum,    :count)], :clash)
catch err
    @printf("duplicate-key check  → %s: %s\n", typeof(err), err.msg)
end
