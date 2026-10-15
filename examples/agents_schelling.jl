# Example: Agents.jl (agent-based) backend
#
# Schelling segregation on a 20×20 grid: infer the agents' tolerance θ from
# the segregation index (mean fraction of same-group neighbors) observed after
# a fixed number of steps. The index is continuous and increases with
# tolerance over the useful range, making θ identifiable by ABC.
#
# Requires the Agents.jl example environment (kept separate so the heavy
# dependency stays out of the core package):
#   julia --project=examples/agents_env -e 'using Pkg; Pkg.instantiate()'
# Run from the package root:
#   julia --project=examples/agents_env examples/agents_schelling.jl

using SequentialABC
using Agents
using Distributions
using Random
using Printf

# --- agent type and rules (Agents.jl v7 API) --------------------------------
@agent struct Schelling(GridAgent{2})
    mood::Bool = false
    group::Int
end

function schelling_step!(agent, model)
    tol = model.tolerance
    n_same = 0
    n_total = 0
    for neighbor in nearby_agents(agent, model)
        n_total += 1
        n_same += agent.group == neighbor.group
    end
    frac = n_total == 0 ? 1.0 : n_same / n_total
    if frac >= tol
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end

const GRID     = (20, 20)
const N_AGENTS = 300
const STEPS    = 25
const TOL_TRUE = 0.6

function build_model(tol, rng)
    space = GridSpace(GRID; periodic = false)
    model = StandardABM(Schelling, space;
        agent_step!  = schelling_step!,
        properties   = Dict(:tolerance => tol),
        rng          = rng)                 # model owns the deterministic stream
    for n in 1:N_AGENTS
        add_agent_single!(model; group = n <= N_AGENTS ÷ 2 ? 1 : 2)
    end
    return model
end

happy_fraction(model) = count(a -> a.mood, allagents(model)) / nagents(model)

# Segregation index: mean fraction of same-group neighbors over all agents.
# Continuous in [0,1] (≈0.5 for a random configuration, →1 when segregated),
# and increases with tolerance over the useful range — unlike the happy
# fraction, which saturates at 1.0 and carries almost no information about θ.
function segregation_index(model)
    total = 0.0
    for agent in allagents(model)
        n_same = 0
        n_total = 0
        for neighbor in nearby_agents(agent, model)
            n_total += 1
            n_same += agent.group == neighbor.group
        end
        n_total > 0 && (total += n_same / n_total)
    end
    return total / nagents(model)
end

# --- ABC components ---------------------------------------------------------
simulator = AgentsSimulator(
    model_builder = (θ, rng) -> build_model(θ[1], rng),
    step!         = (model, _rng) -> step!(model, 1),   # model carries its rules + rng
    summary       = segregation_index,
    steps         = STEPS,
)

prior = Uniform(0.1, 0.95)
cfg = ABCSMCConfig(
    θ -> logpdf(prior, θ[1]),
    rng -> [rand(rng, prior)],
    identity,
    (s, s_obs) -> abs(s - s_obs),
    GaussianKernel(), [0.01;;], simulator,
)

# Synthetic "observed" segregation outcome (its own seed, independent of sampler)
obs_model = build_model(TOL_TRUE, MersenneTwister(7))
for _ in 1:STEPS
    step!(obs_model, 1)
end
y_obs = segregation_index(obs_model)
@printf("Observed segregation index after %d steps: %.3f  (true tolerance = %.2f)\n\n",
        STEPS, y_obs, TOL_TRUE)

# --- run ABC-SMC with adaptive ε and checkpointing --------------------------
N = 100
ckpt = mktempdir()
state = make_state(cfg, Vector{Float64}, N, 0.35, 42)
run_smc!(state, y_obs; T=4, schedule=QuantileSchedule(0.5), checkpoint_dir=ckpt)

@printf("Generation  ε         ESS     attempts\n")
@printf("───────────────────────────────────────\n")
for (g, path) in list_checkpoints(ckpt)
    st = load_checkpoint(path, cfg)
    @printf("    %d      %.4f   %6.1f   %d\n", g, st.ε, st.ess, st.n_attempts)
end

tol_mean = sum(state.θ[i][1] * state.w[i] for i in eachindex(state.w))
tol_var  = sum(state.w[i] * (state.θ[i][1] - tol_mean)^2 for i in eachindex(state.w))
@printf("\nPosterior tolerance: mean = %.3f, sd = %.3f   (true = %.2f)\n",
        tol_mean, sqrt(tol_var), TOL_TRUE)
@printf("Final ε = %.4f, ESS = %.1f, resamplings = %d\n", state.ε, state.ess, state.n_resamples)

@assert abs(tol_mean - TOL_TRUE) < 0.2 "posterior mean too far from truth: $tol_mean"
println("\n✓ Agents.jl-backend example completed successfully")