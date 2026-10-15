abstract type ABCSimulator end

"""
    AgentsSimulator(model_builder, step!, summary, steps)

Agents.jl backend. `model_builder(θ)` builds the model, `step!(model, rng)`
steps it (RNG threaded explicitly — never the global RNG), `summary(model)`
extracts the summary statistic after `steps` steps.
"""
struct AgentsSimulator{MB,St,Su} <: ABCSimulator
    model_builder::MB     # θ → Agents.jl model
    step!::St             # (model, rng) → nothing
    summary::Su           # model → summary
    steps::Int
end

"""
    ContinuousSimulator(model_builder, evolve, summary)

Continuous (ODE/PDE/SSA) backend. `model_builder(θ)` builds the problem,
`evolve(problem, rng)` runs it to completion, `summary(final_state)`
extracts the summary statistic.
"""
struct ContinuousSimulator{MB,Ev,Su} <: ABCSimulator
    model_builder::MB     # θ → initial state / problem
    evolve::Ev            # (state_or_problem, rng) → final_state
    summary::Su           # final_state → summary
end

function run_sim_once(sim::AgentsSimulator, θ, rng)
    model = sim.model_builder(θ)
    for _ in 1:sim.steps
        sim.step!(model, rng)
    end
    return sim.summary(model)
end

function run_sim_once(sim::ContinuousSimulator, θ, rng)
    problem = sim.model_builder(θ)
    final_state = sim.evolve(problem, rng)
    return sim.summary(final_state)
end

# Keyword constructors (the planned public API)
AgentsSimulator(; model_builder, step!, summary, steps) =
    AgentsSimulator(model_builder, step!, summary, steps)

ContinuousSimulator(; model_builder, evolve, summary) =
    ContinuousSimulator(model_builder, evolve, summary)