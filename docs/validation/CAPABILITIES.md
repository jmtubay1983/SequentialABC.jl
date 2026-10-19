# Production capability audit — SequentialABC.jl

Model-agnostic audit of the production engine against the capabilities a stochastic
external simulator needs in order to be calibrated by this package. This table is the
Step 1 deliverable of the validation-harness task: it is written **before** any gate so
that the gates are a check on this record, not a substitute for it.

Status values:

- **present** — a production symbol implements the capability and is exercised by a gate.
- **partial** — some production support exists, but it does not cover the full requirement.
- **missing** — no production symbol implements the capability (grep count 0 in `src/`).

Gates that probe each row live in `test/gates/` and are **not** part of `Pkg.test()`;
see `test/gates/run_gates.jl`.

## Audit commands

```bash
ls src
sed -n '1,240p' src/SequentialABC.jl
ls test
grep -RIn -e EvaluationContext -e EvaluationResult -e replicate_seeds -e proposal_id \
          -e evaluation_id -e provenance -e make_particle_seed -e checkpoint src
```

Symbol counts in `src/` (`grep -RIc --include='*.jl'`):

| symbol | hits | status |
|---|---:|---|
| `make_particle_seed` | 15 | present |
| `save_checkpoint` | 9 | present |
| `load_latest_state` | 8 | present |
| `load_checkpoint` | 4 | present |
| `importance_weight` | 5 | present |
| `run_smc` | 26 | present |
| `run_inference` | 9 | present |
| `ContinuousSimulator` | 9 | present |
| `AgentsSimulator` | 9 | present |
| `EvaluationContext` | 0 | missing |
| `EvaluationResult` | 0 | missing |
| `replicate_seeds` | 0 | missing |
| `proposal_id` | 0 | missing |
| `evaluation_id` | 0 | missing |
| `provenance` | 0 | missing |
| `run_smc_from_cfg` | 0 | missing |
| `cache_evaluation` / `get_cached_evaluation` | 0 | missing |
| `max_simulations` / `sim_budget` | 0 | missing |
| `ancestry` / `ancestor_id` | 0 | missing |

## Capability table

| # | Capability | Status | Evidence (file:line) |
|---:|---|---|---|
| 1 | Canonical simulator callback | **present** | `src/simulator.jl:1` (`abstract type ABCSimulator`), `src/simulator.jl:13` (`AgentsSimulator`), `src/simulator.jl:27` (`ContinuousSimulator`), `src/simulator.jl:33` and `:41` (`run_sim_once` dispatch) |
| 2 | Seed derivation | **present** | `src/rng.jl:13` (`make_particle_seed(gen, i, attempt, m, base)`), SplitMix64-style mixing of the coordinate tuple with `SEED_SCHEMA_VERSION` — deterministic, no global RNG, cross-version stable |
| 3 | Per-replicate seeds | **partial** | The *derivation* exists (`src/rng.jl:26`–`:30`, role indices at `src/smc.jl:35`/`:116`/`:120`/`:158`) but there is no `EvaluationContext` exposing `replicate_seeds` to a callback (rows above are 0 hits). A downstream adapter cannot read replicate seeds out of a production context object today. |
| 4 | Checkpoint write / read / restart | **present** | `src/checkpoint.jl:22` (`save_checkpoint`), `src/checkpoint.jl:69` (`load_checkpoint`), `src/checkpoint.jl:88` (`load_latest_state`), `src/checkpoint.jl:51` (`list_checkpoints`) |
| 5 | Importance weights | **present** | `src/weights.jl:17` (`importance_weight`, log-space), `src/weights.jl:37` (`ess`), `src/weights.jl:56` (`systematic_resample`) |
| 6 | Proposal / evaluation IDs | **missing** | grep `proposal_id`, `evaluation_id` = 0 hits; `ABCSMCState` fields at `src/state.jl:19`–`:31` are `cfg, θ, w, summaries, dists, ε, gen, ess, base_seed, n_attempts, n_resamples` only |
| 7 | Ancestry | **missing** | grep `ancestry`, `ancestor_id` = 0 hits; resampling in `src/smc.jl:158` draws ancestors but only the resulting `θ` vector is retained (`src/state.jl:19`–`:31`) |
| 8 | Mid-generation checkpointing | **missing** | grep `cache_evaluation`, `get_cached_evaluation` = 0 hits; checkpoints are **generation-complete only** (`src/checkpoint.jl:22` `save_checkpoint` is called after generation 1 and after every completed generation at `src/smc.jl:205`/`:211`). There is no mid-generation (pre-accept / partial-batch) checkpoint that would let a long run resume from within a generation. Standard ABC-SMC redraw-until-accepted does not re-simulate retained particles, so the missing piece is mid-run resume granularity, not redundant simulation. |
| 9 | Cache of retained particles | **missing** | grep `cache_evaluation`, `get_cached_evaluation` = 0 hits; `run_generation!` at `src/smc.jl:95` re-evaluates every particle on every generation, with no branch that reuses a retained evaluation |
| 10 | Provenance | **missing** | grep `provenance` = 0 hits in `src/` |
| 11 | Threaded execution | **partial** | `grep -RIn 'Threads\|@spawn\|@threads' src` → no matches; no internal parallelism, but every RNG is locally constructed (`src/rng.jl:30`) so the engine is deterministic under an externally spawned `Threads.@spawn` task — verified end-to-end by gate 3 |
| 12 | Simulator-failure propagation | **present** | `grep -RIn 'try\|catch' src/smc.jl src/rng.jl src/checkpoint.jl src/weights.jl src/state.jl src/simulator.jl src/config.jl` → no matches, so a simulator exception escapes `run_sim_once` (`src/simulator.jl:33`/`:41`) uncaught through `run_generation!` (`src/smc.jl:95`); verified end-to-end by gate 3 |
| 13 | Simulation budget cap | **partial** | per-particle attempt cap only: `max_attempts::Int=10_000` at `src/smc.jl:6`, `:21`, `:74`, `:95`, `:186`, `:198`, `:263`, raising at `src/smc.jl:56`/`:144`; grep `max_simulations`, `sim_budget` = 0 hits — there is no global cap across generations |
| 14 | Version-stable seed derivation | **present** | `src/rng.jl:13` uses SplitMix64 finalizer mixing with `SEED_SCHEMA_VERSION = 0x0000000000000001` (`src/rng.jl:4`). Checkpoints store `seed_schema_version` (`src/checkpoint.jl:36`) and `load_checkpoint` rejects mismatched versions (`src/checkpoint.jl:69-82`) |

## Consequences for the gates

- Rows 6, 7, 8, 9, 10 and 13 cannot be exercised against production code. The gates
  record `CAPABILITY_NOT_IMPLEMENTED: <name>` for each instead of asserting a
  test-local stand-in, so the report cannot claim coverage that the engine does not
  provide. Threaded execution (row 11) is exercised empirically by gate 3 even though no
  internal parallelism exists.
- Row 3 (per-replicate seeds) is present as a *seed-derivation mechanism* but there is
  no `EvaluationContext` exposing `replicate_seeds` to a callback (rows above are 0 hits).
  A downstream adapter cannot read replicate seeds out of a production context object
  today.
- Gate 4 reports `CAPABILITY_GAPS(...)` when production features (EvaluationContext /
  EvaluationResult / replicate_seeds / provenance) are missing, and reports
  `ABM_INTEGRATION_NOT_RUN` only when no downstream adapter is supplied at all. The
  two are not interchangeable: a gap is a missing production capability; NOT_RUN is an
  undeployed downstream adapter.
- Row 13 being *partial* means a run is bounded per particle, not per simulation budget;
  a long run over many generations is not cost-capped.
