# ============================================================================
# Run metadata
# ============================================================================
#
# `RunMetadata` is the identity of one run-history store. It is created once,
# when the run is initialized, and never changes afterwards.
#
# Schema independence: the storage schema, the seed schema, the checkpoint
# schema, and (in a later PR) the portable-export schema are independent
# concepts. `RunMetadata` records the storage schema and the *declared* seed
# schema; it never interprets or re-derives a seed.

"""
    RUNSTORE_SCHEMA_NAME

Name of the run-history storage schema.

The storage schema is independent of the seed schema, the checkpoint schema,
and any portable-export schema.
"""
const RUNSTORE_SCHEMA_NAME = "sequentialabc.runstore"

"""
    RUNSTORE_SCHEMA_VERSION

Version of the run-history storage schema understood and written by this
build.

Storage schema versioning is independent of the seed schema
(`EVALUATION_SEED_SCHEMA_VERSION`) and of the checkpoint schema versions.
"""
const RUNSTORE_SCHEMA_VERSION::UInt64 = UInt64(1)

"""
    RunMetadata

Immutable identity of one run-history store.

Fields:
  - `run_id::String`              — nonempty run identifier
  - `method::Symbol`              — nonempty method symbol (e.g. `:smc`)
  - `package_version::String`     — version of the writing package
  - `julia_version::String`       — version of the writing Julia
  - `source_revision::Union{Nothing,String}` — optional source revision
  - `seed_schema_name::String`    — declared name of the seed schema in use
  - `seed_schema_version::UInt64` — declared version of the seed schema
  - `base_seed::Int64`            — checked run-level base seed
  - `created_at::DateTime`        — creation timestamp (UTC semantics)
  - `user_provenance::RecordedProvenance` — canonical user provenance
  - `storage_schema_name::String`     — storage schema name
  - `storage_schema_version::UInt64`  — storage schema version

The base seed is checked against the `Int64` range, so no silent truncation is
possible. The source revision is supplied by the caller; the store never
shells out to Git. All timestamps in the run-history contract are UTC
semantic: they are stored and interpreted as UTC, and no offset is carried
alongside them.

The seed schema is recorded, not interpreted: a store that holds evaluations
derived under a legacy seed stream says so explicitly instead of pretending
the public evaluation seed schema was used.
"""
struct RunMetadata
    run_id::String
    method::Symbol
    package_version::String
    julia_version::String
    source_revision::Union{Nothing,String}
    seed_schema_name::String
    seed_schema_version::UInt64
    base_seed::Int64
    created_at::DateTime
    user_provenance::RecordedProvenance
    storage_schema_name::String
    storage_schema_version::UInt64

    function RunMetadata(
        run_id::AbstractString,
        method::Symbol,
        package_version::AbstractString,
        julia_version::AbstractString,
        source_revision::Union{Nothing,AbstractString},
        seed_schema_name::AbstractString,
        seed_schema_version::Integer,
        base_seed::Integer,
        created_at::DateTime,
        user_provenance::RecordedProvenance,
        storage_schema_name::AbstractString,
        storage_schema_version::Integer,
    )
        rid = String(run_id)
        isempty(rid) && throw(ArgumentError("RunMetadata run_id must be nonempty"))
        isempty(string(method)) && throw(ArgumentError(
            "RunMetadata method must be a nonempty Symbol, got $(repr(method))"))
        pv = String(package_version)
        isempty(pv) && throw(ArgumentError(
            "RunMetadata package_version must be nonempty"))
        jv = String(julia_version)
        isempty(jv) && throw(ArgumentError(
            "RunMetadata julia_version must be nonempty"))
        rev = source_revision === nothing ? nothing : String(source_revision)
        rev !== nothing && isempty(rev) && throw(ArgumentError(
            "RunMetadata source_revision must be nonempty when supplied"))
        ssn = String(seed_schema_name)
        isempty(ssn) && throw(ArgumentError(
            "RunMetadata seed_schema_name must be nonempty"))
        ssv = _checked_uint64(seed_schema_version, "seed_schema_version")
        bs = _checked_int64(base_seed, "base_seed")
        ssn2 = String(storage_schema_name)
        isempty(ssn2) && throw(ArgumentError(
            "RunMetadata storage_schema_name must be nonempty"))
        ssv2 = _checked_uint64(storage_schema_version, "storage_schema_version")
        return new(
            rid, method, pv, jv, rev, ssn, ssv, bs, created_at,
            user_provenance, ssn2, ssv2,
        )
    end
end

function _checked_int64(x::Integer, name::AbstractString)
    if typemin(Int64) <= x <= typemax(Int64)
        return Int64(x)
    end
    throw(ArgumentError(
        "$(name) must fit in Int64; got $(x) of type $(typeof(x))"))
end

function _checked_uint64(x::Integer, name::AbstractString)
    if typemin(UInt64) <= x <= typemax(UInt64)
        return UInt64(x)
    end
    throw(ArgumentError(
        "$(name) must fit in UInt64; got $(x) of type $(typeof(x))"))
end

Base.:(==)(a::RunMetadata, b::RunMetadata) =
    a.run_id == b.run_id &&
    a.method == b.method &&
    a.package_version == b.package_version &&
    a.julia_version == b.julia_version &&
    a.source_revision == b.source_revision &&
    a.seed_schema_name == b.seed_schema_name &&
    a.seed_schema_version == b.seed_schema_version &&
    a.base_seed == b.base_seed &&
    a.created_at == b.created_at &&
    a.user_provenance == b.user_provenance &&
    a.storage_schema_name == b.storage_schema_name &&
    a.storage_schema_version == b.storage_schema_version

Base.copy(m::RunMetadata) = RunMetadata(
    m.run_id, m.method, m.package_version, m.julia_version, m.source_revision,
    m.seed_schema_name, m.seed_schema_version, m.base_seed, m.created_at,
    copy(m.user_provenance), m.storage_schema_name, m.storage_schema_version,
)

"""
    current_run_metadata(; run_id, method, base_seed,
                         source_revision=nothing,
                         user_provenance=RecordedProvenance(),
                         seed_schema_name=EVALUATION_SEED_SCHEMA_NAME,
                         seed_schema_version=EVALUATION_SEED_SCHEMA_VERSION,
                         storage_schema_name=RUNSTORE_SCHEMA_NAME,
                         storage_schema_version=RUNSTORE_SCHEMA_VERSION,
                         created_at=now(UTC), package_version=..., julia_version=...)

Build a [`RunMetadata`](@ref) with the current package version, Julia version,
and UTC creation timestamp filled in.

The package version is read from the loaded package; no external process is
started and Git is never consulted. `source_revision` is supplied by the
caller, if it is known.
"""
function current_run_metadata(; run_id::AbstractString,
                              method::Symbol,
                              base_seed::Integer,
                              source_revision::Union{Nothing,AbstractString}=nothing,
                              user_provenance::RecordedProvenance=RecordedProvenance(),
                              seed_schema_name::AbstractString=EVALUATION_SEED_SCHEMA_NAME,
                              seed_schema_version::Integer=EVALUATION_SEED_SCHEMA_VERSION,
                              storage_schema_name::AbstractString=RUNSTORE_SCHEMA_NAME,
                              storage_schema_version::Integer=RUNSTORE_SCHEMA_VERSION,
                              created_at::DateTime=now(UTC),
                              package_version::AbstractString=_package_version(),
                              julia_version::AbstractString=string(VERSION))
    return RunMetadata(
        run_id, method, package_version, julia_version, source_revision,
        seed_schema_name, seed_schema_version, base_seed, created_at,
        user_provenance, storage_schema_name, storage_schema_version,
    )
end

function _package_version()
    v = try
        Base.pkgversion(SequentialABC)
    catch
        nothing
    end
    return v === nothing ? "unknown" : string(v)
end
