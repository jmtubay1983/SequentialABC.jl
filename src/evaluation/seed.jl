"""
    EVALUATION_SEED_SCHEMA_VERSION::UInt64

Version of the seed derivation encoding used by [`evaluation_seed`](@ref).

Schema 1 mixes the complete evaluation identity
`(method, generation, particle, proposal, attempt, replicate, base_seed)` into
a `UInt64` using an explicit, stable byte encoding. Changing any identity field
changes the derived seed within the tested domain. This is deliberately *not*
Julia's `hash()` and is not intended as a cryptographic or content-addressable
identifier; it is a persistent, deterministic, per-evaluation seed.

The legacy [`make_particle_seed`](@ref) stream is a separate, explicitly named
legacy schema. It is preserved unchanged and remains the seed source for the
existing SMC/PMC/MCMC engines.
"""
const EVALUATION_SEED_SCHEMA_VERSION::UInt64 = UInt64(1)

"""
    EVALUATION_SEED_SCHEMA_NAME::String

Human-readable name of the public evaluation seed schema recorded by
[`context_primitive`](@ref). The legacy engine stream is named separately by
`LEGACY_PARTICLE_SEED_SCHEMA_NAME`.
"""
const EVALUATION_SEED_SCHEMA_NAME = "evaluation_seed_v1"

"""
    LEGACY_PARTICLE_SEED_SCHEMA_NAME::String

Name of the pre-existing particle seed stream produced by
[`make_particle_seed`](@ref). This schema is frozen: existing SMC/PMC/MCMC
runs must continue to reproduce their historical seed stream.
"""
const LEGACY_PARTICLE_SEED_SCHEMA_NAME = "make_particle_seed_legacy_hash"

# SplitMix64 multipliers used by the documented final avalanche below.
const _SPLITMIX_MULTIPLIER_1::UInt64 = UInt64(0xBF58476D1CE4E5B9)
const _SPLITMIX_MULTIPLIER_2::UInt64 = UInt64(0x94D049BB133111EB)

# Explicit FNV-1a 64-bit encoding. Every field is encoded as a fixed-width
# little-endian byte sequence; `method` is encoded as its UTF-8 bytes preceded
# by an 8-byte length. No Julia `hash()` is used anywhere in this derivation.
const _FNV1A_OFFSET_BASIS::UInt64 = UInt64(0xCBF29CE484222325)
const _FNV1A_PRIME::UInt64 = UInt64(0x00000100000001B3)

@inline function _seed_mix_byte(h::UInt64, b::UInt8)::UInt64
    return xor(h, UInt64(b)) * _FNV1A_PRIME
end

@inline function _seed_mix_u64(h::UInt64, x::UInt64)::UInt64
    # Explicit little-endian byte order, independent of host endianness.
    for shift in 0:8:56
        h = _seed_mix_byte(h, UInt8((x >> shift) & 0xFF))
    end
    return h
end

@inline function _seed_finalize(h::UInt64)::UInt64
    # SplitMix64 finalizer: documented integer avalanche, no `hash()`.
    h = xor(h, h >> 30)
    h *= _SPLITMIX_MULTIPLIER_1
    h = xor(h, h >> 27)
    h *= _SPLITMIX_MULTIPLIER_2
    h = xor(h, h >> 31)
    return h
end

# Validate and narrow a public identity index to the fixed-width storage type.
@inline function _evaluation_index(value::Integer, name::AbstractString)::Int64
    value >= 1 || throw(ArgumentError("$name must be ≥ 1, got $(value)"))
    value <= typemax(Int64) || throw(ArgumentError(
        "$name must fit in Int64, got $(value)"))
    return Int64(value)
end

"""
    evaluation_seed(
        method::Symbol,
        generation::Integer,
        particle::Integer,
        proposal::Integer,
        attempt::Integer,
        replicate::Integer,
        base_seed::UInt64,
    ) -> UInt64

Derive a per-evaluation seed from the complete evaluation identity using the
versioned [`EVALUATION_SEED_SCHEMA_VERSION`](@ref) encoding.

Canonical byte stream (schema 1), in order:

1. `EVALUATION_SEED_SCHEMA_VERSION` as 8 little-endian bytes;
2. UTF-8 byte length of `String(method)` as 8 little-endian bytes;
3. UTF-8 bytes of `String(method)`;
4. `generation`, `particle`, `proposal`, `attempt`, `replicate`, and
   `base_seed`, each as 8 little-endian bytes.

The bytes are folded with FNV-1a 64-bit and finalized with the SplitMix64
avalanche. The derivation is deterministic, portable, and independent of Julia
`hash()`.

Inputs are validated: `method` must be a nonempty `Symbol`; the five indices
must be ≥ 1 and representable as `Int64`; and `base_seed` is a `UInt64`.
`UInt64(0)` is a valid seed value; there is no reserved zero sentinel.

This is *not* a replacement for [`make_particle_seed`](@ref). The legacy
[`make_particle_seed`](@ref) stream is preserved bit-for-bit for existing
inference runs; `evaluation_seed` is the seed source for the new evaluation
contract only.
"""
function evaluation_seed(
    method::Symbol,
    generation::Integer,
    particle::Integer,
    proposal::Integer,
    attempt::Integer,
    replicate::Integer,
    base_seed::UInt64,
)::UInt64
    isempty(string(method)) && throw(ArgumentError(
        "evaluation_seed method must be a nonempty Symbol, got $(repr(method))"))
    g = _evaluation_index(generation, "evaluation_seed generation")
    p = _evaluation_index(particle, "evaluation_seed particle")
    q = _evaluation_index(proposal, "evaluation_seed proposal")
    a = _evaluation_index(attempt, "evaluation_seed attempt")
    r = _evaluation_index(replicate, "evaluation_seed replicate")

    method_bytes = codeunits(String(method))
    h = _FNV1A_OFFSET_BASIS
    h = _seed_mix_u64(h, EVALUATION_SEED_SCHEMA_VERSION)
    h = _seed_mix_u64(h, UInt64(length(method_bytes)))
    for b in method_bytes
        h = _seed_mix_byte(h, b)
    end
    h = _seed_mix_u64(h, UInt64(g))
    h = _seed_mix_u64(h, UInt64(p))
    h = _seed_mix_u64(h, UInt64(q))
    h = _seed_mix_u64(h, UInt64(a))
    h = _seed_mix_u64(h, UInt64(r))
    h = _seed_mix_u64(h, base_seed)
    return _seed_finalize(h)
end
