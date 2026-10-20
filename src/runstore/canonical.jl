# ============================================================================
# Canonical file encodings for the native store
# ============================================================================
#
# The native store uses two encodings:
#
#   * a canonical `key=value` text encoding for the schema file, the
#     manifests, the COMPLETE markers, and the writer lease;
#   * Julia `Serialization` for the record partitions.
#
# The text encoding is deterministic: fixed key order, one `key=value` per
# line, LF line endings, values that are either an integer, a lowercase hex
# digest, an ISO-8601 UTC timestamp, or a percent-encoded string. The same
# bytes are therefore produced on every platform, which is what makes the
# SHA-256 digests of the metadata and of the manifests reproducible.

const _CANONICAL_UNRESERVED = Set{UInt8}(collect(codeunits(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")))

"""
    _canonical_encode(value::AbstractString) -> String

Percent-encode a text value for the canonical `key=value` encoding.

Every byte outside the unreserved set of RFC 3986 is written as `%XX` with
uppercase hexadecimal digits, so a value can never contain a newline, a
space, or a `=` that would make the encoding ambiguous.
"""
function _canonical_encode(value::AbstractString)
    io = IOBuffer()
    for byte in codeunits(value)
        if byte in _CANONICAL_UNRESERVED
            write(io, byte)
        else
            print(io, '%', uppercase(string(byte, base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

"""
    _canonical_decode(value::AbstractString) -> String

Decode a percent-encoded value of the canonical encoding.
"""
function _canonical_decode(value::AbstractString)
    bytes = UInt8[]
    index = firstindex(value)
    last = lastindex(value)
    while index <= last
        char = value[index]
        if char == '%'
            index + 2 <= last || throw(ArgumentError(
                "truncated percent escape in $(repr(value))"))
            digits = SubString(value, index + 1, index + 2)
            byte = tryparse(UInt8, digits; base = 16)
            byte === nothing && throw(ArgumentError(
                "invalid percent escape $(repr(digits)) in $(repr(value))"))
            push!(bytes, byte)
            index += 3
        else
            isascii(char) || throw(ArgumentError(
                "non-ASCII byte in canonical value $(repr(value))"))
            push!(bytes, UInt8(char))
            index = nextind(value, index)
        end
    end
    return String(bytes)
end

"""
    _write_canonical_kv(path::AbstractString, entries::Vector{Pair{String,String}})

Write a canonical `key=value` file.

The caller decides the key order, which makes the file byte-deterministic.
"""
function _write_canonical_kv(path::AbstractString, entries::Vector{Pair{String,String}})
    open(path, "w") do io
        for (key, value) in entries
            print(io, key, "=", value, "\n")
        end
    end
    return path
end

"""
    _read_canonical_kv(path::AbstractString) -> Dict{String,String}

Read a canonical `key=value` file.

The parser is strict: every line must be `key=value` with a lowercase key,
a nonempty value, and no whitespace in the value; duplicate keys are an
error. A malformed file is a corruption, not a file to be repaired.
"""
function _read_canonical_kv(path::AbstractString)
    isfile(path) || return nothing
    entries = Dict{String,String}()
    for raw in readlines(path)
        line = endswith(raw, "\r") ? chop(raw) : raw
        isempty(line) && throw(ArgumentError("empty line"))
        separator = findfirst('=', line)
        separator === nothing && throw(ArgumentError(
            "line without a key/value separator"))
        key = SubString(line, firstindex(line), separator - 1)
        value = SubString(line, separator + 1, lastindex(line))
        isempty(key) && throw(ArgumentError("empty key"))
        occursin(r"^[a-z][a-z0-9_]*$", key) || throw(ArgumentError(
            "invalid key $(repr(String(key)))"))
        isempty(value) && throw(ArgumentError("empty value for key $(key)"))
        any(isspace, value) && throw(ArgumentError(
            "whitespace in the value of key $(key)"))
        haskey(entries, key) && throw(ArgumentError("duplicate key $(key)"))
        entries[String(key)] = String(value)
    end
    return entries
end

"""
    _file_digest(path::AbstractString) -> String

The lowercase hex SHA-256 digest of the bytes of a file.
"""
function _file_digest(path::AbstractString)
    open(path, "r") do io
        return SHA.bytes2hex(SHA.sha256(io))
    end
end

"""
    _evaluation_key_set_hash(keys) -> String

A deterministic digest of a set of evaluation identities.

The keys are sorted and written in one canonical line each, so the digest
does not depend on the order in which the records were produced. It is used
to detect missing or extra evaluations of a generation, both at publication
and at reopen.
"""
function _evaluation_key_set_hash(keys)
    io = IOBuffer()
    for key in sort!(collect(keys))
        print(io, _canonical_encode(string(key.method)), "|", key.generation, "|",
              key.particle, "|", key.proposal, "|", key.attempt, "|",
              key.replicate, "\n")
    end
    return SHA.bytes2hex(SHA.sha256(take!(io)))
end

"""
    _native_timestamp(dt::DateTime) -> String

Format a timestamp as an ISO-8601 UTC string with millisecond precision.
"""
function _native_timestamp(dt::DateTime)
    return Dates.format(dt, "yyyy-mm-ddTHH:MM:SS.sss")
end
