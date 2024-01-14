module Utils
   
using Random

"""
    cwr(n::Int, k::Int)

Combinations with repetitions
"""
function cwr(n::Int, k::Int)
    combs = Vector{Int}[]
    buf = ones(Int, k)

    while true
        for i = k:-1:1
            if buf[i] > n
                buf[i-1] += 1
                buf[i:k] .= buf[i-1]
            end
        end
        push!(combs, copy(buf))
        all(buf .>= n) && break
        buf[k] += 1
    end

    return combs
end
export cwr

"""
    Shuffler

Abstraction over Random.shuffle, because it needs Random.AbstractRNG for evaluating. 
Here used MersenneTwister, stored in struct, `seed` is optional, default=42.
"""
struct Shuffler
    data::AbstractArray
    twister::AbstractRNG

    function Shuffler(data::AbstractArray; seed::Int=42)
        if seed <= 0
            throw("Seed must be greater tan 0")
        end

        new(data, MersenneTwister(seed))
    end
end
(shuffler::Shuffler)() = shuffle(shuffler.twister, shuffler.data)
export Shuffler

import Base: split
"""
    split(data::AbstractVector, slen::Int)

Splt data into batches.

!Warning: no checks.
"""
function split(data::AbstractVector, slen::Int)
    return [
        data[(s*slen+1):((s+1)*slen)]
        for s in 0:(div(length(data),slen)-1)
    ]
end
export split


end