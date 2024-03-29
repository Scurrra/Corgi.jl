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