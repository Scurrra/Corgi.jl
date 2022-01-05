"""
    cwr(n::Int, k::Int)

Combinations with repetitions
"""
function cwr(n, k)
    combs = Vector[]
    buf = fill(1, k)

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