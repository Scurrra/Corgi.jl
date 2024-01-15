module Logging

"""
    AbstractLogger

Logger base type.
"""
abstract type AbstractLogger end

"""
    NullLogger

Null logger, i.e. no logger placeholder.
"""
struct NullLogger <: AbstractLogger end
(nl::NullLogger)(_...) = undef

"""
    CostsLogger

Logger that logs only costs while optimization.
"""
struct CostsLogger <: AbstractLogger
    costs::Vector{Float64}

    CostsLogger() = new([])
end

"""
    call()
    
Log something into logger.
"""
function (cl::CostsLogger)(cost::Float64)
    push!(
        cl.costs,
        cost
    )
end

export AbstractLogger, CostsLogger

end