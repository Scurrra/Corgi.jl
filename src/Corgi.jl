module Corgi

using DataFrames
using Distributions
using LinearAlgebra
using Optim: optimize
using Random
using Statistics

include("Utils/Utils.jl")

include("Logging.jl")

include("preprocessing/Preprocessing.jl")
using .Preprocessing

abstract type AbstractLinearModel end
export AbstractLinearModel

include("optimization/Optimization.jl")
using .Optimization

include("linear_models/LinearModels.jl")
using .LinearModels

end # module
