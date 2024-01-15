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

### PLACEHOLDER FOR LINEAR MODELS
abstract type AbstractLinearModel end
export AbstractLinearModel
###

include("optimization/Optimization.jl")
using .Optimization

end # module
