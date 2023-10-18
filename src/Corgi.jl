module Corgi

using DataFrames
using Distributions
using LinearAlgebra
using Optim: optimize
using Statistics

include("preprocessing/Preprocessing.jl")
using .Preprocessing

include("optimization/Optimization.jl")
using .Optimization

end # module
