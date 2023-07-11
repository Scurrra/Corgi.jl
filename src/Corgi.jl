module Corgi

using DataFrames
using Distributions
using LinearAlgebra
using Optim: optimize
using Statistics

include("preprocessing/Preprocessing.jl")
using .Preprocessing

end # module
