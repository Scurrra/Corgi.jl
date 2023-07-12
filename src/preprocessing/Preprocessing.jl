module Preprocessing

using DataFrames
using Distributions
using LinearAlgebra
using Optim
using Statistics: mean, std

abstract type AbstractTransformer end
abstract type AbstractScaler <: AbstractTransformer end

export MaxAbsScaler, StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
export fit!, transform!, transform, inverse_transform!, inverse_transform

include("MinMaxScaler.jl")
include("MaxAbsScaler.jl")
include("StandardScaler.jl")
include("OneHotEncoder.jl")
include("PolynomialFeatures.jl")

end