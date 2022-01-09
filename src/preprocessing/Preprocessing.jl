module Preprocessing

using DataFrames
using Distributions
using LinearAlgebra
using Optim: optimize
using Statistics

export StandardScaler, MinMaxScaler, PowerTransformer, OneHotEncoder, PolynomialFeatures
export fit!, transform!, transform, inverse_transform!, inverse_transform, fit_transform

include("MinMaxScaler.jl")
include("PowerTransformer.jl")
include("StandardScaler.jl")
include("OneHotEncoder.jl")
include("PolynomialFeatures.jl")

end