module Preprocessing

using DataFrames
using Distributions
using LinearAlgebra
using Optim: optimize
using Statistics
using Zygote: gradient # to be removed

abstract type AbstractTransformer{T<:Union{AbstractMatrix{<:Real}, AbstractDataFrame}, OUTRANGE} end

export StandardScaler, MinMaxScaler, PowerTransformer, OneHotEncoder, PolynomialFeatures
export fit!, transform!, transform, inverse_transform!, inverse_transform

include("MinMaxScaler.jl")
include("MaxAbsScaler.jl")
# include("PowerTransformer.jl")
# include("StandardScaler.jl")
# include("OneHotEncoder.jl")
# include("PolynomialFeatures.jl")

end