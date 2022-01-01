module Preprocessing

using LinearAlgebra
using Statistics

export StandardScaler, MinMaxScaler, fit!, transform!, transform, inverse_transform!, inverse_transform, fit_transform

include("StandardScaler.jl")
include("MinMaxScaler.jl")

end