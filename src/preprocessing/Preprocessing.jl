module Preprocessing

using LinearAlgebra
using Statistics

export MinMaxScaler, PowerTransformer, StandardScaler
export fit!, transform!, transform, inverse_transform!, inverse_transform, fit_transform

include("StandardScaler.jl")
include("MinMaxScaler.jl")
include("PowerTransformer.jl")

end