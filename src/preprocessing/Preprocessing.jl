module Preprocessing

using LinearAlgebra
using Statistics

export StandardScaler, fit!, transform!, transform, inverse_transform!, inverse_transform, fit_transform

include("StandardScaler.jl")

end