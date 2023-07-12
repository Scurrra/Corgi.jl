"""
    MinMaxScaler{OUTRANGE} <: AbstractScaler

Scaler that transforms `features` data by scaling them to fit range `OUTRANGE`
"""
struct MinMaxScaler{OUTRANGE,FTYPE<:Union{Colon,Vector{<:Union{Int,Symbol,String}}}} <: AbstractScaler
    min::Matrix{Float64}
    max::Matrix{Float64}
    features::FTYPE

    function MinMaxScaler{OUTRANGE}(data::AbstractMatrix{<:Real}; features=:) where {OUTRANGE}
        min = minimum(Float64, data; dims=1)
        max = maximum(Float64, data; dims=1)
        new{Float64.(OUTRANGE),typeof(features)}(min, max, features)
    end
end
MinMaxScaler{OUTRANGE}(data::AbstractDataFrame; features=:) where {OUTRANGE} = MinMaxScaler{OUTRANGE}(data[!, features] |> Matrix; features=features)

"""
    MinMaxScaler(data::T; features=:, outrange::NTuple{2, <:Real}=(0., 1.))
    
Construct scaler that scales `features` from `data` to fit `outrange`
"""
MinMaxScaler(data::AbstractDataFrame; features=:, outrange::NTuple{2,<:Real}=(0.0, 1.0)) = MinMaxScaler{outrange}(data[!, features] |> Matrix; features=features)
MinMaxScaler(data::AbstractMatrix{<:Real}; features=:, outrange::NTuple{2,<:Real}=(0.0, 1.0)) = MinMaxScaler{outrange}(data[:, features]; features=features)

"""
   	transform!(scaler::MinMaxScaler{OUTRANGE, FTYPE}, data::T)

Scale features of `data` according to `OUTRANGE` inplace.
"""
transform!(scaler::MinMaxScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = @.(data[:, scaler.features] = (data[:, scaler.features] - scaler.min) / (scaler.max - scaler.min) * (OUTRANGE[2] - OUTRANGE[1]) + OUTRANGE[1])

"""
   	transform(scaler::MinMaxScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale features of `data` according to `OUTRANGE`.
"""
transform(scaler::MinMaxScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::MinMaxScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation inplace.
"""
inverse_transform!(scaler::MinMaxScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = @.(data[:, scaler.features] = (data[:, scaler.features] - OUTRANGE[1]) / (OUTRANGE[2] - OUTRANGE[1]) * (scaler.max - scaler.min) + scaler.min)

"""
   	inverse_transform(scaler::MinMaxScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::MinMaxScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = inverse_transform!(scaler, copy(data))