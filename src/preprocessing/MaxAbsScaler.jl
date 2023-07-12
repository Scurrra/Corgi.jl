"""
    MaxAbsScaler{FTYPE} <: AbstractScaler

Scaler that transforms `features` data by scaling them to fit range [-1, 1]
"""
struct MaxAbsScaler{FTYPE<:Union{Colon,Vector{<:Union{Int,Symbol,String}}}} <: AbstractScaler
    max::Matrix{Float64}
    features::FTYPE

    function MaxAbsScaler(data::AbstractMatrix{<:Real}; features=:)
        max = maximum(Float64 ∘ abs, data; dims=1)
        new{typeof(features)}(max, features)
    end
end

"""
    MaxAbsScaler(data::T; features=:)
    
Construct scaler that scales `features` from `data` to [-1, 1].
"""
MaxAbsScaler(data::AbstractDataFrame; features=:) = MaxAbsScaler(data[!, features] |> Matrix; features=features)

"""
   	transform!(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale features of `data` to [-1, 1] inplace.
"""
transform!(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = @.(data[:, scaler.features] = data[:, scaler.features] / scaler.max)

"""
   	transform(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale features of `data` to [-1, 1].
"""
transform(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation inplace.
"""
inverse_transform!(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = @.(data[:, scaler.features] = data[:, scaler.features] * scaler.max)

"""
   	inverse_transform(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::MaxAbsScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = inverse_transform!(scaler, copy(data))