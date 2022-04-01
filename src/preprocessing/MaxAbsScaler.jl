"""
    MaxAbsScaler{T, OUTRANGE} <: AbstractTransformer{T, OUTRANGE}

Scaler that transforms `features` data of type `T` by scaling them to fit range `OUTRANGE`
"""
struct MaxAbsScaler{T, OUTRANGE} <: AbstractTransformer{T, OUTRANGE}
    max::Matrix{Float64}
    features::Vector{Union{Int, Symbol, String}}
    
    function MaxAbsScaler{T, OUTRANGE}(data::AbstractMatrix{<:Real}; features=:) where {T, OUTRANGE}
        max = maximum(abs, data, dims=1) .|> Float64
        new{T, Float64.(OUTRANGE)}(max, features == (:) ? collect(1:size(data, 2)) : collect(features))
    end
end
MaxAbsScaler{T, OUTRANGE}(data::T; features=:) where {T<:AbstractDataFrame, OUTRANGE} = MaxAbsScaler{T, OUTRANGE}(data[!, features] |> Matrix; features=features)

"""
    MaxAbsScaler{T}(data::T; features=:, outrange::NTuple{2, <:Real}=(-1., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
MaxAbsScaler{T}(data::T; features=:, outrange::NTuple{2, <:Real}=(-1., 1.)) where {T<:AbstractDataFrame} = MaxAbsScaler{T, outrange}(data[!, features] |> Matrix; features=features)
MaxAbsScaler{T}(data::T; features=:, outrange::NTuple{2, <:Real}=(-1., 1.)) where {T<:AbstractMatrix{<:Real}} = MaxAbsScaler{T, outrange}(data[:, features]; features=features)

"""
    MaxAbsScaler(data::T; features=:, outrange::NTuple{2, <:Real}=(-1., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
MaxAbsScaler(data::T; features=:, outrange::NTuple{2, <:Real}=(-1., 1.)) where {T} = MaxAbsScaler{T}(data; features=features, outrange=outrange)


"""
   	transform!(scaler::MaxAbsScaler{T, OUTRANGE}, data::T)

Scale features of `data` of type `T` according to `OUTRANGE`.
"""
transform!(scaler::MaxAbsScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = @.(data[:, scaler.features] = data[:, scaler.features] / scaler.max * (OUTRANGE[2] - OUTRANGE[1]) + OUTRANGE[1])

"""
   	transform(scaler::MaxAbsScaler{T, OUTRANGE}, data::T)

Scale features of `data` of type `T` according to `OUTRANGE`.
"""
transform(scaler::MaxAbsScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::MaxAbsScaler{T, OUTRANGE}, data::T)

Scale back the `data` of type `T` to the original representation.
"""
inverse_transform!(scaler::MaxAbsScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = @.(data[:, scaler.features] = (data[:, scaler.features] - OUTRANGE[1]) / (OUTRANGE[2] - OUTRANGE[1]) * scaler.max)

"""
   	inverse_transform(scaler::MaxAbsScaler{T, OUTRANGE}, data::T)

Scale back the `data` of type `T` to the original representation.
"""
inverse_transform(scaler::MaxAbsScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = inverse_transform!(scaler, copy(data))