"""
    MaxAbsScaler{T, OUTRANGE} <: AbstractTransformer{T, OUTRANGE}

Scaler that transforms `features` data of type `T` by scaling them to fit range `OUTRANGE`
"""
struct MinMaxScaler{T, OUTRANGE} <: AbstractTransformer{T, OUTRANGE}
    min
    max
    features
    
    function MinMaxScaler{T, OUTRANGE}(data::AbstractMatrix{<:Real}; features=:) where {T, OUTRANGE}
        min = minimum(data .|> Float64, dims=1)
        max = maximum(data .|> Float64, dims=1)
        new{T, Float64.(OUTRANGE)}(min, max, features)
    end
end
MinMaxScaler{T, OUTRANGE}(data::T; features=:) where {T<:AbstractDataFrame, OUTRANGE} = MinMaxScaler{T, OUTRANGE}(data[!, features] |> Matrix; features=features)

"""
    MaxAbsScaler{T}(data::T; features=:, outrange::NTuple{2, <:Real}=(0., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
MinMaxScaler{T}(data::T; features=:, outrange::NTuple{2, <:Real}=(0., 1.)) where {T<:AbstractDataFrame} = MinMaxScaler{T, outrange}(data[!, features] |> Matrix; features=features)
MinMaxScaler{T}(data::T; features=:, outrange::NTuple{2, <:Real}=(0., 1.)) where {T<:AbstractMatrix{<:Real}} = MinMaxScaler{T, outrange}(data[:, features]; features=features)

"""
    MaxAbsScaler(data::T; features=:, outrange::NTuple{2, <:Real}=(0., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
MinMaxScaler(data::T; features=:, outrange::NTuple{2, <:Real}=(0., 1.)) where {T} = MinMaxScaler{T}(data; features=features, outrange=outrange)


"""
   	transform!(scaler::MinMaxScaler{T, OUTRANGE}, data::T)

Scale features of `data` of type `T` according to `OUTRANGE`.
"""
#transform!(scaler::MinMaxScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = (data[:, scaler.features] .= (data[:, scaler.features] .- scaler.min) ./ (scaler.max .- scaler.min) .* (OUTRANGE[2] - OUTRANGE[1]) .+ OUTRANGE[1])
transform!(scaler::MinMaxScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = @.(data[:, scaler.features] = (data[:, scaler.features] - scaler.min) / (scaler.max - scaler.min) * (OUTRANGE[2] - OUTRANGE[1]) + OUTRANGE[1])

"""
   	transform(scaler::MinMaxScaler{T, OUTRANGE}, data::T)

Scale features of `data` of type `T` according to `OUTRANGE`.
"""
transform(scaler::MinMaxScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::MinMaxScaler{T, OUTRANGE}, data::T)

Scale back the `data` of type `T` to the original representation.
"""
inverse_transform!(scaler::MinMaxScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = @.(data[:, scaler.features] = (data[:, scaler.features] - OUTRANGE[1]) / (OUTRANGE[2] - OUTRANGE[1]) * (scaler.max - scaler.min) + scaler.min)

"""
   	inverse_transform(scaler::MinMaxScaler{T, OUTRANGE}, data::T)

Scale back the `data` of type `T` to the original representation.
"""
inverse_transform(scaler::MinMaxScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = inverse_transform!(scaler, copy(data))


"""
   	fit_transform(data::T; features, outrange::NTuple{2,Float64}=(0.0, 1.0))

Fit and scale `features` of `data` of type `T` according to `outrange`.
"""
fit_transform(data::T; features=:, outrange::NTuple{2,Float64}=(0.0, 1.0)) where {T, OUTRANGE} = transform!(MinMaxScaler{T, outrange}(data; features=features), data)