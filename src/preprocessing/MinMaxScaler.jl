"""
	MinMaxScaler(; dims::Int=1, feature_range::Tuple=(0., 1.))
	MinMaxScaler(data::AbstractArray; dims::Int=1, feature_range::Tuple=(0., 1.))

Transform features by scaling each feature to a given range. Create a structure `MinMaxScaler` instance with specified parameters.

`dims` is the dimention of data to be scaled by. `feature_range` is the desired range of transformed data.
"""
mutable struct MinMaxScaler{T}
    dims::Int
    feature_range::Tuple

    min::Union{T,Nothing}
    max::Union{T,Nothing}

    MinMaxScaler(; dims::Int = 1, feature_range::Tuple = (0.0, 1.0)) = new{T}(dims, feature_range, nothing, nothing)

    function MinMaxScaler(data::AbstractArray; dims::Int = 1, feature_range::Tuple = (0.0, 1.0))
        min = minimum(data, dims = dims)
        max = maximum(data, dims = dims)
        new{typeof(min)}(dims, feature_range, min, max)
    end
end

"""
   	fit!(scaler::MinMaxScaler, data::AbstractArray)

Fit `data` using existing `scaler` parameters.
"""
function fit!(scaler::MinMaxScaler, data::AbstractArray)
    scaler = MinMaxScaler(data, dims = scaler.dims, feature_range = scaler.feature_range)
end

"""
   	transform!(scaler::MinMaxScaler, data::AbstractArray)

Scale features of `data` according to `feature_range`.
"""
transform!(scaler::MinMaxScaler, data::AbstractArray) = @.(data = (data - scaler.min) / (scaler.max - scaler.min) * (scaler.feature_range[2] - scaler.feature_range[1]) + scaler.feature_range[1])

"""
   	transform(scaler::MinMaxScaler, data::AbstractArray)

Scale features of `data` according to `feature_range`.
"""
transform(scaler::MinMaxScaler, data::AbstractArray) = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::MinMaxScaler, data::AbstractArray)

Scale back the `data` to the original representation.
"""
inverse_transform!(scaler::MinMaxScaler, data::AbstractArray) = @.(data = (data - scaler.feature_range[1]) / (scaler.feature_range[2] - scaler.feature_range[1]) * (scaler.max - scaler.min) + scaler.min)

"""
   	inverse_transform(scaler::MinMaxScaler, data::AbstractArray)

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::MinMaxScaler, data::AbstractArray) = inverse_transform!(scaler, copy(data))

"""
   	fit!(data::AbstractArray)

Fit and scale features of `data` according to `feature_range`.

`dims` is the dimention of data to be scaled by. 
"""
function fit_transform(data::AbstractArray; dims::Int = 1, feature_range::Tuple = (0.0, 1.0))
    transform!(MinMaxScaler(data, dims = dims, feature_range = feature_range), data)
end