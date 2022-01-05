"""
	MinMaxScaler(; dims::Int=1, feature_range::Tuple=(0., 1.))
	MinMaxScaler(data::AbstractArray; dims::Int=1, feature_range::Tuple=(0., 1.))

Transform features by scaling each feature to a given range. Create a structure `MinMaxScaler` instance with specified parameters.

`dims` is the dimention of data to be scaled by. `feature_range` is the desired range of transformed data.
"""
struct MinMaxScaler
    dims::Int
    feature_range::NTuple{2,Float64}

    min
    max

    MinMaxScaler(; dims::Int = 1, feature_range::NTuple{2,Float64} = (0.0, 1.0)) = new(dims, feature_range)

    function MinMaxScaler(data::AbstractArray; dims::Int = 1, feature_range::NTuple{2,Float64} = (0.0, 1.0))
        min = minimum(data .|> Float64, dims = dims)
        max = maximum(data .|> Float64, dims = dims)
        new(dims, feature_range, min, max)
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
   	fit_transform(data::AbstractArray; dims::Int=1, feature_range::Tuple=(0.0, 1.0))

Fit and scale features of `data` according to `feature_range`.

`dims` is the dimention of data to be scaled by. 
"""
fit_transform(data::AbstractArray; dims::Int = 1, feature_range::NTuple{2,Float64} = (0.0, 1.0)) = transform!(MinMaxScaler(data, dims = dims, feature_range = feature_range), data)