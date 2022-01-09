using Statistics

"""
	StandardScaler(; dims::Int=1, with_μ::Bool=true, with_σ::Bool=true)
	StandardScaler(data::AbstractArray; dims::Int=1, with_μ::Bool=true, with_σ::Bool=true)

Standardize features by removing the mean and scaling to unit variance. Create a structure `StandardScaler` instance with specified parameters.

`dims` is the dimention of data to be scaled by. If `with_μ` is true, center the data before scaling. If `with_σ` is true, scale the data to unit variance (or equivalently, unit standard deviation). 
"""
mutable struct StandardScaler
    with_μ::Bool
    with_σ::Bool
    dims::Int

    μ
    σ

    StandardScaler(; dims::Int = 1, with_μ::Bool = true, with_σ::Bool = true) = new(with_μ, with_σ, dims)
    
	function StandardScaler(data::AbstractArray; dims::Int = 1, with_μ::Bool = true, with_σ::Bool = true)
        μ = !with_μ ? 0.0 : mean(data .|> Float64, dims = dims)
        σ = !with_σ ? 1.0 : std(data .|> Float64, dims = dims)
	    new(with_μ, with_σ, dims, μ, σ)
    end
end

"""
   	fit!(scaler::StandardScaler, data::AbstractArray)

Fit `data` using existing `scaler` parameters.
"""
function fit!(scaler::StandardScaler, data::AbstractArray)
    scaler = StandardScaler(data, dims = scaler.dims, with_μ = scaler.with_μ, with_σ = scaler.with_σ)
end

"""
   	transform!(scaler::StandardScaler, data::AbstractArray)

Perform standardization by centering and scaling.
"""
transform!(scaler::StandardScaler, data::AbstractArray) = @.(data = (data - scaler.μ) / scaler.σ)

"""
   	transform(scaler::StandardScaler, data::AbstractArray)

Perform standardization by centering and scaling.
"""
transform(scaler::StandardScaler, data::AbstractArray) = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::StandardScaler, data::AbstractArray)

Scale back the `data` to the original representation.
"""
inverse_transform!(scaler::StandardScaler, data::AbstractArray) = @.(data = data * scaler.σ + scaler.μ)

"""
   	inverse_transform(scaler::StandardScaler, data::AbstractArray)

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::StandardScaler, data::AbstractArray) = inverse_transform!(scaler, copy(data))

"""
   	fit_transform(data::AbstractArray)

Fit and perform standardization by centering and scaling `data` with specified parameters.

`dims` is the dimention of data to be scaled by. If `with_μ` is true, center the data before scaling. If `with_σ` is true, scale the data to unit variance(or equivalently, unit standard deviation). 
"""
function fit_transform(data::AbstractArray; dims::Int = 1, with_μ::Bool = true, with_σ::Bool = true)
    transform!(StandardScaler(data, dims = dims, with_μ = with_μ, with_σ = with_σ), copy(data))
end