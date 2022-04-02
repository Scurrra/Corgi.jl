using Statistics

"""
StandardScaler{T, OUTRANGE} <: AbstractTransformer{T, OUTRANGE}

Standardize features by removing the mean and scaling to unit variance. Create a structure `StandardScaler` instance with specified parameters.

If `with_μ` is true, center the data before scaling. If `with_σ` is true, scale the data to unit variance (or equivalently, unit standard deviation). 
"""
mutable struct StandardScaler{T, OUTRANGE} <: AbstractTransformer{T, OUTRANGE}
    with_μ::Bool
    with_σ::Bool

    μ::Matrix{Float64}
    σ::Matrix{Float64}

    features::Vector{Union{Int, Symbol, String}}

	function StandardScaler{T, OUTRANGE}(data::AbstractMatrix{<:Real}; features=:, with_μ::Bool=true, with_σ::Bool=true) where {T, OUTRANGE}
        μ = !with_μ ? zeros(1, size(data, 2)) : mean(data, dims=1) .|> Float64
        σ = !with_σ ? ones(1, size(data, 2)) : std(data, dims=1) .|> Float64
	    new{T, OUTRANGE}(with_μ, with_σ, μ, σ, features == (:) ? collect(1:size(data, 2)) : collect(features))
    end
end
StandardScaler{T, OUTRANGE}(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true) where {T<:AbstractDataFrame, OUTRANGE} = StandardScaler{T, OUTRANGE}(data[!, features] |> Matrix; features=features, with_μ=with_μ, with_σ=with_σ)

"""
    StandardScaler{T}(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2, <:Real}=(0., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
StandardScaler{T}(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2, <:Real}=(0., 1.)) where {T<:AbstractDataFrame} = StandardScaler{T, outrange}(data[!, features] |> Matrix; features=features, with_μ=with_μ, with_σ=with_σ)
StandardScaler{T}(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2, <:Real}=(0., 1.)) where {T<:AbstractMatrix{<:Real}} = StandardScaler{T, outrange}(data[:, features]; features=features, with_μ=with_μ, with_σ=with_σ)

"""
    StandardScaler(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2, <:Real}=(0., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
StandardScaler(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2, <:Real}=(0., 1.)) where {T} = StandardScaler{T}(data; features=features, with_μ=with_μ, with_σ=with_σ, outrange=outrange)

"""
    transform!(scaler::StandardScaler{T, OUTRANGE}, data::T)

Perform standardization by centering and scaling.
"""
transform!(scaler::StandardScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = @.(data[:, scaler.features] = (data[:, scaler.features] - scaler.μ) / scaler.σ * (OUTRANGE[2] - OUTRANGE[1]) + OUTRANGE[1])

"""
    transform(scaler::StandardScaler{T, OUTRANGE}, data::T)

Perform standardization by centering and scaling.
"""
transform(scaler::StandardScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = transform!(scaler, copy(data))


"""
    inverse_transform!(scaler::StandardScaler{T, OUTRANGE}, data::T)

Scale back the `data` to the original representation.
"""
inverse_transform!(scaler::StandardScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = @.(data[:, scaler.features] = (data[:, scaler.features] - OUTRANGE[1]) / (OUTRANGE[2] - OUTRANGE[1]) * scaler.σ + scaler.μ)

"""
   	inverse_transform(scaler::StandardScaler{T, OUTRANGE}, data::T)

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::StandardScaler{T, OUTRANGE}, data::T) where {T, OUTRANGE} = inverse_transform!(scaler, copy(data))