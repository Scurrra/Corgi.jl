using Statistics

"""
    StandardScaler <: AbstractScaler

Standardize features by removing the mean and scaling to unit variance. Create a structure `StandardScaler` instance with specified parameters.

If `with_μ` is true, center the data before scaling. If `with_σ` is true, scale the data to unit variance (or equivalently, unit standard deviation). 
"""
mutable struct StandardScaler{FTYPE<:Union{Colon,Vector{<:Union{Int,Symbol,String}}}} <: AbstractScaler
    with_μ::Bool
    with_σ::Bool

    μ::Matrix{Float64}
    σ::Matrix{Float64}

    features::FTYPE

    function StandardScaler(data::AbstractMatrix{<:Real}; features=:, with_μ::Bool=true, with_σ::Bool=true)
        μ = !with_μ ? zeros(1, size(data, 2)) : mean(data, dims=1) .|> Float64
        σ = !with_σ ? ones(1, size(data, 2)) : std(data, dims=1) .|> Float64
        new{typeof(features)}(with_μ, with_σ, μ, σ, features)
    end
end

"""
    StandardScaler(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true)
    
Construct scaler that scales `features` from `data` of type `T`.
"""
StandardScaler(data::AbstractDataFrame; features=:, with_μ::Bool=true, with_σ::Bool=true) = StandardScaler(data[!, features] |> Matrix; features=features, with_μ=with_μ, with_σ=with_σ)

"""
    transform!(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Perform standardization by centering and scaling.
"""
transform!(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = @.(data[:, scaler.features] = (data[:, scaler.features] - scaler.μ) / scaler.σ)

"""
    transform(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Perform standardization by centering and scaling.
"""
transform(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = transform!(scaler, copy(data))


"""
    inverse_transform!(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform!(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = @.(data[:, scaler.features] = data[:, scaler.features] * scaler.σ + scaler.μ)

"""
   	inverse_transform(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::StandardScaler{FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {FTYPE} = inverse_transform!(scaler, copy(data))