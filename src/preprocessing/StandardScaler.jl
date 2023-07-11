using Statistics

"""
    StandardScaler{OUTRANGE} <: AbstractScaler{OUTRANGE}

Standardize features by removing the mean and scaling to unit variance. Create a structure `StandardScaler` instance with specified parameters.

If `with_μ` is true, center the data before scaling. If `with_σ` is true, scale the data to unit variance (or equivalently, unit standard deviation). 
"""
mutable struct StandardScaler{OUTRANGE,FTYPE<:Union{Colon,Vector{<:Union{Int,Symbol,String}}}} <: AbstractScaler{OUTRANGE}
    with_μ::Bool
    with_σ::Bool

    μ::Matrix{Float64}
    σ::Matrix{Float64}

    features::FTYPE

    function StandardScaler{OUTRANGE}(data::AbstractMatrix{<:Real}; features=:, with_μ::Bool=true, with_σ::Bool=true) where {OUTRANGE}
        μ = !with_μ ? zeros(1, size(data, 2)) : mean(data, dims=1) .|> Float64
        σ = !with_σ ? ones(1, size(data, 2)) : std(data, dims=1) .|> Float64
        new{OUTRANGE,typeof(features)}(with_μ, with_σ, μ, σ, features)
    end
end
StandardScaler{OUTRANGE}(data::AbstractDataFrame; features=:, with_μ::Bool=true, with_σ::Bool=true) where {OUTRANGE} = StandardScaler{OUTRANGE}(data[!, features] |> Matrix; features=features, with_μ=with_μ, with_σ=with_σ)

"""
    StandardScaler(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2, <:Real}=(0., 1.))
    
Construct scaler that scales `features` from `data` of type `T` to fit `outrange`
"""
StandardScaler(data::AbstractDataFrame; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2,<:Real}=(0.0, 1.0)) = StandardScaler{outrange}(data[!, features] |> Matrix; features=features, with_μ=with_μ, with_σ=with_σ)
StandardScaler(data::AbstractMatrix{<:Real}; features=:, with_μ::Bool=true, with_σ::Bool=true, outrange::NTuple{2,<:Real}=(0.0, 1.0)) = StandardScaler{outrange}(data[:, features]; features=features, with_μ=with_μ, with_σ=with_σ)

"""
    transform!(scaler::StandardScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Perform standardization by centering and scaling.
"""
transform!(scaler::StandardScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = @.(data[:, scaler.features] = (data[:, scaler.features] - scaler.μ) / scaler.σ * (OUTRANGE[2] - OUTRANGE[1]) + OUTRANGE[1])

"""
    transform(scaler::StandardScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Perform standardization by centering and scaling.
"""
transform(scaler::StandardScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = transform!(scaler, copy(data))


"""
    inverse_transform!(scaler::StandardScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform!(scaler::StandardScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = @.(data[:, scaler.features] = (data[:, scaler.features] - OUTRANGE[1]) / (OUTRANGE[2] - OUTRANGE[1]) * scaler.σ + scaler.μ)

"""
   	inverse_transform(scaler::StandardScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::StandardScaler{OUTRANGE,FTYPE}, data::Union{AbstractDataFrame,AbstractMatrix{<:Real}}) where {OUTRANGE,FTYPE} = inverse_transform!(scaler, copy(data))