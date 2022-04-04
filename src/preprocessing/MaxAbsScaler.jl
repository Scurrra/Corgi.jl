"""
    MaxAbsScaler{OUTRANGE, FTYPE} <: AbstractScaler{OUTRANGE}

Scaler that transforms `features` data by scaling them to fit range `OUTRANGE`
"""
struct MaxAbsScaler{OUTRANGE, FTYPE<:Union{Colon, Vector{<:Union{Int, Symbol, String}}}} <: AbstractScaler{OUTRANGE}
    max::Matrix{Float64}
    features::FTYPE
    
    function MaxAbsScaler{OUTRANGE}(data::AbstractMatrix{<:Real}; features=:) where {OUTRANGE}
        max = maximum(Float64 ∘ abs, data; dims=1)
        new{Float64.(OUTRANGE), typeof(features)}(max, features)
    end
end
MaxAbsScaler{OUTRANGE}(data::AbstractDataFrame; features=:) where {OUTRANGE} = MaxAbsScaler{OUTRANGE}(data[!, features] |> Matrix; features=features)

"""
    MaxAbsScaler(data::T; features=:, outrange::NTuple{2, <:Real}=(-1., 1.))
    
Construct scaler that scales `features` from `data` to fit `outrange`
"""
MaxAbsScaler(data::AbstractDataFrame; features=:, outrange::NTuple{2, <:Real}=(-1., 1.)) = MaxAbsScaler{outrange}(data[!, features] |> Matrix; features=features)
MaxAbsScaler(data::AbstractMatrix{<:Real}; features=:, outrange::NTuple{2, <:Real}=(-1., 1.)) = MaxAbsScaler{outrange}(data[:, features]; features=features)


"""
   	transform!(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale features of `data` according to `OUTRANGE`.
"""
transform!(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}}) where {OUTRANGE, FTYPE} = @.(data[:, scaler.features] = data[:, scaler.features] / scaler.max * (OUTRANGE[2] - OUTRANGE[1]) + OUTRANGE[1])

"""
   	transform(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale features of `data` according to `OUTRANGE`.
"""
transform(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}}) where {OUTRANGE, FTYPE} = transform!(scaler, copy(data))

"""
   	inverse_transform!(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform!(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}}) where {OUTRANGE, FTYPE} = @.(data[:, scaler.features] = (data[:, scaler.features] - OUTRANGE[1]) / (OUTRANGE[2] - OUTRANGE[1]) * scaler.max)

"""
   	inverse_transform(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}})

Scale back the `data` to the original representation.
"""
inverse_transform(scaler::MaxAbsScaler{OUTRANGE, FTYPE}, data::Union{AbstractDataFrame, AbstractMatrix{<:Real}}) where {OUTRANGE, FTYPE} = inverse_transform!(scaler, copy(data))