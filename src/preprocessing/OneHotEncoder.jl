using DataFrames

"""
    OneHotEncoder(data::AbstractDataFrame; features::Vector{Union{String,Symbol}} = [], classes::Dict{Union{String,Symbol},Union{String,Symbol}})

Encode categorical features as a one-hot numeric array.

`features` is a vector of all columns to be encoded. `classes` is a dict of all possible classes for every feature in `features`.

NOTE: For now every feature for encoding must be specified. If it's not every column in `data` will be in `features`.
"""
struct OneHotEncoder
    features::Vector{String}
    classes::Dict{String,Any}

    function OneHotEncoder(data::AbstractDataFrame; features::Vector{String} = String[], classes::Dict{String,Any} = Dict{String,Any}())
        features = length(features) == 0 ? names(data) : features
    
        classes = length(classes) == 0 ? Dict(
            feature => unique(data[!, feature])
            for feature in features
        ) : data
    
        new(features, classes)
    end
end

"""
   	transform!(scaler::OneHotEncoder, data::AbstractDataFrame)

Transform `data` with OneHotEncoder `scaler`.
"""
function transform!(scaler::OneHotEncoder, data::AbstractDataFrame)
    for feature in scaler.features
        for class in scaler.classes[feature]
            data[!, Symbol(
                replace(string(feature) * "[" * class * "]", " " => "_")
            )] = Int8.(data[!, feature] .== class)
        end
        select!(data, Not(feature))
    end

    return data
end

"""
   	transform(scaler::OneHotEncoder, data::AbstractDataFrame)

Transform `data` with OneHotEncoder `scaler`.
"""
transform(scaler::OneHotEncoder, data::AbstractDataFrame) = transform!(scaler, copy(data))