using DataFrames

"""
    OneHotEncoder(data::AbstractDataFrame; features::Vector{Union{String,Symbol}} = [], classes::Dict{Union{String,Symbol},Union{String,Symbol}})

Encode categorical features as a one-hot numeric array.

`features` is a vector of all columns to be encoded. `classes` is a dict of all possible classes for every feature in `features`.

NOTE: For now every feature for encoding must be specified. If it's not every column in `data` will be in `features`.
"""
struct OneHotEncoder <: AbstractTransformer
    features::Vector{String}
    classes::Dict{String, Vector{String}}

    function OneHotEncoder(data::AbstractDataFrame; features::Vector{String}=String[], classes::Dict{String, String} = Dict{String, String}())
        features = length(features) == 0 ? names(data) : string.(features)
    
        classes = length(classes) == 0 ? Dict(
            feature => string.(unique(data[!, feature]))
            for feature in features
        ) : classes
    
        new(features, classes)
    end
end

"""
   	transform!(encoder::OneHotEncoder, data::AbstractDataFrame)

Transform `data` with OneHotEncoder `encoder`.
"""
function transform!(encoder::OneHotEncoder, data::AbstractDataFrame)
    for feature in encoder.features
        data[!, feature] = string.(data[!, feature])
        for class in encoder.classes[feature]
            data[!, Symbol(
                replace(feature * "[" * string(class) * "]", " " => "_")
            )] = Int8.(data[!, feature] .== class)
        end
        select!(data, Not(feature))
    end

    return data
end

"""
   	transform(encoder::OneHotEncoder, data::AbstractDataFrame)

Transform `data` with OneHotEncoder `encoder`.
"""
transform(encoder::OneHotEncoder, data::AbstractDataFrame) = transform!(encoder, copy(data))