include("../utils.jl")

"""
    PolynomialFeatures(; degree::NTuple{2, Int}=(1,1), interaction::Bool=true, bias::Bool=true)

Generate polynomial and interaction features. `degree` is a tuple of two elements? where the first one is the minimal degree and the second one is the maximum one. If `interaction` is true only interaction features are produced: features that are products of at most degree distinct input features. If `bias` is true, the bias column is added. 
"""
struct PolynomialFeatures
    degree::NTuple{2,Int}
    interaction::Bool
    bias::Bool

    PolynomialFeatures(; degree::NTuple{2,Int}=(1, 1), interaction::Bool=true, bias::Bool=true) = new(degree, interaction, bias)
end

"""
    transform(scaler::PolynomialFeatures, data::AbstractMatrix)

Transform `data` to polynomial features, where `data` is of type AbstractMatrix.
"""
function transform(scaler::PolynomialFeatures, data::AbstractMatrix)
    n = size(data, 2)

    if scaler.degree[2] >= 1
        if !scaler.interaction
            for degree = max(scaler.degree[1], 2):scaler.degree[2]
                data = [data hcat(map(a -> a .^ degree, eachcol(data))...)]
            end
        else
            for degree = max(scaler.degree[1], 2):scaler.degree[2]
                coefs = cwr(n, degree)
                for coef in coefs
                    data = [data prod(data[:, coef], dims = 2)]
                end
            end
        end
    end

    data = scaler.bias ? [fill(1.0, size(data, 1)) data] : data

    return data
end