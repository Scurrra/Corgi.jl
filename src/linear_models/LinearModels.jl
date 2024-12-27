module LinearModels
   
using LossFunctions, LossFunctions.Traits

import Corgi: AbstractLinearModel

abstract type AbstractLinearRegression <: AbstractLinearModel end

include("regression/LinearRegression.jl")
include("regression/LassoRegression.jl")
include("regression/RidgeRegression.jl")
include("regression/ElasticNet.jl")
include("regression/SGDRegressor.jl")
include("regression/HuberRegressor.jl")

export AbstractLinearRegression

abstract type AbstractLinearClassification <: AbstractLinearModel end

include("classification/LogisticRegression.jl")
include("classification/SGDClassifier.jl")
include("classification/RidgeClassifier.jl")

export AbstractLinearClassification

end