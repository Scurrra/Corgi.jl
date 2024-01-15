module Optimization

include("lrscheduling/LRScheduling.jl")
using .LRScheduling

include("regularization/Regularization.jl")
using .Regularization

"""
AbstractOptimizer{η,R,S}

Optimizer base type. `η` -- speed of optimization.
"""
abstract type AbstractOptimizer end

include("optimizers/SGD.jl")
include("optimizers/Momentum.jl")
include("optimizers/Nesterov.jl")
include("optimizers/AdaGrad.jl")
include("optimizers/RMSProp.jl")
include("optimizers/AdaDelta.jl")
include("optimizers/Adam.jl")
include("optimizers/AMSGrad.jl")
include("optimizers/AdaMax.jl")

end