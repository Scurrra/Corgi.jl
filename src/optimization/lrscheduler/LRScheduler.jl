module LRScheduler

"""
AbstractLRScheduler

Learning rate scheduler base type.
"""
abstract type AbstractLRScheduler end

"""
    DiscreteLRScheduler <: AbstractLRScheduler

Discrete learning rate schedulers: 
"""
abstract type DiscreteLRScheduler <: AbstractLRScheduler end

"""
    ContinuousLRScheduler <: AbstractLRScheduler

Continuous learning rate schedulers: 
"""
abstract type ContinuousLRScheduler <: AbstractLRScheduler end

export StepLR, MultiStepLR, ConstantLR
export LinearLR, ExponentialLR, MultiplicativeLR, LambdaLR, PolynomialLR

include("discrete.jl")
include("continuous.jl")

end