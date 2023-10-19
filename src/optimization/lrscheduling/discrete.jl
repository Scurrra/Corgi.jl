"""
    StepLR <: DiscreteLRScheduler

Step learning rate scheduler. Decays the learning rate `η` by `γ` each `step`. 
"""
mutable struct StepLR <: DiscreteLRScheduler
    η::AbstractFloat
    γ::AbstractFloat
    step::Int8

    StepLR(η::AbstractFloat; γ::AbstractFloat=0.1, step::Int=1) = new(η, γ, step)
end
function (lrs::StepLR)(epoch::Int)
    if epoch % lrs.step == 0
        lrs.η *= lrs.γ
    end

    return lrs.η
end

"""
    MultiStepLR <: DiscreteLRScheduler

Multistep learning rate scheduler. Decays the learning rate `η` by `γ` on each of `milestones`. 
"""
mutable struct MultiStepLR <: DiscreteLRScheduler
    η::AbstractFloat
    γ::AbstractFloat
    milestones::AbstractArray{Int8}

    MultiStepLR(η::AbstractFloat; γ::AbstractFloat=0.1, milestones::AbstractArray{Int8}=[]) = new(η, γ, unique(sort(milestones)))
end
function (lrs::MultiStepLR)(epoch::Int)
    if epoch ∈ lrs.milestones
        lrs.η *= lrs.γ
    end

    return lrs.η
end

"""
    ConstantLR <: DiscreteLRScheduler

Constant learning rate scheduler. Decays the learning rate `η` by `γ` until the number of epoch reaches `total_iters`. 
"""
struct ConstantLR <: DiscreteLRScheduler
    η::AbstractFloat
    λ::AbstractFloat
    total_iters::Int8

    ConstantLR(η::AbstractFloat; λ::AbstractFloat=0.1, total_iters::Int=1) = new(η, λ, total_iters)
end
function (lrs::ConstantLR)(epoch::Int)
    if epoch <= lrs.total_iters
        return lrs.η * lrs.λ
    end

    return lrs.η
end
