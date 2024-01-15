"""
    LinearLR <: ContinuousLRScheduler

Linear learning rate scheduler. Linearly changes the learning rate from `λ η` to `η` until the number of epoch reaches `total_iters`. 
"""
mutable struct LinearLR <: ContinuousLRScheduler
    η::AbstractFloat
    λ::AbstractFloat
    total_iters::Int

    LinearLR(η::AbstractFloat; λ::AbstractFloat=0.1, total_iters::Int=1) = new(η, λ, total_iters)
end
function (lrs::LinearLR)(epoch::Int)
    if epoch <= lrs.total_iters
        minη = lrs.η * lrs.λ
        return minη + (lrs.η - minη) / lrs.total_iters * epoch
    end

    return lrs.η
end

"""
    ExponentialLR <: ContinuousLRScheduler

Exponential learning rate scheduler. Multiplies the learning rate `η` by `γ` until the number of epoch reaches `total_iters`. 
"""
mutable struct ExponentialLR <: ContinuousLRScheduler
    η::AbstractFloat
    γ::AbstractFloat
    total_iters::Int

    ExponentialLR(η::AbstractFloat; γ::AbstractFloat=0.1, total_iters::Int=1) = new(η, λ, total_iters)
end
function (lrs::ExponentialLR)(epoch::Int)
    if epoch <= lrs.total_iters
        lrs.η *= lrs.γ
    end

    return lrs.η
end


"""
    MultiplicativeLR <: ContinuousLRScheduler

Multiplicative learning rate scheduler. Multiplies the learning rate `η` by `λ(epoch)` until the number of epoch reaches `total_iters`. 
"""
mutable struct MultiplicativeLR <: ContinuousLRScheduler
    η::AbstractFloat
    λ::Function
    total_iters::Int

    MultiplicativeLR(η::AbstractFloat; λ::Function=(x)->x, total_iters::Int=1) = new(η, λ, total_iters)
end
function (lrs::MultiplicativeLR)(epoch::Int)
    if epoch <= lrs.total_iters
        lrs.η *= lrs.λ(epoch)
    end

    return lrs.η
end

"""
    LambdaLR <: ContinuousLRScheduler

Lambda learning rate scheduler. Multiplies the initial learning rate `η` by `λ(epoch)` until the number of epoch reaches `total_iters`. 
"""
struct LambdaLR <: ContinuousLRScheduler
    η::AbstractFloat
    λ::Function
    total_iters::Int

    LambdaLR(η::AbstractFloat; λ::Function=(x)->x, total_iters::Int=1) = new(η, λ, total_iters)
end
function (lrs::LambdaLR)(epoch::Int)
    if epoch <= lrs.total_iters
        return lrs.η * lrs.λ(epoch)
    end

    return lrs.η
end

"""
    PolynomialLR <: ContinuousLRScheduler

Polynomial learning rate scheduler. Decays the initial learning rate `η` using polynomial function `η = ηᵢ * (1 - epoch / total_iters) ^ α` until the number of epoch reaches `total_iters`. 
"""
struct PolynomialLR <: ContinuousLRScheduler
    η::AbstractFloat
    α::AbstractFloat
    total_iters::Int

    LambdaLR(η::AbstractFloat; α::AbstractFloat=1.0, total_iters::Int=1) = new(η, α, total_iters)
end
function (lrs::PolynomialLR)(epoch::Int)
    if epoch <= lrs.total_iters
        return lrs.η * (1 - epoch / lrs.total_iters) ^ lrs.α
    end

    return lrs.η
end