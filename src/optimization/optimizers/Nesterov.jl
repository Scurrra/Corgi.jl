import Corgi: AbstractLinearModel, Logging, Utils

"""
    Nesterov <: AbstractOptimizer

Nesterov Accelerated Gradient (γ=0.9 by default), `batchsize` equals 1 by default.
"""
struct Nesterov <: AbstractOptimizer
    regularizer::AbstractRegularizer
    lrscheduler::AbstractLRScheduler

    γ::AbstractFloat
    epochs::Int
    early_stopping::Bool
    batchsize::Int
    seed::Int

    J::Function
    ∇J::Function

    function Nesterov(
        J::Function, ∇J::Function;
        regularizer::AbstractRegularizer=NullRegularizer(),
        lrscheduler::AbstractLRScheduler=ConstantLR(1),
        γ::AbstractFloat=0.9,
        epochs::Int=1000,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42
    )
        if !(0 < γ < 1)
            throw("0 < γ < 1")
        end
        
        if epochs <= 0
            throw("Number of epochs must be greater tan 0")
        end

        if batchsize <= 0
            throw("Number of samples in a batch (batchsize) must be greater tan 0")
        end

        if seed <= 0
            throw("Seed must be greater tan 0")
        end

        new(
            regularizer,
            lrscheduler,
            γ,
            epochs,
            early_stopping,
            batchsize,
            seed,
            J, ∇J
        )
    end
end
function (opt::Nesterov)(
    model::AbstractLinearModel, data::AbstractMatrix, target::AbstractVector;
    logger::Logging.AbstractLogger=Logging.NullLogger()
)
    indexes = Utils.Shuffler(Vector(1:size(data, 1)); seed=opt.seed)

    if !opt.early_stopping
        v = zeroes(length(model.ω))
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)

            for indx in Utils.split(indexes(), opt.batchsize)
                u = opt.γ * v .+ lr * (opt.∇J(model.ω .- opt.γ * v, data[indx], target) .+ opt.regularizer(model.ω .- opt.γ * v)[2] / length(indx))
                model.ω .-= u
                v = u
            end

            logger(opt.J(model.ω, data, target) .+ opt.regularizer(model.ω)[1] / length(data))
        end
    else
        ω = model.ω
        last_cost = opt.J(ω, data, target) .+ opt.regularizer(ω)[1] / length(data)
        v = zeroes(length(ω))
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)
            for indx in Utils.split(indexes(), opt.batchsize)
                u = opt.γ * v .+ lr * (opt.∇J(ω .- opt.γ * v, data[indx], target) .+ opt.regularizer(ω .- opt.γ * v)[2] / length(indx))
                ω .-= u
                v = u
            end

            current_cost = opt.J(ω, data, target) .+ opt.regularizer(ω)[1] / length(data)
            if last_cost > current_cost
                last_cost = current_cost
                logger(current_cost)
            else
                break
            end
        end
        model.ω .= ω
    end

    if !(typeof(logger) <: Logging.NullLogger)
        return logger
    end
end

export Nesterov