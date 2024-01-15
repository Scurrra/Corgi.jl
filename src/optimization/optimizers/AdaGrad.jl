import Corgi: AbstractLinearModel, Logging, Utils

"""
    AdaGrad <: AbstractOptimizer

Adaptive Gradient, `batchsize` equals 1 by default. 
Learning rate is supposed to be constabt over time, 'cause there is "the method" to modify it.
"""
struct AdaGrad <: AbstractOptimizer
    regularizer::AbstractRegularizer
    lrscheduler::AbstractLRScheduler

    epochs::Int
    early_stopping::Bool
    batchsize::Int
    seed::Int

    J::Function
    ∇J::Function

    function AdaGrad(
        J::Function, ∇J::Function;
        regularizer::AbstractRegularizer=NullRegularizer(),
        lrscheduler::AbstractLRScheduler=ConstantLR(0.01),
        epochs::Int=1000,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42
    )
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
            epochs,
            early_stopping,
            batchsize,
            seed,
            J, ∇J
        )
    end
end
function (opt::AdaGrad)(
    model::AbstractLinearModel, data::AbstractMatrix, target::AbstractVector; 
    logger::Logging.AbstractLogger=Logging.NullLogger()
)
    ϵ = 10^-10
    indexes = Utils.Shuffler(Vector(1:size(data, 1)); seed=opt.seed)

    if !opt.early_stopping
        G = 0
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)

            for indx in Utils.split(indexes(), opt.batchsize)
                g = opt.∇J(model.ω, data[indx], target) .+ opt.regularizer(model.ω)[2] / length(indx)
                G += sum(x->x^2, g)
                model.ω .-= lr / sqrt(G + ϵ) * g
            end

            logger(opt.J(model.ω, data, target) .+ opt.regularizer(model.ω)[1] / length(data))
        end
    else
        G = 0
        ω = model.ω
        last_cost = opt.J(ω, data, target) .+ opt.regularizer(ω)[1] / length(data)
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)
            for indx in Utils.split(indexes(), opt.batchsize)
                g = opt.∇J(ω, data[indx], target) .+ opt.regularizer(ω)[2] / length(indx)
                G += sum(x->x^2, g)
                ω .-= lr / sqrt(G + ϵ) * g
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

export AdaGrad