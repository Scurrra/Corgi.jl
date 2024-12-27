import Corgi: AbstractLinearModel, Logging, Utils

"""
    AdaDelta <: AbstractOptimizer

Adaptive Delta is same as RMSProp, but learning rate (should) be replaced by root mean square of weights delta
(so it changes with every batch), `batchsize` equals 1 by default. 
"""
struct AdaDelta <: AbstractOptimizer
    regularizer::AbstractRegularizer
    lrscheduler::AbstractLRScheduler

    γ::AbstractFloat
    epochs::Int
    early_stopping::Bool
    batchsize::Int
    seed::Int

    J::Function
    ∇J::Function

    function AdaDelta(
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
function (opt::AdaDelta)(
    model::AbstractLinearModel, data::AbstractMatrix, target::AbstractVector;
    logger::Logging.AbstractLogger=Logging.NullLogger()
)
    ϵ = 10^-6
    indexes = Utils.Shuffler(Vector(1:size(data, 1)); seed=opt.seed)

    if !opt.early_stopping
        Eg = 0
        EΔ = 0
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)

            for indx in Utils.split(indexes(), opt.batchsize)
                g = opt.∇J(model.ω, data[indx, :], target[indx]) .+ opt.regularizer(model.ω)[2] / length(indx)
                Eg = opt.γ * Eg + (1 - opt.γ) * sum(x -> x^2, g)
                Δ = sqrt(EΔ + ϵ) / sqrt(Eg + ϵ) * g
                EΔ = opt.γ * EΔ + (1 - opt.γ) * sum(x -> x^2, Δ)
                model.ω .-= lr * g
            end

            logger(opt.J(model.ω, data, target) .+ opt.regularizer(model.ω)[1] / length(target))
        end
    else
        Eg = 0
        EΔ = 0
        ω = model.ω
        last_cost = opt.J(ω, data, target) .+ opt.regularizer(ω)[1] / length(target)
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)

            for indx in Utils.split(indexes(), opt.batchsize)
                g = opt.∇J(ω, data[indx, :], target[indx]) .+ opt.regularizer(ω)[2] / length(indx)
                Eg = opt.γ * Eg + (1 - opt.γ) * sum(x -> x^2, g)
                Δ = sqrt(EΔ + ϵ) / sqrt(Eg + ϵ) * g
                EΔ = opt.γ * EΔ + (1 - opt.γ) * sum(x -> x^2, Δ)
                ω .-= lr * g
            end

            current_cost = opt.J(ω, data, target) .+ opt.regularizer(ω)[1] / length(target)
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

export AdaDelta