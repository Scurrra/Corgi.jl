import Corgi: AbstractLinearModel, Logging, Utils

"""
    AdaMax <: AbstractOptimizer

Adaptive Moment Estimation based on infinity norm, `batchsize` equals 1 by default. 
"""
struct AdaMax <: AbstractOptimizer
    regularizer::AbstractRegularizer
    lrscheduler::AbstractLRScheduler

    β₁::AbstractFloat
    β₂::AbstractFloat
    epochs::Int
    early_stopping::Bool
    batchsize::Int
    seed::Int

    J::Function
    ∇J::Function

    function AdaMax(
        J::Function, ∇J::Function;
        regularizer::AbstractRegularizer=NullRegularizer(),
        lrscheduler::AbstractLRScheduler=ConstantLR(0.002),
        β₁::AbstractFloat=0.9,
        β₂::AbstractFloat=0.999,
        epochs::Int=1000,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42
    )
        if !(0 < β₁ < 1)
            throw("0 << β₁ < 1") 
        end

        if !(0 < β₂ < 1)
            throw("0 << β₂ < 1") 
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
            β₁, β₂,
            epochs,
            early_stopping,
            batchsize,
            seed,
            J, ∇J
        )
    end
end
function (opt::AdaMax)(
    model::AbstractLinearModel, data::AbstractMatrix, target::AbstractVector; 
    logger::Logging.AbstractLogger=Logging.NullLogger()
)
    ϵ = 10^-8
    indexes = Utils.Shuffler(Vector(1:size(data, 1)); seed=opt.seed)

    if !opt.early_stopping
        m = zeros(length(model.ω))
        v = 0
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)

            for indx in Utils.split(indexes(), opt.batchsize)
                g = opt.∇J(model.ω, data[indx], target) .+ opt.regularizer(model.ω)[2] / length(indx)
                m = opt.β₁ * m .+ (1 - opt.β₂) * g
                v = max(opt.β₂ * v, sqrt(sum(x->x^2, g)) + ϵ)
                model.ω .-= lr / (1 - opt.β₁^epoch) / v * m
            end

            logger(opt.J(model.ω, data, target) .+ opt.regularizer(model.ω)[1] / length(data))
        end
    else
        m = zeros(length(model.ω))
        v = 0
        ω = model.ω
        last_cost = opt.J(ω, data, target) .+ opt.regularizer(ω)[1] / length(data)
        for epoch in 1:opt.epochs
            lr = opt.lrscheduler(epoch)

            for indx in Utils.split(indexes(), opt.batchsize)
                g = opt.∇J(ω, data[indx], target) .+ opt.regularizer(ω)[2] / length(indx)
                m = opt.β₁ * m .+ (1 - opt.β₂) * g
                v = max(opt.β₂ * v, sqrt(sum(x->x^2, g)) + ϵ)
                ω .-= lr / (1 - opt.β₁^epoch) / v * m
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

export AdaMax