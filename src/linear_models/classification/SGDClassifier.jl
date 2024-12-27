import Corgi: Logging, Optimization

struct SGDClassifier <: AbstractLinearClassification
    weights::Vector
    regularizer::Optimization.Regularization.AbstractRegularizer
    lrscheduler::Optimization.LRScheduling.AbstractLRScheduler

    SGDClassifier(N::Int; 
        regularizer::Optimization.Regularization.AbstractRegularizer=NullRegularizer(),
        lrscheduler::Optimization.LRScheduling.AbstractLRScheduler=ConstantLR(1),
        init::Function=zeros
    ) = N > 0 ? new(init(N), C) : throw("N <= 0")
end
SGDClassifier(data::AbstractMatrix; regularizer=NullRegularizer(), lrscheduler=ConstantLR(1)) = SGDClassifier{size(data,1)}(; regularizer=regularizer, lrscheduler=lrscheduler)
SGDClassifier(data::AbstractMatrix, init::Function; regularizer=NullRegularizer(), lrscheduler=ConstantLR(1)) = SGDRegressor{size(data,1)}(; regularizer=regularizer, lrscheduler=lrscheduler, init=init)

(reg::SGDClassifier)(data::AbstractVector) = length(reg.weights) == length(data) ? sum(reg.weights .* data) : throw("Dimension Mismatch")
(reg::SGDClassifier)(data::AbstractMatrix) = length(reg.weights) == size(data, 2) ? data * reg.weights : throw("Dimension Mismatch")

function fit!(reg::SGDClassifier, data::AbstractMatrix, target::AbstractVector;
        logger::Logging.AbstractLogger=Logging.NullLogger(),
        epochs::Int=1000,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42)

    @assert length(reg.weights) == size(data, 2) "Dimension Mismatch"
    @assert length(target) == size(data, 1) "Dimension Mismatch"

    loss = L2HingeLoss()
    J = (w, d, t) -> loss(d*w, t) / size(d, 1)
    ∇J = (w, d, t) -> deriv(loss, d*w, t) / size(d, 1)

    (
        Optimization.SGD(
            J, ∇J;
            regularizer=reg.regularizer,
            lrscheduler=reg.lrscheduler,
            epochs=epochs,
            early_stopping=early_stopping,
            batchsize=batchsize,
            seed=seed
        )
    )(reg, data, target; logger=logger)
end

export SGDRegressor, fit!