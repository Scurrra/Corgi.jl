import Corgi: Logging, Optimization

struct RidgeClassifier <: AbstractLinearClassification
    weights::Vector
    C

    RidgeClassifier(N::Int; C=1.0, init::Function=zeros) = N > 0 ? new(init(N), C) : throw("N <= 0")
end
RidgeClassifier(data::AbstractMatrix; C=1.0) = RidgeClassifier{size(data,1)}(; C=C)
RidgeClassifier(data::AbstractMatrix, init::Function; C=1.0) = RidgeClassifier{size(data,1)}(; C=C, init=init)

(reg::RidgeClassifier)(data::AbstractVector) = length(reg.weights) == length(data) ? sum(reg.weights .* data) : throw("Dimension Mismatch")
(reg::RidgeClassifier)(data::AbstractMatrix) = length(reg.weights) == size(data, 2) ? data * reg.weights : throw("Dimension Mismatch")

function fit!(reg::RidgeClassifier, data::AbstractMatrix, target::AbstractVector;
        logger::Logging.AbstractLogger=Logging.NullLogger(),
        opt::Optimization.AbstractOptimizer=Optimization.SGD,
        epochs::Int=1000,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42)

    @assert length(reg.weights) == size(data, 2) "Dimension Mismatch"
    @assert length(target) == size(data, 1) "Dimension Mismatch"

    target[target.==0] .= -1

    loss = L2DistLoss()
    J = (w, d, t) -> loss(d*w, t) / size(d, 1)
    ∇J = (w, d, t) -> deriv(loss, d*w, t) / size(d, 1)

    (
        opt(
            J, ∇J;
            regularizer=Optimization.Regularization.RidgeRegularizer(reg.C, 1.0),
            epochs=epochs,
            early_stopping=early_stopping,
            batchsize=batchsize,
            seed=seed
        )
    )(reg, data, target; logger=logger)
end

export RidgeRegression, fit!