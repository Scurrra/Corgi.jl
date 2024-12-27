import Corgi: Logging, Optimization

struct HuberRegressor <: AbstractLinearRegression
    weights::Vector
    epsilon

    HuberRegressor(N::Int; epsilon=1.35, init::Function=zeros) = N > 0 ? new(init(N), C) : throw("N <= 0")
end
HuberRegressor(data::AbstractMatrix; C=1.0) = HuberRegressor{size(data,1)}(; C=C)
HuberRegressor(data::AbstractMatrix, init::Function; C=1.0) = HuberRegressor{size(data,1)}(; C=C, init=init)

(reg::HuberRegressor)(data::AbstractVector) = length(reg.weights) == length(data) ? sum(reg.weights .* data) : throw("Dimension Mismatch")
(reg::HuberRegressor)(data::AbstractMatrix) = length(reg.weights) == size(data, 2) ? data * reg.weights : throw("Dimension Mismatch")

function fit!(reg::HuberRegressor, data::AbstractMatrix, target::AbstractVector;
        alpha=0.0001,
        logger::Logging.AbstractLogger=Logging.NullLogger(),
        opt::Optimization.AbstractOptimizer=Optimization.SGD,
        epochs::Int=100,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42)

    @assert length(reg.weights) == size(data, 2) "Dimension Mismatch"
    @assert length(target) == size(data, 1) "Dimension Mismatch"

    loss = HuberLoss(1.35)
    J = (w, d, t) -> loss(d*w, t) / size(d, 1)
    ∇J = (w, d, t) -> deriv(loss, d*w, t) / size(d, 1)

    (
        opt(
            J, ∇J;
            regularizer=Optimization.Regularization.RidgeRegularizer(alpha, 1.0),
            epochs=epochs,
            early_stopping=early_stopping,
            batchsize=batchsize,
            seed=seed
        )
    )(reg, data, target; logger=logger)
end

export HuberRegressor, fit!