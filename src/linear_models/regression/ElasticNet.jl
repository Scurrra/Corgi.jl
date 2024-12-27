import Corgi: Logging, Optimization

struct ElasticNet <: AbstractLinearRegression
    weights::Vector
    C
    l1_ratio

    ElasticNet(N::Int; C=1.0, l1_ratio=0.5, init::Function=zeros) = N > 0 ? new(init(N), C) : throw("N <= 0")
end
ElasticNet(data::AbstractMatrix; C=1.0, l1_ratio=0.5) = ElasticNet{size(data,1)}(; C=C, l1_ratio=l1_ratio)
ElasticNet(data::AbstractMatrix, init::Function; C=1.0, l1_ratio=0.5) = ElasticNet{size(data,1)}(; C=C, l1_ratio=l1_ratio, init=init)

(reg::ElasticNet)(data::AbstractVector) = length(reg.weights) == length(data) ? sum(reg.weights .* data) : throw("Dimension Mismatch")
(reg::ElasticNet)(data::AbstractMatrix) = length(reg.weights) == size(data, 2) ? data * reg.weights : throw("Dimension Mismatch")

function fit!(reg::ElasticNet, data::AbstractMatrix, target::AbstractVector;
        logger::Logging.AbstractLogger=Logging.NullLogger(),
        opt::Optimization.AbstractOptimizer=Optimization.SGD,
        epochs::Int=1000,
        early_stopping::Bool=false,
        batchsize::Int=1,
        seed::Int=42)

    @assert length(reg.weights) == size(data, 2) "Dimension Mismatch"
    @assert length(target) == size(data, 1) "Dimension Mismatch"

    loss = L2DistLoss()
    J = (w, d, t) -> loss(d*w, t) / size(d, 1)
    ∇J = (w, d, t) -> deriv(loss, d*w, t) / size(d, 1)

    (
        opt(
            J, ∇J;
            regularizer=Optimization.Regularization.ElasticRegularizer(reg.C, reg.l1_ratio),
            epochs=epochs,
            early_stopping=early_stopping,
            batchsize=batchsize,
            seed=seed
        )
    )(reg, data, target; logger=logger)
end

export ElasticNet, fit!