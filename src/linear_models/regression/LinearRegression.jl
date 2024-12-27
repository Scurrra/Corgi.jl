import Corgi: Logging, Optimization

struct LinearRegression <: AbstractLinearRegression
    weights::Vector

    LinearRegression(N::Int; init::Function=zeros) = N > 0 ? new(init(N)) : throw("N <= 0")
end
LinearRegression(data::AbstractMatrix) = LinearRegression{size(data,1)}()
LinearRegression(data::AbstractMatrix, init::Function) = LinearRegression{size(data,1)}(; init=init)

(reg::LinearRegression)(data::AbstractVector) = length(reg.weights) == length(data) ? sum(reg.weights .* data) : throw("Dimension Mismatch")
(reg::LinearRegression)(data::AbstractMatrix) = length(reg.weights) == size(data, 2) ? data * reg.weights : throw("Dimension Mismatch")

function fit!(reg::LinearRegression, data::AbstractMatrix, target::AbstractVector;
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
            epochs=epochs,
            early_stopping=early_stopping,
            batchsize=batchsize,
            seed=seed
        )
    )(reg, data, target; logger=logger)
end

export LinearRegression, fit!