using Distributions
using Optim: minimizer, optimize
using Zygote: gradient # to be removed

Φ(x::Float64) = cdf(Normal(), x)
p(n::Int) = (collect(1:n) .- 1 / 3) ./ (n + 1 / 3)
ρ(x::Float64, c::Float64=0.5) = abs(x) > c ? 1 : 1 - (1 - (x / c)^2)^3

# Box-Cox
g(λ::Float64, y::Float64) = λ == 0 ? log(y) : (y^λ - 1) / λ
g(λ::Float64, y::Vector{Float64}) = (x -> g(λ, x)).(y)
∂g(λ::Float64, y::Float64) = gradient(x -> g(λ, x), y)[1]
∂g(λ::Float64, y::Vector{Float64}) = (x -> ∂g(λ, x)).(y)
g̊(λ::Float64, y::Float64, C) = λ < 1 ?
                            (y <= C[2] ? g(λ, y) : g(λ, C[2]) + (y - C[2]) * ∂g(λ, C[2])) :
                            (y >= C[1] ? g(λ, y) : g(λ, C[1]) + (y - C[1]) * ∂g(λ, C[1]))
g̊(λ::Float64, y::Vector{Float64}) = (x -> g̊(λ, x, quantile(y, [1 // 4, 3 // 4]))).(y)

# Yeo-Johnson
h(λ::Float64, y::Float64) = y >= 0 ?
                         (λ == 0 ? log(1 + y) : ((1 + y)^λ - 1) / λ) :
                         (λ == 2 ? -log(1 - y) : -((1 - y)^(2 - λ) - 1) / (2 - λ))
h(λ::Float64, y::Vector{Float64}) = (x -> h(λ, x)).(y)
∂h(λ::Float64, y::Float64) = gradient(x -> h(λ, x), y)[1]
∂h(λ::Float64, y::Vector{Float64}) = (x -> ∂h(λ, x)).(y)
h̊(λ::Float64, y::Float64, C) = λ < 1 ?
                            (y <= C[2] ? h(λ, y) : h(λ, C[2]) + (y - C[2]) * ∂h(λ, C[2])) :
                            (y >= C[1] ? h(λ, y) : h(λ, C[1]) + (y - C[1]) * ∂h(λ, C[1]))
h̊(λ::Float64, y::Vector{Float64}) = (x -> h̊(λ, x, quantile(y, [1 // 4, 3 // 4]))).(y)

"""
    Apply a power transform featurewise to make data more Gaussian-like.
PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. 
This implementation uses [this paper](https://arxiv.org/abs/2005.07946)
"""
struct PowerTransformer{TYPE} <: AbstractTransformer
    λ::Float64

    PowerTransformer{TYPE}(λ::Float64) where {TYPE} = new{TYPE}(λ)
end

function PowerTransformer{:BC}(data::Vector{Float64})
    # Step 1
    f1(λ::Float64) =
        let g_y = g̊(λ, data)
            -ρ.((g_y .- mean(g_y)) ./ std(g_y) .- Φ.(p(length(data))) .^ -1) |> sum
        end
    λ_0 = minimizer(optimize(f1, -5.0, 5.0))::Float64

    # Step 2
    g_λ_0 = g(λ_0, data)
    w = Int.(abs.(g_λ_0 .- mean(g_λ_0)) .<= std(g_λ_0) / Φ(0.995))
    f2(λ::Float64, w = w) =
        let g_λ = g(λ, data)
            μ = sum(w .* g_λ) / sum(w)
            σ = sum(w .* (g_λ .- μ) .^ 2) / sum(w)
            -sum(w .* ((λ - 1) * log.(data) .- 1 / 2 * log(σ)))
        end
    λ_1 = minimizer(optimize(f2, -5.0, 5.0))::Float64

    # Step 3
    g_λ_1 = g(λ_1, data)
    w = Int.(abs.(g_λ_1 .- mean(g_λ_1)) .<= std(g_λ_1) / Φ(0.995))
    λ_2 = minimizer(optimize(f2, -5.0, 5.0))::Float64

    PowerTransformer{:BC}(λ_2)
end

function PowerTransformer{:YJ}(data::Vector{Float64})
    # Step 1
    f1(λ::Float64) =
        let h_y = h̊(λ, data)
            -ρ.((h_y .- mean(h_y)) ./ std(h_y) .- Φ.(p(length(data))) .^ -1) |> sum
        end
    λ_0 = minimizer(optimize(f1, -5.0, 5.0))::Float64
    
    # Step 2
    h_λ_0 = h(λ_0, data)
    w = Int.(abs.(h_λ_0 .- mean(h_λ_0)) .<= std(h_λ_0) / Φ(0.995))
    f2(λ::Float64, w = w) =
        let h_λ = h(λ, data)
            μ = sum(w .* h_λ) / sum(w)
            σ = sum(w .* (h_λ .- μ) .^ 2) / sum(w)
            -sum(w .* ((λ - 1) .* sign.(data) .* log.(abs.(data) .+ 1) .- 1 / 2 * log(σ)))
        end
    λ_1 = minimizer(optimize(f2, -5.0, 5.0))::Float64
    
    # Step 3
    h_λ_1 = h(λ_1, data)
    w = Int.(abs.(h_λ_1 .- mean(h_λ_1)) .<= std(h_λ_1) / Φ(0.995))
    λ_2 = minimizer(optimize(f2, -5.0, 5.0))::Float64
    PowerTransformer{:YJ}(λ_2)
end


"""
   	transform!(transformer::PowerTransformer, data::Vector{Float64})
Apply the power transform to `data`.
"""
transform!(transformer::PowerTransformer{:BC}, data::Vector{Float64}) = data = g(transformer.λ, data)
transform!(transformer::PowerTransformer{:YJ}, data::Vector{Float64}) = data = h(transformer.λ, data)

"""
   	transform(transformer::PowerTransformer, data::Vector{Float64})
Apply the power transform to `data`.
"""
transform(transformer::PowerTransformer{TYPE}, data::Vector{Float64}) where {TYPE} = transform!(transformer, copy(data))

function h_inv(λ::Float64, y::Number)
    # y >= 0
    x = λ == 0 ? exp(1 + y) : (y * λ + 1)^(1 / λ) - 1
    if x >= 0
        return x
    end

    # y < 0
    x = λ == 2 ? 1 - exp(-y) : 1 - (1 - y * (2 - λ))^(1 / (2 - λ))
    if x < 0
        return x
    end

    return nothing
end

"""
   	inverse_transform!(transformer::PowerTransformer, data::Vector{<:Real})
Apply the inverse power transform to `data`.
"""
inverse_transform!(transformer::PowerTransformer{:BC}, data::Vector{Float64}) = @.(data = (transformer.λ == 0 ? exp(data) : (data * transformer.λ + 1) ^ (1 / transformer.λ)));
inverse_transform!(transformer::PowerTransformer{:YJ}, data::Vector{Float64}) = @.(data = (x -> h_inv(transformer.λ, x))(data));                                                                     

"""
   	inverse_transform(transformer::PowerTransformer, data::Vector{<:Real})
Apply the inverse power transform to `data`.
"""
inverse_transform(transformer::PowerTransformer{TYPE}, data::Vector{Float64}) where {TYPE} = inverse_transform!(transformer, copy(data))