using Distributions
using Optim: optimize

@enum PTType BC YJ

Φ(x::Number) = cdf(Normal(), x)
p(n::Int) = (collect(1:n) .- 1 / 3) ./ (n + 1 / 3)
ρ(x, c = 0.5) = abs(x) > c ? 1 : 1 - (1 - (x / c)^2)^3

# Box-Cox
g(λ::Float64, y) = λ == 0 ? log(y) : (y^λ - 1) / λ
g(λ::Float64, y::Vector) = (x -> g(λ, x)).(y)
∂g(λ::Float64, y) = gradient(x -> g(λ, x), y)[1]
∂g(λ::Float64, y::Vector) = (x -> ∂g(λ, x)).(y)
g̊(λ::Float64, y, C) = λ < 1 ?
                       (y <= C[2] ? g(λ, y) : g(λ, C[2]) + (y - C[2]) * ∂g(λ, C[2])) :
                       (y >= C[1] ? g(λ, y) : g(λ, C[1]) + (y - C[1]) * ∂g(λ, C[1]))
g̊(λ::Float64, y::Vector) = (x -> g̊(λ, x, quantile(y, [1 // 4, 3 // 4]))).(y)

# Yeo-Johnson
h(λ::Float64, y) = y >= 0 ?
                   (λ == 0 ? log(1 + y) : ((1 + y)^λ - 1) / λ) :
                   (λ == 2 ? -log(1 - y) : -((1 - y)^(2 - λ) - 1) / (2 - λ))
h(λ::Float64, y::Vector) = (x -> h(λ, x)).(y)
∂h(λ::Float64, y) = gradient(x -> h(λ, x), y)[1]
∂h(λ::Float64, y::Vector) = (x -> ∂h(λ, x)).(y)
h̊(λ::Float64, y, C) = λ < 1 ?
                       (y <= C[2] ? h(λ, y) : h(λ, C[2]) + (y - C[2]) * ∂h(λ, C[2])) :
                       (y >= C[1] ? h(λ, y) : h(λ, C[1]) + (y - C[1]) * ∂h(λ, C[1]))
h̊(λ::Float64, y::Vector) = (x -> h̊(λ, x, quantile(y, [1 // 4, 3 // 4]))).(y)

"""
    Apply a power transform featurewise to make data more Gaussian-like.
PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. 
This implementation uses [this paper](https://arxiv.org/abs/2005.07946)
"""
struct PowerTransformer
    type::PTType
    λ::Real

    PowerTransformer(; type::PTType = YJ) = new(type)

    function PowerTransformer(data::Vector{<:Real}; type::PTType = YJ)
        if type == BC

            # Step 1
            f1_bc(λ) =
                let g_y = g̊(λ, data)
                    -ρ.((g_y .- mean(g_y)) ./ std(g_y) .- Φ.(p(length(data))) .^ -1) |> sum
                end
            λ_0 = minimizer(optimize(f1_bc, -5.0, 5.0))

            # Step 2
            g_λ_0 = g(λ_0, data)
            w = Int.(abs.(g_λ_0 .- mean(g_λ_0)) .<= std(g_λ_0) / Φ(0.995))
            f2_bc(λ, w = w) =
                let g_λ = g(λ, data)
                    μ = sum(w .* g_λ) / sum(w)
                    σ = sum(w .* (g_λ .- μ) .^ 2) / sum(w)
                    -sum(w .* ((λ - 1) * log.(data) .- 1 / 2 * log(σ)))
                end
            λ_1 = minimizer(optimize(f2_bc, -5.0, 5.0))

            # Step 3
            g_λ_1 = g(λ_1, y)
            w = Int.(abs.(g_λ_1 .- mean(g_λ_1)) .<= std(g_λ_1) / Φ(0.995))
            λ_2 = minimizer(optimize(f2_bc, -5.0, 5.0))

            new(type, λ_2)

        else #if type == YJ

            # Step 1
            f1_jy(λ) =
                let h_y = h̊(λ, data)
                    -ρ.((h_y .- mean(h_y)) ./ std(h_y) .- Φ.(p(length(data))) .^ -1) |> sum
                end
            λ_0 = minimizer(optimize(f1_jy, -5.0, 5.0))

            # Step 2
            h_λ_0 = h(λ_0, data)
            w = Int.(abs.(h_λ_0 .- mean(h_λ_0)) .<= std(h_λ_0) / Φ(0.995))
            f2_jy(λ, w = w) =
                let h_λ = h(λ, data)
                    μ = sum(w .* h_λ) / sum(w)
                    σ = sum(w .* (h_λ .- μ) .^ 2) / sum(w)
                    -sum(w .* ((λ - 1) .* sign.(data) .* log.(abs.(data) .+ 1) .- 1 / 2 * log(σ)))
                end
            λ_1 = minimizer(optimize(f2_jy, -5.0, 5.0))

            # Step 3
            h_λ_1 = h(λ_1, data)
            w = Int.(abs.(h_λ_1 .- mean(h_λ_1)) .<= std(h_λ_1) / Φ(0.995))
            λ_2 = minimizer(optimize(f2_jy, -5.0, 5.0))

            new(type, λ_2)

        end
    end
end

"""
   	fit!(scaler::PowerTransformer, data::Vector{<:Real})
Fit `data` using existing `scaler` parameters.
"""
function fit!(scaler::PowerTransformer, data::Vector{<:Real})
    scaler = PowerTransformer(data, type = scaler.type)
end

"""
   	transform!(scaler::PowerTransformer, data::Vector{<:Real})
Apply the power transform to `data`.
"""
transform!(scaler::PowerTransformer, data::Vector{<:Real}) = scaler.type == BC ? g(scaler.λ, data) : h(scaler.λ, data);

"""
   	transform(scaler::PowerTransformer, data::Vector{<:Real})
Apply the power transform to `data`.
"""
transform(scaler::PowerTransformer, data::Vector{<:Real}) = transform!(scaler, copy(data))

function h_inv(λ::Real, y::Number)
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
   	inverse_transform!(scaler::PowerTransformer, data::Vector{<:Real})
Apply the inverse power transform to `data`.
"""
inverse_transform!(scaler::PowerTransformer, data::Vector{<:Real}) = scaler.type == BC ?
                                                                     (scaler.λ == 0 ? exp.(data) : (data .* scaler.λ .+ 1) .^ (1 / scaler.λ)) :
                                                                     (x -> h_inv(scaler.λ, x)).(data);

"""
   	inverse_transform(scaler::PowerTransformer, data::Vector{<:Real})
Apply the inverse power transform to `data`.
"""
inverse_transform(scaler::PowerTransformer, data::Vector{<:Real}) = inverse_transform!(scaler, copy(data))

"""
   	fit_transform(data::Vector{<:Real}; type::PTType=YJ)
Fit and scale `data` according using the power transformer.
`dims` is the dimention of data to be scaled by. 
"""
function fit_transform(data::Vector{<:Real}; type::PTType = YJ)
    transform!(PowerTransformer(data, type = type), data)
end

