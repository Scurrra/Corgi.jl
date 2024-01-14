module Regularization

"""
    AbstractRegularizer

Regularizing base type.
"""
abstract type AbstractRegularizer end

"""
    NullRegularizer <: AbstractRegularizer

Null regularizer.
"""
struct NullRegularizer <: AbstractRegularizer end
(reg::NullRegularizer)(ω::AbstractArray) = (zeros(typeof(ω[1]), size(ω)), zeros(typeof(ω[1]), size(ω)))

"""
    Regularizer{λ₁,λ₂} <: AbstractRegularizer

Basic regularizer type, where `λ₁` is a coefficient for the Lasso regularization and `λ₂` is a coefficient for the Ridge regularization.
"""
struct Regularizer{λ₁<:AbstractFloat,λ₂<:AbstractFloat} <: AbstractRegularizer
    function Regularizer(λ₁::AbstractFloat, λ₂::AbstractFloat)
        new{λ₁,λ₂}()
    end
end
(reg::Regularizer{λ₁,λ₂})(ω::AbstractArray) where {λ₁,λ₂} = (λ₁ * sum(abs, ω) + λ₂ * sum(abs2, ω), λ₁ .* sign.(ω) + λ₂ .* ω .* 2)

"""
    LassoRegularizer <: AbstractRegularizer

Lasso regularizer.
"""
struct LassoRegularizer{λ<:AbstractFloat} <: AbstractRegularizer
    function LassoRegularizer(λ::AbstractFloat)
        if λ < 0 || λ > 1
            throw("λ must be in range [0, 1]")
        end

        new{λ}()
    end
end
(reg::LassoRegularizer{λ})(ω::AbstractArray) where {λ} = (λ * sum(abs, ω), λ .* sign.(ω))

"""
    RidgeRegularizer <: AbstractRegularizer

Ridge regularizer.
"""
struct RidgeRegularizer{λ<:AbstractFloat} <: AbstractRegularizer
    function RidgeRegularizer(λ::AbstractFloat)
        if λ < 0 || λ > 1
            throw("λ must be in range [0, 1]")
        end

        new{λ}()
    end
end
(reg::RidgeRegularizer{λ})(ω::AbstractArray) where {λ} = (λ * sum(abs2, ω), λ .* ω .* 2)

"""
    ElasticRegularizer <: AbstractRegularizer

Elastic regularizer, where `λ` is a coefficient for the Lasso regularization and `(1 - λ)` is a coefficient for the Ridge regularization.
"""
struct ElasticRegularizer{λ<:AbstractFloat} <: AbstractRegularizer
    function ElasticRegularizer(λ::AbstractFloat)
        if λ < 0 || λ > 1
            throw("λ must be in range [0, 1]")
        end

        new{λ}()
    end
end
(reg::ElasticRegularizer{λ})(ω::AbstractArray) where {λ} = (λ * sum(abs, ω) + (1 - λ) * sum(abs2, ω), λ .* sign.(ω) + (1 - λ) .* ω .* 2)

export AbstractRegularizer
export Regularizer, LassoRegularizer, RidgeRegularizer, ElasticRegularizer

end