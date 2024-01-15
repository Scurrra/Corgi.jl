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
    Regularizer <: AbstractRegularizer

Basic regularizer type, where `λ₁` is a coefficient for the Lasso regularization and `λ₂` is a coefficient for the Ridge regularization.
"""
struct Regularizer{C,λ₁,λ₂} <: AbstractRegularizer
    function Regularizer(C::Real, λ₁::AbstractFloat, λ₂::AbstractFloat)
        new{C,λ₁,λ₂}()
    end
end
(reg::Regularizer{C,λ₁,λ₂})(ω::AbstractArray) where {C,λ₁,λ₂} = C .* (λ₁ * sum(abs, ω) + λ₂ * sum(abs2, ω), λ₁ .* sign.(ω) + λ₂ .* ω .* 2)

"""
    LassoRegularizer <: AbstractRegularizer

Lasso regularizer.
"""
struct LassoRegularizer{C,λ} <: AbstractRegularizer
    function LassoRegularizer(C::Real, λ::AbstractFloat)
        if λ < 0 || λ > 1
            throw("λ must be in range [0, 1]")
        end

        new{C,λ}()
    end
end
(reg::LassoRegularizer{C,λ})(ω::AbstractArray) where {C,λ} = C .* (λ * sum(abs, ω), λ .* sign.(ω))

"""
    RidgeRegularizer <: AbstractRegularizer

Ridge regularizer.
"""
struct RidgeRegularizer{C,λ} <: AbstractRegularizer
    function RidgeRegularizer(C::Real, λ::AbstractFloat)
        if λ < 0 || λ > 1
            throw("λ must be in range [0, 1]")
        end

        new{C,λ}()
    end
end
(reg::RidgeRegularizer{C,λ})(ω::AbstractArray) where {C,λ} = C .* (λ * sum(abs2, ω), λ .* ω .* 2)

"""
    ElasticRegularizer <: AbstractRegularizer

Elastic regularizer, where `λ` is a coefficient for the Lasso regularization and `(1 - λ)` is a coefficient for the Ridge regularization.
"""
struct ElasticRegularizer{C,λ} <: AbstractRegularizer
    function ElasticRegularizer(C::Real, λ::AbstractFloat)
        if λ < 0 || λ > 1
            throw("λ must be in range [0, 1]")
        end

        new{C,λ}()
    end
end
(reg::ElasticRegularizer{C,λ})(ω::AbstractArray) where {C,λ} = C .* (λ * sum(abs, ω) + (1 - λ) * sum(abs2, ω), λ .* sign.(ω) + (1 - λ) .* ω .* 2)

export AbstractRegularizer, NullRegularizer
export Regularizer, LassoRegularizer, RidgeRegularizer, ElasticRegularizer

end