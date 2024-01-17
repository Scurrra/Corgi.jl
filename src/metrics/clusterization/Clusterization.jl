module Clusterization

import ..Metrics: AbstractMetric

"""
    AbstractClusterizationMetric <: AbstractMetric

Base type for clsterization metrics.
"""
abstract type AbstractClusterizationMetric <: AbstractMetric end

"""
    AbstractInnerClusterizationMetric <: AbstractClusterizationMetric

Base type for inner clsterization metrics.
"""
abstract type AbstractInnerClusterizationMetric <: AbstractClusterizationMetric end

"""
    AbstractOuterClusterizationMetric <: AbstractClusterizationMetric

Base type for outer clsterization metrics.
"""
abstract type AbstractOuterClusterizationMetric <: AbstractClusterizationMetric end


end