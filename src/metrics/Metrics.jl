module Metrics
    
"""
    AbstractMetric

Base metric type. There are two ways to obtain metric: using metrc function itself 
and by constructing metric structure. The second way let's you to pass metric type to some function
having an ability to check what exactly you pass to it. 
"""
abstract type AbstractMetric end

include("clusterization/Clusterization.jl")

end