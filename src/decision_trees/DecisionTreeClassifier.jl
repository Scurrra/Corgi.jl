gini(p::Vector{Number}) = let a = sum(p) / length(p);
    2 * a * (1 - a)
end

entropy(p::Vector{Number}) = let a = sum(p) / length(p);
    -(a * log(a) + (1-a) * log(1 - a))
end

struct Leaf
    value::Union{Function, AbstractLinearModel}
    voters::Int
end

struct Node
    feature::Union{String, Symbol, Integer, Nothing}
    condition::Function

    left::Union{Node, Leaf, Nothing}
    right::Union{Node, Leaf, Nothing}
end
(node::Node)(data) = let comp = node.condition(data);
    ismissing(comp) && return [node.left, node.right]
    comp ? node.left : node.right
end

mutable struct DecisionTreeClassifier <: AbstractDecisionTree
    root::Union{Node, Nothing}
    criteria::Function
    splitter::Symbol

    max_depth::Int
    min_samples_per_leaf::Int
    use_linear_models::Bool
    leaf_model<:AbstractLinearClassification
    leaf_function::Function

    features::Vector{Union{Symbol, String, Integer}}
    categorical_features::Vector{Union{Symbol, String, Integer}}
    
    categorical_feature_splitter::Symbol
    numerical_feature_splitter::Symbol
    max_bins::Integer

    feature_connection::Symbol

    rng

    function DecisionTreeClassifier(
            features::Vector{Union{Symbol, String, Integer}};
            criteria::Function=gini,
            splitter::Symbol=:BEST,
            categorical_features::Vector{Union{Symbol, String, Integer}}=[],
            categorical_feature_splitter::Symbol=:OneToMany,
            numerical_feature_splitter::Symbol=:STUPID,
            max_bins::Integer=255, # just like in LightGBM
            feature_connection::Symbol=:DROP,
            max_depth::Int=-1,
            min_samples_per_leaf::Int=1,
            use_linear_models::Bool=false,
            leaf_model<:AbstractLinearClassification=LogisticRegression,
            leaf_function::Function=mean,
            seed::Int=42
        )
        
        @assert min_samples_per_leaf > 0 "Samples per leaf should be positive number"
        @assert max_depth > 0 || max_depth == -1 "Min depth should be positive number"
        @assert categorical_feature_splitter == :OneToMany || categorical_feature_splitter == :ManyToMany
        @assert numerical_feature_splitter == :STUPID || numerical_feature_splitter == :BINS || numerical_feature_splitter == :CLUSTERS
        @assert max_bins > 0
        @assert splitter == :BEST || splitter == :RANDOM || splitter == :RANDOM_BEST "Splitter can be `:BEST` or `:RANDOM`, `:RANDOM_BEST"
        @assert feature_connection == :DROP || feature_connection == :SKIP "Features may be either dropped (`:DROP`) or skipped (`:SKIP`)"
        @assert length(features) > 0 "Empty tree?"
        
        new(
            nothing,
            criteria,
            splitter,
            max_depth,
            min_samples_per_leaf,
            use_linear_models,
            leaf_model,
            leaf_function,
            features,
            categorical_features,
            categorical_feature_splitter,
            numerical_feature_splitter,
            feature_connection,
            Xoshiro(seed)
        )
    end
end


function fit!(tree::DecisionTreeClassifier, data<:Union{AbstractMatrix, AbstractDataFrame}, target::BitVector)
    @assert length(target) == size(data, 1) "Dimension Mismatch"

    histogram = tree.numerical_feature_splitter == :BINS ? Dict(
        feature => begin
            values = sort(data[.!(ismissing.(data[:, feature])), feature])
            step = div(length(values), tree.max_bins - Int(length(values) % tree.max_bins == 0) + 1)
            values[(step+1):step:end]
        end
        for feature in tree.features[.!(tree.features.∈(tree.categorical_features,))]
    ) : nothing

    
    function fit_node(features::Vector{Union{Symbol, String, Integer}}, indices::Vector{Int}, features_used::BitVector, feature_skipped=1, depth=tree.depth)
        feature_id = 1
        feature = features[feature_id]
        predicate = nothing
        
        #TODO: leafs
        if depth == 0
            return Leaf(
                data -> tree.leaf_function(target[indices]),
                length(indices)
            )
        elseif depth == -1
            depth = 0
        end

        if tree.splitter == :RANDOM
            feature_id = rand(tree.rng, findall(==(1), features_used))
            feature = features[feature_id]
        else
            best = []
            for f in (tree.feature_connection==:DROP ? features : features[features_used])
                
                best_split = (f, nothing) => 0

                if f in tree.categorical_features
                
                    classes = data[indices, f] |> unique
                    classes = classes[classes.!==missing]
                    length(classes) == 1 && continue 

                    # TODO: compress if-block
                    if tree.categorical_feature_splitter == :OneToMany || length(classes) == 2
                    
                        for class in classes
                            indx = data[indices, f] .== class
                            indm = data[indices, f] .=== missing
                            indl, indr = indx .| indm, .!(indx) .| indm
                            (sum(indl) < tree.min_samples_per_leaf || sum(indr) < tree.min_samples_per_leaf) && continue
                                
                            gain = sum(indl) * tree.criteria(target[indices][indl]) + 
                                sum(indr) * tree.criteria(target[indices][indr])
                            gain /= length(indices)

                            if gain > last(best_split)
                                best_split = (f, class) => gain
                            end
                        end

                    elseif tree.categorical_feature_splitter == :ManyToMany

                        for ind in powerset(classes, 1, div(length(classes), 2))
                            indx = data[indices, f] .∈ (classes[ind],)
                            indm = data[indices, f] .=== missing
                            indl, indr = indx .| indm, .!(indx) .| indm
                            (sum(indl) < tree.min_samples_per_leaf || sum(indr) < tree.min_samples_per_leaf) && continue
                                
                            gain = sum(indl) * tree.criteria(target[indices][indl]) + 
                                sum(indr) * tree.criteria(target[indices][indr])
                            gain /= length(indices)

                            if gain > last(best_split)
                                best_split = (f, classes[ind]) => gain
                            end
                        end
    
                    end

                else
                    
                    if tree.numerical_feature_splitter == :STUPID
                        # you should never use it, but it's default
                        
                        best_split = (f, nothing) => 0
                        for num in data[indices, f]
                            indx = data[indices, f] .<= num
                            indm = data[indices, f] .=== missing
                            indl, indr = indx .| indm, .!(indx) .| indm
                            (sum(indl) < tree.min_samples_per_leaf || sum(indr) < tree.min_samples_per_leaf) && continue
                            
                            gain = sum(indl) * tree.criteria(target[indices][indl]) + 
                                sum(indr) * tree.criteria(target[indices][indr])
                            gain /= length(indices)

                            if gain > last(best_split)
                                best_split = (f, num) => gain
                            end
                        end

                    elseif tree.numerical_feature_splitter == :BINS

                        best_split = (f, data[indices[1], f]) => 0
                        for num in histogram[f]
                            indx = data[indices, f] .<= num
                            indm = data[indices, f] .=== missing
                            indl, indr = indx .| indm, .!(indx) .| indm
                            (sum(indl) < tree.min_samples_per_leaf || sum(indr) < tree.min_samples_per_leaf) && continue
                            
                            gain = sum(indl) * tree.criteria(target[indices][indl]) + 
                                sum(indr) * tree.criteria(target[indices][indr])
                            gain /= length(indices)

                            if gain > last(best_split)
                                best_split = (f, num) => gain
                            end
                        end

                    elseif tree.numerical_feature_splitter == :CLUSTERS

                        indx = sortperm(data[indices, f])
                        indm = findfirst(ismissing, data[indices, f][indx])
                        clusters = group(target[indices[indx[1:(indm-1)]]])
                    
                        best_split = (f, data[indx[1], f]) => 0
                        for ind in cumsum(last.(clusters))
                            indl, indr = [1:ind; indm:length(indices)], (ind+1):length(indices)
                            (length(indl) < tree.min_samples_per_leaf || length(indr) < tree.min_samples_per_leaf) && continue
                            
                            gain = length(indl) * tree.criteria(target[indices[indx]][indl]) + 
                                length(indr) * tree.criteria(target[indices[indx]][indr])
                            gain /= length(indices)

                            if gain > last(best_split)
                                best_split = (f, num) => gain
                            end
                        end
    
                    end

                end

                if best_split[1][2] !== nothing                        
                    push!(best, best_split)
                end

            end

            if length(best) == 1
                return Leaf(
                    data -> tree.leaf_function(target[indices]),
                    length(indices)
                )
            end

            if tree.splitter == :BEST
                @show best
                (feature, predicate), gain = argmax(last, best)
                @show feature, predicate, gain
            elseif tree.splitter == :RANDOM_BEST
                @show best
                (feature, predicate), gain = sample(tree.rng, best, Weights(last.(best)))
                @show feature, predicate, gain
            end

        end

        if tree.feature_connection == :SKIP                
            features_used[feature_skipped] = 1
            feature_skipped = feature_id
            features_used[feature_skipped] = 0         
        else
            deleteat!(features, feature_id)
        end

        return feature in tree.categorical_features ? Node(
            feature,
            (data) -> data[feature] == predicate,
            fit_node(
                features, 
                indices[(data[:, feature].==predicate)[indices]], 
                features_used, 
                feature_skipped,
                depth-1
            ),
            fit_node(
                features, 
                indices[(data[:, feature].!==predicate)[indices]], 
                features_used, 
                feature_skipped,
                depth-1
            )
        ) : Node(
            feature,
            (data) -> data[feature] <= predicate,
            fit_node(
                features, 
                indices[(data[:, feature].<=predicate)[indices]], 
                features_used, 
                feature_skipped,
                depth-1
            ),
            fit_node(
                features, 
                indices[(data[:, feature].>predicate)[indices]], 
                features_used, 
                feature_skipped,
                depth-1
            )
        )
    end
    
end

function (tree::DecisionTreeClassifier)(data)
    leafs = Leaf[]
    nodes = [tree.root(data)]
    while length(nodes) != 0
        node = popfirst!(nodes)
        if typeof(node) <: Leaf
            push!(leafs, node)            
        else
            append!(nodes, node(data))
        end
    end

    length(leafs) == 1 && return leafs[1](data)

    all_voters, value = 0, 0.0
    for leaf in leafs
        all_voters += leaf.voters
        value += leaf.value(data)        
    end
    return value / all_voters 
end