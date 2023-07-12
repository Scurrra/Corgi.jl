# Preprocessing Guide

**Corgi.jl** provides you with the following preprocessing methods:
 - `MaxAbsScaler` 
 - `MinMaxScaler`
 - `StandardScaler`
 - `PolynomialFeatures`
 - `PowerTransformer`
 - `OneHotEncoding`

This transformations are performed with using structs that store all necessary information. Let's consider each method in more detail.

```jldoctest preprocessing
julia> using Corgi.Preprocessing 
```

> Note, that DataFrames.jl shamelessly defines `transform` function that way that it is impossible to be overrode. `DataFrames.jl` should be imported without this function.

## MaxAbsScaler

*MaxAbsScaler* scales specified features by it's maximum absolute value. 

By default it scales each single column (feature) with the corresponding maximum absolute value to the range [-1.0, 1.0]. Scaler can work with Matrices and DataFrames. 

```julia
MaxAbsScaler(data::T; features)
```

`data` is the dataset on which the scaler learns its parameters. It can be of type Matrix or DataFrame.

`features` supposed to be Colon (by default) or Vector of column numbers or names (in case when `data` is a DataFrame, `features` most likely is of type `Vector{Symbol}` or `Vector{String}`).

```jldoctest preprocessing
julia> data = [1 2 3 4; -2 -3 -4 -5; 3 -4 5 -6; 4 5 6 7; -5 -6 -7 -8] .|> Float64
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> scaler = MaxAbsScaler(data)
MaxAbsScaler{Colon}([5.0 6.0 7.0 8.0], Colon())

julia> data_transformed = transform(scaler, data)
5×4 view(::Matrix{Float64}, :, :) with eltype Float64:
  0.2   0.333333   0.428571   0.5
 -0.4  -0.5       -0.571429  -0.625
  0.6  -0.666667   0.714286  -0.75
  0.8   0.833333   0.857143   0.875
 -1.0  -1.0       -1.0       -1.0

julia> data_inv = inverse_transform(scaler, data_transformed)
5×4 view(::Matrix{Float64}, :, :) with eltype Float64:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> transform!(scaler, data); data
5×4 Matrix{Float64}:
  0.2   0.333333   0.428571   0.5
 -0.4  -0.5       -0.571429  -0.625
  0.6  -0.666667   0.714286  -0.75
  0.8   0.833333   0.857143   0.875
 -1.0  -1.0       -1.0       -1.0

julia> inverse_transform!(scaler, data); data
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0
```

As you see, firstly we fit the scaler and than we can transform data both with creating new Matrix or inplace.

## MinMaxScaler

*MinMaxScaler* scales specified features by it's minimum and maximum values.

By default it scales each single column (feature) with the corresponding minimum and maximum values to the range [0.0, 1.0]. Scaler can work with Matrices and DataFrames. 

```julia
MinMaxScaler(data::T; features)
```

`data` is the dataset on which the scaler learns its parameters. It can be of type Matrix or DataFrame.

`features` supposed to be Colon (by default) or Vector of column numbers or names (in case when `data` is a DataFrame, `features` most likely is of type `Vector{Symbol}` or `Vector{String}`).

`outrange` is the range transformed data should be scaled to. It is easily performed because by default scaler produces values in range [0.0, 1.0].

```jldoctest preprocessing
julia> data = [1 2 3 4; -2 -3 -4 -5; 3 -4 5 -6; 4 5 6 7; -5 -6 -7 -8] .|> Float64
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> scaler = MinMaxScaler(data)
MinMaxScaler{(0.0, 1.0), Colon}([-5.0 -6.0 -7.0 -8.0], [4.0 5.0 6.0 7.0], Colon())

julia> data_transformed = transform(scaler, data)
5×4 view(::Matrix{Float64}, :, :) with eltype Float64:
 0.666667  0.727273  0.769231  0.8
 0.333333  0.272727  0.230769  0.2
 0.888889  0.181818  0.923077  0.133333
 1.0       1.0       1.0       1.0
 0.0       0.0       0.0       0.0

julia> data_inv = inverse_transform(scaler, data_transformed)
5×4 view(::Matrix{Float64}, :, :) with eltype Float64:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> transform!(scaler, data); data
5×4 Matrix{Float64}:
 0.666667  0.727273  0.769231  0.8
 0.333333  0.272727  0.230769  0.2
 0.888889  0.181818  0.923077  0.133333
 1.0       1.0       1.0       1.0
 0.0       0.0       0.0       0.0

julia> inverse_transform!(scaler, data); data
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0
```

As you see, firstly we fit the scaler and than we can transform data both with creating new Matrix or inplace.

## StandardScaler

*StandardScaler* scales specified features by it's mean and variance.

By default it standardize each single column (feature) by removing the main and scaling to unit variance. Scaler can work with Matrices and DataFrames. 

```julia
StandardScaler(data::T; features=:, with_μ::Bool=true, with_σ::Bool=true)
```

`data` is the dataset on which the scaler learns its parameters. It can be of type Matrix or DataFrame.

`features` supposed to be Colon (by default) or Vector of column numbers or names (in case when `data` is a DataFrame, `features` most likely is of type `Vector{Symbol}` or `Vector{String}`).


```jldoctest preprocessing
julia> data = [1 2 3 4; -2 -3 -4 -5; 3 -4 5 -6; 4 5 6 7; -5 -6 -7 -8] .|> Float64
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> scaler = StandardScaler(data)
StandardScaler{Colon}(true, true, [0.2 -1.2 0.6 -1.6], [3.7013511046643495 4.54972526643093 5.770615218501404 6.6558245169174945], Colon())

julia> data_transformed = transform(scaler, data)
5×4 view(::Matrix{Float64}, :, :) with eltype Float64:
  0.216137   0.703339   0.4159     0.841368
 -0.594378  -0.395628  -0.797142  -0.510831
  0.756481  -0.615422   0.762484  -0.661075
  1.02665    1.36272    0.935775   1.2921
 -1.40489   -1.05501   -1.31702   -0.961564

julia> data_inv = inverse_transform(scaler, data_transformed)
5×4 view(::Matrix{Float64}, :, :) with eltype Float64:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> transform!(scaler, data); data
5×4 Matrix{Float64}:
  0.216137   0.703339   0.4159     0.841368
 -0.594378  -0.395628  -0.797142  -0.510831
  0.756481  -0.615422   0.762484  -0.661075
  1.02665    1.36272    0.935775   1.2921
 -1.40489   -1.05501   -1.31702   -0.961564

julia> inverse_transform!(scaler, data); data
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0
```

As you see, firstly we fit the scaler and than we can transform data both with creating new Matrix or inplace.

## PolynomialFeatures

*PolynomialFeatures* construct polynolial features for matrix data.

Generate polynomial and interaction features. 

```julia
PolynomialFeatures(; degree::NTuple{2, Int}=(1,1), interaction::Bool=true, bias::Bool=true)
```

`degree` is a tuple of two elements, where the first one is the minimal degree and the second one is the maximum one. 

If `interaction` is true only interaction features are produced: features that are products of at most degree distinct input features. 

`bias` specifies if bias column should be added. 

```jldoctest preprocessing
julia> data = [1 2 3 4; -2 -3 -4 -5; 3 -4 5 -6; 4 5 6 7; -5 -6 -7 -8] .|> Float64
5×4 Matrix{Float64}:
  1.0   2.0   3.0   4.0
 -2.0  -3.0  -4.0  -5.0
  3.0  -4.0   5.0  -6.0
  4.0   5.0   6.0   7.0
 -5.0  -6.0  -7.0  -8.0

julia> transformer = PolynomialFeatures(; degree=(2, 2), interaction=true, bias=true)
PolynomialFeatures((2, 2), true, true)

julia> transform(transformer, data)
5×15 Matrix{Float64}:
 1.0   1.0   2.0   3.0   4.0   1.0    2.0  …    6.0   8.0   9.0   12.0  16.0
 1.0  -2.0  -3.0  -4.0  -5.0   4.0    6.0      12.0  15.0  16.0   20.0  25.0
 1.0   3.0  -4.0   5.0  -6.0   9.0  -12.0     -20.0  24.0  25.0  -30.0  36.0
 1.0   4.0   5.0   6.0   7.0  16.0   20.0      30.0  35.0  36.0   42.0  49.0
 1.0  -5.0  -6.0  -7.0  -8.0  25.0   30.0      42.0  48.0  49.0   56.0  64.0
```

For producing polynomial features we don't need to fit data, so structure constructor just set the transformer options.

## OneHotEncoding

*OneHotEncoder* performs one-hot encoding on the dataset.

Constructs `DataFrame`, where all specified `features` are one-hot encoded.

```julia
OneHotEncoder(data::AbstractDataFrame; features::Vector{<:Union{String,Symbol}}, classes::Dict{<:Union{String,Symbol},Vector{<:Union{Symbol,String}}})
```

This transformer can not determine, which `features` are categorical, so they should be specified, or no transformations will be applied. OneHotEncoder can detect classes, but in some cases the training `data` can don't contain all the classes for some features, so you may specify them by your own. 

```jldoctest preprocessing
julia> using DataFrames: DataFrame

julia> data = DataFrame(gender=["M", "F", "M", "M", "T", "F"], occupation=["student", "student", "cook", "professor", "barista", "data satanist"])
6×2 DataFrame
 Row │ gender  occupation    
     │ String  String        
─────┼───────────────────────
   1 │ M       student
   2 │ F       student
   3 │ M       cook
   4 │ M       professor
   5 │ T       barista
   6 │ F       data satanist

julia> transformer = OneHotEncoder(data; features=[:gender, :occupation])
OneHotEncoder(["gender", "occupation"], Dict("gender" => ["M", "F", "T"], "occupation" => ["student", "cook", "professor", "barista", "data satanist"]))

julia> transform(transformer, data)
6×8 DataFrame
 Row │ gender[M]  gender[F]  gender[T]  occupation[student]  occupation[cook]  ⋯
     │ Int8       Int8       Int8       Int8                 Int8              ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │         1          0          0                    1                 0  ⋯
   2 │         0          1          0                    1                 0
   3 │         1          0          0                    0                 1
   4 │         1          0          0                    0                 0
   5 │         0          0          1                    0                 0  ⋯
   6 │         0          1          0                    0                 0
                                                               3 columns omitted

julia> data
6×2 DataFrame
 Row │ gender  occupation    
     │ String  String        
─────┼───────────────────────
   1 │ M       student
   2 │ F       student
   3 │ M       cook
   4 │ M       professor
   5 │ T       barista
   6 │ F       data satanist

julia> transform!(transformer, data); data
6×8 DataFrame
 Row │ gender[M]  gender[F]  gender[T]  occupation[student]  occupation[cook]  ⋯
     │ Int8       Int8       Int8       Int8                 Int8              ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │         1          0          0                    1                 0  ⋯
   2 │         0          1          0                    1                 0
   3 │         1          0          0                    0                 1
   4 │         1          0          0                    0                 0
   5 │         0          0          1                    0                 0  ⋯
   6 │         0          1          0                    0                 0
                                                               3 columns omitted
```

OneHotEncoder does not provide inverse transform function.