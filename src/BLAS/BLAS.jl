module BLAS

using LinearAlgebra, SparseArrays
using LinearAlgebra: BlasInt, checksquare

# For testing purposes:
global const __counter = Ref(0)


# TODO: BLAS1

# BLAS and BLAS3
include(joinpath("level_2_3", "matdescra.jl"))
include(joinpath("level_2_3", "generator.jl"))
include(joinpath("level_2_3", "matmul.jl"))

end
