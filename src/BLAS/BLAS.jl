module BLAS

using LinearAlgebra, SparseArrays
using LinearAlgebra: BlasInt, checksquare
using MKL_jll: libmkl_rt

# For testing purposes:
global const __counter = Ref(0)

include("enums.jl")
include("types.jl")

# TODO: BLAS1

# BLAS and BLAS3
#include(joinpath("level_2_3", "matdescra.jl"))
include(joinpath("level_2_3", "generator.jl"))
include(joinpath("level_2_3", "matmul.jl"))

end
