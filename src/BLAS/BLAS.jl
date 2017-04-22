module BLAS

using Compat

import Base.LinAlg: checksquare, UnitLowerTriangular, UnitUpperTriangular, BlasInt, BlasFloat

# For testing purposes:
global const __counter = Ref(0)


# TODO: BLAS1

# BLAS and BLAS3
include(joinpath("level_2_3", "matdescra.jl"))
include(joinpath("level_2_3", "generator.jl"))
include(joinpath("level_2_3", "matmul.jl"))

end
