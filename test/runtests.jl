using MKLSparse
using Base.Test

srand(1234321)
sA = sprand(1000,100,0.01)
sS = sA'sA
sTl = tril(sS)
sTu = triu(sS)

include("./dss.jl")
include("./matdescra.jl")
include("./matmul.jl")

