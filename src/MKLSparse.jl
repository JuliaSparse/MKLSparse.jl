__precompile__()

module MKLSparse


const depfile = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(depfile)
    include(depfile)
else
    error("MKLSparse not properly installed. Please run Pkg.build(\"MKLSparse\")")
end

function __init__()
    ccall((:MKL_Set_Interface_Layer, libmkl_rt), Cint, (Cint,), Base.USE_BLAS64 ? 1 : 0)
end

include(joinpath("BLAS", "BLAS.jl"))

end # module
