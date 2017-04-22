__precompile__()

module MKLSparse

function __init__()
    ccall((:MKL_Set_Interface_Layer, :libmkl_rt), Cint, (Cint,), Base.USE_BLAS64 ? 1 : 0)
end

include(joinpath("BLAS", "BLAS.jl"))

end # module
