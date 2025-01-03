module MKLSparse

using LinearAlgebra, SparseArrays
using LinearAlgebra: BlasInt, BlasFloat, checksquare
using MKL_jll: libmkl_rt

# counts total MKL Sparse API calls (for testing purposes)
global const __mklsparse_calls_count = Ref(0)

# increments to the `__mklsparse_calls_count` variable
function _log_mklsparse_call(fname)
    #@debug "$fname called"
    __mklsparse_calls_count[] += 1
end

# common MKL definitions from mkl_service.h, see also MKL.jl

@enum Threading begin
    THREADING_INTEL
    THREADING_SEQUENTIAL
    THREADING_PGI
    THREADING_GNU
    THREADING_TBB
end

@enum Interface begin
    INTERFACE_LP64
    INTERFACE_ILP64
    INTERFACE_GNU
end

function set_threading_layer(layer::Threading = THREADING_INTEL)
    err = @ccall libmkl_rt.MKL_Set_Threading_Layer(layer::Cint)::Cint
    (err == -1) && error("MKL_Set_Threading_Layer() returned -1")
    return nothing
end

function set_interface_layer(interface::Interface = INTERFACE_LP64)
    err = @ccall libmkl_rt.MKL_Set_Interface_Layer(interface::Cint)::Cint
    (err == -1) && error("MKL_Set_Interface_Layer() returned -1")
    return nothing
end

# initialize the MKL interface upon module initialization
# NOTE: this sets the interface for all MKL API calls, not just the sparse ones
function __init__()
    set_interface_layer(INTERFACE_LP64)
end

# Wrappers generated by Clang.jl
include("libmklsparse.jl")
include("types.jl")
include("utils.jl")
include("mklsparsematrix.jl")

# TODO: BLAS1

# BLAS2 and BLAS3
include("deprecated.jl")
include("generic.jl")
include("interface.jl")

export MKLSparseError

end # module
