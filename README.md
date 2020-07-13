# MKLSparse.jl

`MKLSparse.jl` is a Julia package to seamlessly use the sparse functionality in MKL to speed up operations on sparse arrays in Julia.
In order to use `MKLSparse.jl` you do not need to install Intel's MKL library nor build Julia with MKL. `MKLSparse.jl` will automatically download and use the MKL library for you when installed.

### Matrix multiplications

Loading `MKLSparse.jl` will make sparse-dense matrix operations be computed using MKL.

### Solving linear systems

Solving linear systems with triangular sparse matrices is supported.
These matrices should be wrapped in their corresponding type, for example `LowerTriangular` for lower triangular matrices.

For solving general sparse linear systems using MKL we refer to [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl).

## Misc

* The integer type that should be used in order for MKL to be called is the same as used by the Julia BLAS library, see `Base.USE_BLAS64`.

### Possible TODO's

* Wrap BLAS1 (`SparseVector`)
* Wrap DSS
* Wrap Incomplete LU preconditioners
