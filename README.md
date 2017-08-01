# MKLSparse.jl

`MKLSparse.jl` is a Julia package to seamlessly use the sparse functionality in MKL to speed up operations on sparse arrays in Julia.
In order to use `MKLSparse.jl` you need to have MKL installed and the environment variables `MKLROOT` correctly set, see the [MKL getting started guide]( https://software.intel.com/en-us/articles/intel-mkl-103-getting-started) for a guide. You do not need to have built Julia with MKL as the used BLAS library to use the package.

### Matrix multiplication

Loading `MKLSparse.jl` will make sparse-dense matrix operations be computed using MKL. Note that this means that ALL sparse-dense computations in Julia,
even those in other packages, will automatically use MKL. 

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
