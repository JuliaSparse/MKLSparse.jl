# MKLSparse.jl

`MKLSparse.jl` is a Julia package to override sparse-dense operations when MKL is available. It also provides an interface to MKL's Direct Sparse Solver which can be used to factorize and solve sparse system of equations.

In order to use `MKLSparse.jl`you need to have built Julia with MKL as the BLAS library. For instructions how to do so, please see [this link](https://github.com/JuliaLang/julia#intel-compilers-and-math-kernel-library-mkl)

### Matrix multiplication

Loading `MKLSparse.jl` will make sparse-dense matrix operation be computed using MKL. Sparse-sparse operations are not yet wrapped.


### Factorization

After `MKLSparse.jl` is loaded, the methods `factorize`, `lufact`, `cholfact`, `ldltfact` will now instead be computed using MKL. The returned factorization objects can be used to solve equations with backslash just like normally. MKL will also be used if the `A\B` syntax is used and `A` is sparse.

Note, due to an unresolved ambiguity problem you currently need to use `MKLSparse.DSS.lufact` to use the lu factorization.
