# Wrapping headers

This directory contains scripts that can be used to automatically generate wrappers for C headers by Intel MKL libraries.
This is done using Clang.jl.

# Usage

Either run `julia wrapper.jl` directly, or include it and call the `main()` function.
Be sure to activate the project environment in this folder, which will install `Clang.jl` and `JuliaFormatter.jl`.
The `main` function supports the boolean keyword argument `optimized` to clear the generated wrappers.
