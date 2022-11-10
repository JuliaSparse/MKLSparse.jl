# Wrapping headers

This directory contains scripts that can be used to automatically generate wrappers for C headers by Intel MKL libraries.
This is done using Clang.jl.

# Usage

Either run `julia wrapper.jl` directly, or include it and call the `main()` function.
Be sure to activate the project environment in this folder, which will install `MKL_Headers_jll`, `Clang.jl` and `JuliaFormatter.jl`.
The `main` function supports the boolean keyword argument `optimized` to clear the generated wrappers.

# Remark

You should always review any changes to the headers!
Specifically, verify that pointer arguments are of the correct type, and if they aren't, modify the `rewriter.jl` file and regenerate the wrappers.
The `Ref` type should be considered as an alternative to plain `Ptr` if the pointer represents a scalar or single-value argument.
