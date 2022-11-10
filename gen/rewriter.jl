type_modifications = Dict("Cint"            => "BlasInt",
                          "Cfloat"          => "Float32",
                          "Cdouble"         => "Float64",
                          "MKL_Complex8"    => "ComplexF32",
                          "MKL_Complex16"   => "ComplexF64")

function rewrite!(path::String)
  text = read(path, String)
  for (keys, vals) in type_modifications
    text = replace(text, keys => vals)
  end
  for (keys, vals) in cstring_modifications
    text = replace(text, keys => vals)
  end
  # Note: `job` and `idiag` are vectors in some cases, we must be careful with these two arguments.
  for argument in ("job", "m", "n", "k", "job", "nnz", "nnzmax", "lval",
                   "lb", "mblk", "idiag", "ldabsr", "ndiag", "ldAbsr", "sort",
                   "alpha", "beta", "lda", "ldb", "ldc", "ierr", "info")
    for T in ("BlasInt", "Float32", "Float64", "ComplexF32", "ComplexF64")
      text = replace(text, "$argument::Ptr{$T}" => "$argument::Ref{$T}")
    end
  end
  for argument in ("transa", "uplo", "diag")
    text = replace(text, "$argument::Ptr{Cchar}" => "$argument::Ref{Cchar}")
  end
  # Remove comments in libmklsparse.jl
  text = replace(text, "# typedef void ( * sgemm_jit_kernel_t ) ( void * , float * , float * , float * )\n" => "")
  text = replace(text, "# typedef void ( * dgemm_jit_kernel_t ) ( void * , double * , double * , double * )\n" => "")
  text = replace(text, "# typedef void ( * cgemm_jit_kernel_t ) ( void * , ComplexF32 * , ComplexF32 * , ComplexF32 * )\n" => "")
  text = replace(text, "# typedef void ( * zgemm_jit_kernel_t ) ( void * , ComplexF64 * , ComplexF64 * , ComplexF64 * )\n" => "")
  text = replace(text, "# Skipping MacroDefinition: MKL_LONG long int\n" => "")
  text = replace(text, "# Skipping MacroDefinition: MKL_DEPRECATED __attribute__ ( ( deprecated ) )\n" => "")
  text = replace(text, "\n\n\n" => "\n")
  write(path, text)
end
