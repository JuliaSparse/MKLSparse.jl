type_modifications = Dict("Cint"            => "BlasInt",
                          "Cfloat"          => "Float32",
                          "Cdouble"         => "Float64",
                          "MKL_Complex8"    => "ComplexF32",
                          "MKL_Complex16"   => "ComplexF64")

cstring_modifications = Dict("transa::Cstring"    => "transa::Ref{UInt8}",
                             "uplo::Cstring"      => "uplo::Ref{UInt8}",
                             "diag::Cstring"      => "diag::Ref{UInt8}",
                             "matdescra::Cstring" => "matdescra::Ptr{UInt8}")

function rewrite!(path::String)
  text = read(path, String)
  for (keys, vals) in type_modifications
    text = replace(text, keys => vals)
  end
  for (keys, vals) in cstring_modifications
    text = replace(text, keys => vals)
  end
  for argument in ("m", "n", "k", "job", "alpha", "beta", "lda", "ldb", "ldc") 
    for T in ("BlasInt", "Float32", "Float64", "ComplexF32", "ComplexF64")
      text = replace(text, "$argument::Ptr{$T}" => "$argument::Ref{$T}")
    end
  end
  write(path, text)
end
