# Script to parse MKL headers and generate Julia wrappers.
using MKL_Headers_jll
using Clang
using Clang.Generators
using JuliaFormatter

include("rewriter.jl")

function wrapper(name::String, headers::Vector{String}, optimized::Bool=false)

  @info "Wrapping $name"
  cd(@__DIR__)
  include_dir = joinpath(MKL_Headers_jll.artifact_dir, "include")

  options = load_options(joinpath(@__DIR__, "mkl.toml"))
  ignore_list = ["MKL_Complex8", "MKL_Complex16", "MKLVersion", "MKL_INT64",
  "MKL_INT", "MKL_UINT", "MKL_UINT8", "MKL_INT8", "MKL_INT16", "MKL_BF16", "MKL_INT32",
  "MKL_F16", "MKL_DOMAIN_ALL", "MKL_DOMAIN_BLAS", "MKL_DOMAIN_FFT", "MKL_DOMAIN_VML",
  "MKL_DOMAIN_PARDISO", "MKL_DOMAIN_LAPACK", "MKL_CBWR_BRANCH", "MKL_CBWR_ALL",
  "MKL_CBWR_STRICT", "MKL_CBWR_OFF", "MKL_CBWR_UNSET_ALL", "MKL_CBWR_BRANCH_OFF",
  "MKL_CBWR_AUTO", "MKL_CBWR_COMPATIBLE", "MKL_CBWR_SSE2", "MKL_CBWR_SSSE3",
  "MKL_CBWR_SSE4_1", "MKL_CBWR_SSE4_2", "MKL_CBWR_AVX", "MKL_CBWR_AVX2", "MKL_CBWR_AVX512_MIC",
  "MKL_CBWR_AVX512", "MKL_CBWR_AVX512_MIC_E1", "MKL_CBWR_AVX512_E1", "MKL_CBWR_SUCCESS",
  "MKL_CBWR_ERR_INVALID_SETTINGS", "MKL_CBWR_ERR_INVALID_INPUT", "MKL_CBWR_ERR_UNSUPPORTED_BRANCH",
  "MKL_CBWR_ERR_UNKNOWN_BRANCH", "MKL_CBWR_ERR_MODE_CHANGE_FAILURE", "MKL_CBWR_SSE3"]
  optimized && (options["general"]["output_ignorelist"] = ignore_list)

  args = get_default_args()
  push!(args, "-I$include_dir")
  
  ctx = create_context(headers, args, options)
  build!(ctx)

  path = options["general"]["output_file_path"]

  format_file(path, YASStyle())
  optimized && rewrite!(path)
  return nothing
end

function main(; optimized::Bool=false)
  mkl = joinpath(MKL_Headers_jll.artifact_dir, "include")
  wrapper("libmklsparse", ["$mkl/mkl_spblas.h", "$mkl/mkl_sparse_qr.h"], optimized)
end

# If we want to use the file as a script with `julia wrapper.jl`
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
