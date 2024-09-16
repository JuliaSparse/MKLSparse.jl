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
  ignore_list = [
    # exclude mkl_types.h definitions not related to MKLSparse
    "MKLVersion",
    "MKL_U?INT\\d*", "MKL_B?F\\d+", "_?MKL_Complex\\d+",
    "MKL_DOMAIN_[A-Z]+", "MKL_CBWR_[A-Z0-9_]+",
    # exclude uppercase functions
    "MKL_[A-Z_]+(BSR|COO|CSR|CSC|DIA|SKY)[A-Z0-9_]*"]
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
