# Script to parse MKL headers and generate Julia wrappers.
using Clang
using Clang.Generators
using JuliaFormatter

include("rewriter.jl")

function wrapper(name::String, headers::Vector{String}, optimized::Bool=false)

  @info "Wrapping $name"
  cd(@__DIR__)
  include_dir = joinpath(@__DIR__, "mkl-include-2022.2.0-intel_8748", "include")

  options = load_options(joinpath(@__DIR__, "mkl.toml"))
  options["general"]["library_name"] = "libmkl_rt"
  options["general"]["output_file_path"] = joinpath("..", "src", "$(name).jl")
  optimized && (options["general"]["output_ignorelist"] = ["MKL_Complex8",
                                                           "MKL_Complex16",
                                                           "MKLVersion"])

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
  # TODO: Add mkl_spblas.h in the artifact MKL_Headers_jll
  mkl = joinpath(@__DIR__, "mkl-include-2022.2.0-intel_8748", "include")
  wrapper("libmklsparse", ["$mkl/mkl_spblas.h"], optimized)
end

# If we want to use the file as a script with `julia wrapper.jl`
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
