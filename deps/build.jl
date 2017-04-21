if !haskey(ENV, "MKLROOT")
    error("please set the environmental variable MKLROOT to build MKLSparse.jl")
end

if Base.USE_BLAS64
    libmkl_path = joinpath(ENV["MKLROOT"], "lib", "intel64", "libmkl_rt")
else
    libmkl_path = joinpath(ENV["MKLROOT"], "lib", "ia32", "libmkl_rt")
end

try
    Base.Libdl.dlopen_e(libmkl_path)
catch
    error("failed to open library at $libmkl_path")
end


open(joinpath(@__DIR__, "deps.jl"), "w") do f
    print(f,
"""
# This is an auto-generated file; do not edit

# Macro to load a library
macro checked_lib(libname, path)
    Base.Libdl.dlopen_e(path) == C_NULL && error("Unable to load \\n\\n\$libname (\$path)\\n\\nPlease re-run Pkg.build(package), and restart Julia.")
    quote const \$(esc(libname)) = \$path end
end

# Load dependencies
@checked_lib libmkl_rt "$libmkl_path"
"""
)
end
