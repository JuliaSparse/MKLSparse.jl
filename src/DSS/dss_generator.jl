include("mkl_dss_h.jl")

for (T, opt) in ((Union(Float64, Complex128), MKL_DSS_DEFAULTS),
                 (Union(Float32, Complex64), MKL_DSS_DEFAULTS+MKL_DSS_SINGLE_PRECISION))
    @eval begin
        function dss_create{S <: $T}(::Type{S})
            handle = Int[0]
            ccall(("dss_create", :libmkl_rt), BlasInt, (Ptr{Void}, Ptr{BlasInt}),
                        handle, &($(opt)))
            return handle
        end
    end
end

function dss_define_structure(handle::Vector{Int}, rowindex::Vector{BlasInt},
                              nrows::BlasInt, ncols::BlasInt, columns::Vector{BlasInt},
                              nnz::BlasInt, opt::Int=MKL_DSS_DEFAULTS)
    nrows == ncols || throw(DimensionMismatch())
    @errcheck ccall(("dss_define_structure", :libmkl_rt), BlasInt,
                (Ptr{Void}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                handle, &opt, rowindex, &nrows, &ncols, columns, &nnz)
end

function dss_reorder(handle::Vector{Int}, perm::Vector{BlasInt},
                     opt::Int=MKL_DSS_DEFAULTS)
    @errcheck ccall(("dss_reorder", :libmkl_rt), BlasInt,
                (Ptr{Void}, Ptr{BlasInt}, Ptr{BlasInt}),
                handle, &opt, perm)
end

for (mv, T) in ((:dss_factor_real, Float32),
                (:dss_factor_real, Float64),
                (:dss_factor_complex, Complex64),
                (:dss_factor_complex, Complex128))
    @eval begin
        function dss_factor!(handle::Vector{Int}, A::SparseMatrixCSC{$T, BlasInt}, opt::Int=MKL_DSS_DEFAULTS)
            @errcheck ccall(($(string(mv)), :libmkl_rt), BlasInt,
                        (Ptr{Void}, Ptr{BlasInt}, Ptr{$T}),
                        handle, &opt, A.nzval)
        end
    end
end

for (mv, T) in ((:dss_solve_real, Float32),
                (:dss_solve_real, Float64),
                (:dss_solve_complex, Complex64),
                (:dss_solve_complex, Complex128))
    @eval begin
        function dss_solve!(handle::Vector{Int},B::StridedVecOrMat{$T},
                            X::StridedVecOrMat{$T}, opt::Int=MKL_DSS_DEFAULTS)
            size(B) == size(X) || throw(DimensionMismatch())
            nrhs = size(B, 2)
            @errcheck ccall(($(string(mv)), :libmkl_rt), BlasInt,
                        (Ptr{Void}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T}),
                        handle, &opt, B, &nrhs, X)
        end
    end
end

function dss_delete(handle::Vector{Int}, opt::Int=MKL_DSS_DEFAULTS)
    @errcheck ccall(("dss_delete", :libmkl_rt), BlasInt,
                (Ptr{Void}, Ptr{BlasInt}),
                handle, &opt)
end
