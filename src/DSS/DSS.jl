module DSS
import Base.LinAlg: BlasInt, BlasFloat, A_ldiv_B!, At_ldiv_B!, Ac_ldiv_B!

include("dss_generator.jl")
include("matstruct.jl")

for (mv, trans) in ((:A_ldiv_B!, MKL_DSS_TRANSPOSE_SOLVE),
                    (:At_ldiv_B!, 0),
                    (:Ac_ldiv_B!, 0))
    @eval begin
        function $(mv){T<:BlasFloat}(A::SparseMatrixCSC{T,BlasInt},
                                     B::StridedVecOrMat{T}, X::StridedVecOrMat{T})

            (n = size(A,1)) == size(A,2) || throw(DimensionMismatch())
            size(A,2) == size(B,1) == size(X,1) || throw(DimensionMismatch())
            size(B,2) == size(X,2) || throw(DimensionMismatch())

            mat_struct = MatrixSymStructure(A)

            if issym(mat_struct)
                opt_struct = (T <: Complex ? MKL_DSS_SYMMETRIC_COMPLEX: MKL_DSS_SYMMETRIC)
                A = tril(A)
            else
                opt_struct = (T <: Complex ? MKL_DSS_NON_SYMMETRIC_COMPLEX: MKL_DSS_NON_SYMMETRIC)
            end

            # Special case needed for CSC -> CSR in case of solving conj transpose
            if T <: Complex && $mv ==  $(:Ac_ldiv_B!)
                A = conj(A)
            end

            handle = dss_create(T)
            dss_define_structure(handle, A.colptr, n, n, A.rowval, length(A.nzval), opt_struct)
            dss_reorder(handle, BlasInt[0])

            if ischolcand(mat_struct)
                try
                    opt_factor = (T <: Complex ? MKL_DSS_HERMITIAN_POSITIVE_DEFINITE: MKL_DSS_POSITIVE_DEFINITE)
                    dss_factor!(handle, A, opt_factor)
                catch e
                    # We should check if the error is actually a pos def error but
                    # DSS seems to return the wrong error message. See #1
                    isa(e, MKL_DSS_Exception) || rethrow(e)
                    opt_factor = (T <: Complex ? MKL_DSS_HERMITIAN_INDEFINITE: MKL_DSS_INDEFINITE)
                    dss_factor!(handle, A, opt_factor)
                end
            else
                dss_factor!(handle, A, MKL_DSS_INDEFINITE)
            end

            dss_solve!(handle, B, X, $(trans))
            dss_delete(handle)
            return X
        end
    end
end

end # module
