module DSS
import Base.LinAlg: BlasInt, BlasFloat, chksquare, factorize,
       A_ldiv_B!, At_ldiv_B!, Ac_ldiv_B!, A_ldiv_B, At_ldiv_B, Ac_ldiv_B

include("dss_generator.jl")
include("matstruct.jl")

type DSSFactor{T}
    A::SparseMatrixCSC{T, BlasInt}
    handle::Vector{Int}
    n::Int # Size of matrix
end

function factorize{T<:BlasFloat}(A::SparseMatrixCSC{T, BlasInt})

    n = chksquare(A)
    mat_struct = MatrixSymStructure(A)
    # Convert from CSC -> CSR
    A = transpose(A)
    if issym(mat_struct)
        opt_struct = (T <: Complex ? MKL_DSS_SYMMETRIC_COMPLEX: MKL_DSS_SYMMETRIC)
        A = tril(A)
    else
        opt_struct = (T <: Complex ? MKL_DSS_NON_SYMMETRIC_COMPLEX: MKL_DSS_NON_SYMMETRIC)
    end

    handle = dss_create(T)
    dss_define_structure(handle, A.colptr, n, n, A.rowval,
                         length(A.nzval), opt_struct)
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
    F = DSSFactor(A, handle, n)
    finalizer(F, free!)
    F
end

function free!(F::DSSFactor)
    dss_delete(F.handle)
end

for mv in ((:A_ldiv_B! ),
           (:At_ldiv_B!),
           (:Ac_ldiv_B!))
    @eval begin
        function $(mv){T<:BlasFloat}(A::SparseMatrixCSC{T,BlasInt},
                                     B::StridedVecOrMat{T},
                                     X::StridedVecOrMat{T})
            F = factorize(A)
            return $(mv)(F, B, X)
        end
    end
end

for (mv, trans) in ((:A_ldiv_B!,  MKL_DSS_DEFAULTS),
                    (:At_ldiv_B!, MKL_DSS_TRANSPOSE_SOLVE),
                    (:Ac_ldiv_B!, MKL_DSS_CONJUGATE_SOLVE))
    @eval begin
        function $(mv){T<:BlasFloat}(F::DSSFactor,
                                     B::StridedVecOrMat{T},
                                     X::StridedVecOrMat{T})
            F.n == size(B,1) == size(X,1) || throw(DimensionMismatch())
            size(B,2) == size(X,2) || throw(DimensionMismatch())
            dss_solve!(F.handle, B, X, $(trans))
            return X
        end
    end
end

# Non mutating functions
for (mv, mv!) in ((:A_ldiv_B,  :A_ldiv_B!),
                  (:At_ldiv_B, :At_ldiv_B!),
                  (:Ac_ldiv_B, :Ac_ldiv_B!))
    @eval begin
        function $(mv){T<:BlasFloat}(F::DSSFactor,
                                     B::StridedVecOrMat{T})
            X = similar(B)
            return $(mv!)(F, B, X)
        end
    end

   @eval begin
        function $(mv){T<:BlasFloat}(A::SparseMatrixCSC{T,BlasInt},
                                     B::StridedVecOrMat{T})
            X = similar(B)
            return $(mv!)(A, B, X)
        end
    end
end

end # module
