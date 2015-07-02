module DSS
import Base.LinAlg: BlasInt, BlasFloat, chksquare, factorize, show,
       A_ldiv_B!, At_ldiv_B!, Ac_ldiv_B!, A_ldiv_B, At_ldiv_B, Ac_ldiv_B

include("dss_generator.jl")
include("matstruct.jl")

type DSSFactor{T}
    A::SparseMatrixCSC{T, BlasInt}
    handle::Vector{Int}
    ftype::ASCIIString
    n::Int
end
show(io::IO, luf::DSSFactor) = print(io, "DSS $(luf.ftype) Factorization of a $(luf.n)-by-$(luf.n) sparse matrix")

function cholfact{T<:BlasFloat}(A::SparseMatrixCSC{T, BlasInt})
    chksquare(A)
    mat_struct = MatrixSymStructure(A)
    if ischolcand(mat_struct)
        return _cholfact(A, mat_struct)
    else
        throw(ArgumentError(string("matrix must be hermitian and have a positive diagonal",
                                   " to be a candidate for Cholesky factorization")))
    end
end

function ldltfact{T<:BlasFloat}(A::SparseMatrixCSC{T, BlasInt})
    chksquare(A)
    mat_struct = MatrixSymStructure(A)
    if ishermitian(mat_struct)
        return _ldltfact(A, mat_struct)
    else
        throw(ArgumentError("matrix must be symmetric/hermitian to have a LDLT factorization"))
    end
end

function lufact{T<:BlasFloat}(A::SparseMatrixCSC{T, BlasInt})
    chksquare(A)
    mat_struct = MatrixSymStructure(A)
    return _lufact(A, mat_struct)
end

for (mv, ftype, cm_struct, rm_struct) in
    ((:_cholfact, "Cholesky", MKL_DSS_HERMITIAN_POSITIVE_DEFINITE, MKL_DSS_POSITIVE_DEFINITE),
     (:_ldltfact, "LDLT", MKL_DSS_HERMITIAN_INDEFINITE,        MKL_DSS_INDEFINITE),
     (:_lufact,   "LU",   MKL_DSS_INDEFINITE,                  MKL_DSS_INDEFINITE))
    @eval begin
        function $(mv){T<:BlasFloat}(A::SparseMatrixCSC{T, BlasInt}, mat_struct::MatrixSymStructure)
            n = size(A,1)
            handle = dss_create(T)
            A = transpose(A)
            if issym(mat_struct)
                opt_struct = (T <: Complex ? MKL_DSS_SYMMETRIC_COMPLEX: MKL_DSS_SYMMETRIC)
                A = tril(A)
            else
                opt_struct = (T <: Complex ? MKL_DSS_NON_SYMMETRIC_COMPLEX: MKL_DSS_NON_SYMMETRIC)
            end
            dss_define_structure(handle, A.colptr, n, n, A.rowval,
                                 length(A.nzval), opt_struct)
            dss_reorder(handle, BlasInt[0])

            opt_factor = (T <: Complex ? $(cm_struct): $(rm_struct))
            dss_factor!(handle, A, opt_factor)

            F = DSSFactor(A, handle, $(ftype), n)
            finalizer(F, free!)
            F
        end
    end
end

function factorize{T<:BlasFloat}(A::SparseMatrixCSC{T, BlasInt})
    chksquare(A)
    mat_struct = MatrixSymStructure(A)
    if ischolcand(mat_struct)
        try
            return _cholfact(A, mat_struct)
        catch e
            # We should check if the error is actually a pos def error but
            # DSS seems to return the wrong error message. See #2
            isa(e, DSSError) || rethrow(e)
        end
    end
    if ishermitian(mat_struct)
        return _ldltfact(A, mat_struct)
    else
        return _lufact(A, mat_struct)
    end
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
