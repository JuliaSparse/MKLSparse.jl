matdescra(A::LowerTriangular) = "TLNF"
matdescra(A::UpperTriangular) = "TUNF"
matdescra(A::Diagonal) = "DUNF"
matdescra(A::UnitLowerTriangular) = "TLUF"
matdescra(A::UnitUpperTriangular) = "TUUF"
matdescra(A::Symmetric) = string('S', A.uplo, 'N', 'F')
matdescra(A::Hermitian) = string('H', A.uplo, 'N', 'F')
matdescra(A::SparseMatrixCSC) = "GFNF"
matdescra(A::Transpose) = matdescra(A.parent)
matdescra(A::Adjoint) = matdescra(A.parent)

function cscmv!(transA::Char, α::T, matdescrA::String,
                A::AbstractSparseMatrix{T}, x::StridedVector{T},
                β::T, y::StridedVector{T}) where {T <: BlasFloat}
    check_trans(transA)
    check_mat_op_sizes(y, A, transA, x, 'N')

    mkl_call(Val{:mkl_TSmvI}(), typeof(A),
             transA, A.m, A.n, α, matdescrA,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y)
    return y
end

function cscmm!(transA::Char, α::T, matdescrA::String,
                A::SparseMatrixCSC{T}, B::StridedMatrix{T},
                β::T, C::StridedMatrix{T}) where {T <: BlasFloat}
    check_trans(transA)
    check_mat_op_sizes(C, A, transA, B, 'N')
    mB, nB = size(B)
    mC, nC = size(C)

    mkl_call(Val{:mkl_TSmmI}(), typeof(A),
             transA, A.m, nC, A.n, α, matdescrA,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC)
    return C
end

function cscsv!(transA::Char, α::T, matdescrA::String,
                A::SparseMatrixCSC{T}, x::StridedVector{T},
                y::StridedVector{T}) where {T <: BlasFloat}
    n = checksquare(A)
    check_trans(transA)
    check_mat_op_sizes(y, A, transA, x, 'N')

    mkl_call(Val{:mkl_TSsvI}(), typeof(A),
             transA, A.m, α, matdescrA,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y)
    return y
end

function cscsm!(transA::Char, α::T, matdescrA::String,
                A::SparseMatrixCSC{T}, B::StridedMatrix{T},
                C::StridedMatrix{T}) where {T <: BlasFloat}
    mB, nB = size(B)
    mC, nC = size(C)
    n = checksquare(A)
    check_trans(transA)
    check_mat_op_sizes(C, A, transA, B, 'N')

    mkl_call(Val{:mkl_TSsmI}(), typeof(A),
             transA, A.n, nC, α, matdescrA,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC)
    return C
end
