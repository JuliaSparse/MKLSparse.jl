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

# The increments to the `__counter` variable is for testing purposes

function cscmv!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, Int32}, x::StridedVector{T},
                β::T, y::StridedVector{T}) where {T <: BlasFloat}
    check_transa(transa)
    check_mat_op_sizes(y, A, transa, x, 'N')
    __counter[] += 1
    T == Float32    && (mkl_scscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    T == Float64    && (mkl_dcscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    T == ComplexF32 && (mkl_ccscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    T == ComplexF64 && (mkl_zcscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    return y
end

function cscmm!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, Int32}, B::StridedMatrix{T},
                β::T, C::StridedMatrix{T}) where {T <: BlasFloat}
    check_transa(transa)
    check_mat_op_sizes(C, A, transa, B, 'N')
    mB, nB = size(B)
    mC, nC = size(C)
    __counter[] += 1

    T == Float32    && (mkl_scscmm(transa, A.m, nC, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC))
    T == Float64    && (mkl_dcscmm(transa, A.m, nC, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC))
    T == ComplexF32 && (mkl_ccscmm(transa, A.m, nC, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC))
    T == ComplexF64 && (mkl_zcscmm(transa, A.m, nC, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC))
    return C
end

function cscsv!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, Int32}, x::StridedVector{T},
                y::StridedVector{T}) where {T <: BlasFloat}
    n = checksquare(A)
    check_transa(transa)
    check_mat_op_sizes(y, A, transa, x, 'N')
    __counter[] += 1

    T == Float32    && (mkl_scscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    T == Float64    && (mkl_dcscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    T == ComplexF32 && (mkl_ccscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    T == ComplexF64 && (mkl_zcscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    return y
end

function cscsm!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, Int32}, B::StridedMatrix{T},
                C::StridedMatrix{T}) where {T <: BlasFloat}
    mB, nB = size(B)
    mC, nC = size(C)
    n = checksquare(A)
    check_transa(transa)
    check_mat_op_sizes(C, A, transa, B, 'N')
    __counter[] += 1

    T == Float32    && (mkl_scscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    T == Float64    && (mkl_dcscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    T == ComplexF32 && (mkl_ccscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    T == ComplexF64 && (mkl_zcscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    return C
end
