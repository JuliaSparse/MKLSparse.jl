function _check_transa(t::Char)
    if !(t in ('C', 'N', 'T'))
        throw(ArgumentError("transa: is '$t', must be 'N', 'T', or 'C'"))
    end
end

# Checks sizes for the multiplication C <- tA[A] * tB[B]
function _check_mat_mult_matvec(C, A, tA, B, tB)
    _size(t::Char, M::AbstractMatrix) = t == 'N' ? size(M) : reverse(size(M))
    _size(t::Char, V::AbstractVector) = t == 'N' ? (size(V, 1), 1) : (1, size(V, 1))
    _str(M::AbstractMatrix) = string("[", size(M, 1), ", ", size(M, 2), "]")
    _str(V::AbstractVector) = string("[", size(V, 1), "]")
    _t(t) = t == 'T' ? "ᵀ" : t == 'C' ? "ᴴ" : t == 'N' ? "" : "ERROR"

    mA, nA = _size(tA, A)
    mB, nB = _size(tB, B)
    mC, nC = _size('N', C)
    if nA != mB || mC != mA || nC != nB
        str = string("arrays had inconsistent dimensions for C = A", _t(tA), " * B", _t(tB), ": ",
                     _str(C), " = ", _str(A), _t(tA), " * ", _str(B), _t(tB))
        throw(DimensionMismatch(str))
    end
end

function cscmv!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, x::StridedVector{T},
                β::T, y::StridedVector{T}) where {T <: BlasFloat}
    _check_transa(transa)
    _check_mat_mult_matvec(y, A, transa, x, 'N')
    __counter[] += 1

    T == Float32    && (mkl_scscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    T == Float64    && (mkl_dcscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    T == ComplexF32 && (mkl_ccscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    T == ComplexF64 && (mkl_zcscmv(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y))
    return y
end

function cscmm!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, B::StridedMatrix{T},
                β::T, C::StridedMatrix{T}) where {T <: BlasFloat}
    _check_transa(transa)
    _check_mat_mult_matvec(C, A, transa, B, 'N')
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
                A::SparseMatrixCSC{T, BlasInt}, x::StridedVector{T},
                y::StridedVector{T}) where {T <: BlasFloat}
    n = checksquare(A)
    _check_transa(transa)
    _check_mat_mult_matvec(y, A, transa, x, 'N')
    __counter[] += 1

    T == Float32    && (mkl_scscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    T == Float64    && (mkl_dcscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    T == ComplexF32 && (mkl_ccscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    T == ComplexF64 && (mkl_zcscsv(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y))
    return y
end

function cscsm!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, B::StridedMatrix{T},
                C::StridedMatrix{T}) where {T <: BlasFloat}
    mB, nB = size(B)
    mC, nC = size(C)
    n = checksquare(A)
    _check_transa(transa)
    _check_mat_mult_matvec(C, A, transa, B, 'N')
    __counter[] += 1

    T == Float32    && (mkl_scscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    T == Float64    && (mkl_dcscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    T == ComplexF32 && (mkl_ccscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    T == ComplexF64 && (mkl_zcscsm(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC))
    return C
end
