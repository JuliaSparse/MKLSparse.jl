# The increments to the `__counter` variable is for testing purposes

function _check_transa(t::Char)
    if t == 'C'
        return SPARSE_OPERATION_CONJUGATE_TRANSPOSE
    elseif t == 'N'
        return SPARSE_OPERATION_NON_TRANSPOSE
    elseif t == 'T'
        return SPARSE_OPERATION_TRANSPOSE
    else
        error("transa: is '$t', must be 'N', 'T', or 'C'")
    end
end

"""
    _invert_transa(t::Char)

Return the inverted `sparse_operation_t` to that indicated by `t`

This is used when a `Ptr{MKLcsr}` derived from a `SparseMatrixCSC` is
passed as an argument to an MKL function.
"""
function _invert_transa(t::Char)
    if t == 'T'
        return SPARSE_OPERATION_NON_TRANSPOSE
    elseif t == 'N'
        return SPARSE_OPERATION_TRANSPOSE
    else
        error("transa: is '$t', must be 'N', or 'T'")
    end
end

mkl_size(t::Char, M::AbstractVecOrMat) = t == 'N' ? size(M) : reverse(size(M))
mkl_size_trans(t::Char, M::AbstractVecOrMat) = t == 'N' ? reverse(size(M)) : size(M)


# Checks sizes for the multiplication C <- A * B
function _check_mat_mult_matvec(C, A, B, tA)
    _size(v::AbstractMatrix) = size(v)
    _size(v::AbstractVector) = (size(v,1), 1)
    _str(v::AbstractMatrix) = string("[", size(v,1), ", ", size(v,2), "]")
    _str(v::AbstractVector) = string("[", size(v,1), "]")
    mA, nA = mkl_size(tA, A)
    mB, nB = _size(B)
    mC, nC = _size(C)
    if nA != mB || mC != mA || nC != nB
        t = ""
        if tA == 'T'; t = ".\'"; end
        if tA == 'C'; t = "\'"; end
        str = string("arrays had inconsistent dimensions for C <- A", t, " * B: ", _str(C), " <- ", _str(A), t, " * ", _str(B))
        throw(DimensionMismatch(str))
    end
end

for (mv, sv, mm, sm, syrkd, T) in (
    (:mkl_sparse_s_mv, :mkl_sparse_s_trsv, :mkl_sparse_s_mm, :mkl_sparse_s_trsm, :mkl_sparse_s_syrkd, :Float32),
    (:mkl_sparse_d_mv, :mkl_sparse_d_trsv, :mkl_sparse_d_mm, :mkl_sparse_d_trsm, :mkl_sparse_d_syrkd, :Float64),
    (:mkl_sparse_c_mv, :mkl_sparse_c_trsv, :mkl_sparse_c_mm, :mkl_sparse_c_trsm, :mkl_sparse_c_syrkd, :ComplexF32),
    (:mkl_sparse_z_mv, :mkl_sparse_z_trsv, :mkl_sparse_z_mm, :mkl_sparse_z_trsm, :mkl_sparse_z_syrkd, :ComplexF64))
@eval begin
function cscmv!(transa::Char, α::$T, desc::matrix_descr,
                A::SparseMatrixCSC{$T, BlasInt}, x::StridedVector{$T},
                β::$T, y::StridedVector{$T})
    op = _check_transa(transa)
    _check_mat_mult_matvec(y, A, x, transa)
    __counter[] += 1
    p = cscptr(A)
    ret = ccall(($(string(mv)), libmkl_rt), sparse_status_t,
        (sparse_operation_t, $T, Ptr{MKLcsc{$T}}, matrix_descr, Ptr{$T}, $T, Ptr{$T},),
        op, α, p, desc, x, β, y)
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
    _destroy!(p)
    return y
end

function cscmm!(transa::Char, α::$T, desc::matrix_descr,
                A::SparseMatrixCSC{$T, BlasInt}, B::StridedMatrix{$T},
                β::$T, C::StridedMatrix{$T})
    op = _check_transa(transa)
    _check_mat_mult_matvec(C, A, B, transa)
    mB, nB = size(B)
    mC, nC = size(C)
    p = cscptr(A)
    __counter[] += 1
    ret = ccall(($(string(mm)), libmkl_rt), sparse_status_t,
        (sparse_operation_t, $T, Ptr{MKLcsc{$T}}, matrix_descr,
        sparse_layout_t, Ptr{$T}, BlasInt, BlasInt, $T, Ptr{$T}, BlasInt),
        op, α, p, desc, SPARSE_LAYOUT_COLUMN_MAJOR, B, nB, mB, β, C, mC)
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(ret))
    _destroy!(p)
    return C
end

function cscsv!(transa::Char, α::$T, desc::matrix_descr,
                A::SparseMatrixCSC{$T, BlasInt}, x::StridedVector{$T},
                y::StridedVector{$T})
    n = checksquare(A)
    op = _check_transa(transa)
    _check_mat_mult_matvec(y, A, x, transa)
    p = cscptr(A)
    __counter[] += 1
    ret = ccall(
        ($(string(sv)), libmkl_rt), sparse_status_t,
        (sparse_operation_t, $T, Ptr{MKLcsc{$T}}, matrix_descr, Ptr{$T}, Ptr{$T},),
        op, α, p, desc, x, y,
    )
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(ret))
    _destroy!(p)
    return y
end

function cscsm!(transa::Char, α::$T, desc::matrix_descr,
                A::SparseMatrixCSC{$T, BlasInt}, B::StridedMatrix{$T},
                C::StridedMatrix{$T})
    mB, nB = size(B)
    mC, nC = size(C)
    n = checksquare(A)
    op = _check_transa(transa)
    _check_mat_mult_matvec(C, A, B, transa)
    p = cscptr(A)
    __counter[] += 1
    ret = ccall(
        ($(string(sm)), libmkl_rt), sparse_status_t,
        (sparse_operation_t, $T, Ptr{MKLcsc{$T}}, matrix_descr,
        sparse_layout_t, Ptr{$T}, BlasInt, BlasInt, Ptr{$T}, BlasInt,),
        op, α, p, desc, SPARSE_LAYOUT_COLUMN_MAJOR, B, nB, mB, C, mC,
    )
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(ret))
    _destroy!(p)
    return C
end

function cscsyrkd!(transa::Char, α::$T, desc::matrix_descr,
    A::SparseMatrixCSC{$T,BlasInt}, β::$T, C::StridedMatrix{$T})
    n = checksquare(C)
    op = _invert_transa(transa)
    mA, nA = size(A)
    n == (transa == 'N' ? mA : nA) || throw(DimensionMismatch)
    p = csrptrtr(A)
    __counter[] += 1
    ret = ccall(
        ($(string(syrkd)), libmkl_rt), sparse_status_t,
        (sparse_operation_t, Ptr{MKLcsc{$T}}, $T, $T, Ptr{$T}, sparse_layout_t, BlasInt,),
        op, p, α, β, C, SPARSE_LAYOUT_COLUMN_MAJOR, size(C, 1),
    )
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
    _destroy!(p)
    return C
end

end # @eval
end
